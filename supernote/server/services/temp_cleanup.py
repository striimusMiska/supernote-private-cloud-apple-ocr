import asyncio
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

from ..constants import USER_DATA_BUCKET
from ..metrics import (
    TEMP_CLEANUP_BYTES_FREED_TOTAL,
    TEMP_CLEANUP_DURATION_SECONDS,
    TEMP_CLEANUP_FILES_REMOVED_TOTAL,
    TEMP_CLEANUP_RUNS_TOTAL,
)
from .file import FileService

logger = logging.getLogger(__name__)

# `LocalBlobStorage.put()` stages writes as `<hex>.tmp` under the shared temp
# directory (`FileService.temp_dir`, same path as `LocalBlobStorage.root /
# "temp"`) before an atomic rename.
_TMP_FILE_RE = re.compile(r"\.tmp$")

# `get_file_chunk_path()` (supernote/server/utils/paths.py) names each chunk
# of a multi-part upload `<object_name>.part.<n>`.
_CHUNK_FILE_RE = re.compile(r"\.part\.\d+$")


@dataclass
class TempCleanupResult:
    """Result of a single temp-storage cleanup pass."""

    tmp_files_removed: int = 0
    chunk_files_removed: int = 0
    bytes_freed: int = 0

    @property
    def total_removed(self) -> int:
        """Total number of files (temp staging + chunks) removed."""
        return self.tmp_files_removed + self.chunk_files_removed


def _scan_and_remove_stale(
    root: Path, name_pattern: re.Pattern[str], cutoff_epoch: float
) -> tuple[int, int]:
    """Synchronously remove stale files under `root` matching `name_pattern`.

    A file is considered stale (orphaned) once its mtime is older than
    `cutoff_epoch`. Runs in a worker thread via `asyncio.to_thread` since it
    performs blocking filesystem I/O (recursive directory walk + stat +
    unlink), following the same offload-to-thread pattern used elsewhere in
    this service layer for blocking work (e.g. `FileService._load_notebook_sync`).

    Returns (files_removed, bytes_freed).
    """
    if not root.exists():
        return 0, 0

    removed = 0
    bytes_freed = 0
    for path in root.rglob("*"):
        if not path.is_file() or not name_pattern.search(path.name):
            continue
        try:
            stat_result = path.stat()
        except OSError:
            # Already gone (e.g. removed by a concurrent run or the request
            # that owned it finally completing) -- not an error.
            continue

        if stat_result.st_mtime >= cutoff_epoch:
            continue

        try:
            path.unlink()
        except OSError as e:
            logger.warning(f"Failed to remove stale temp file {path}: {e}")
            continue

        removed += 1
        bytes_freed += stat_result.st_size

    return removed, bytes_freed


class TempStorageCleanupService:
    """Periodically deletes orphaned temp-staging files and abandoned upload chunks.

    Mirrors the `RecycleBinCleanupService` background-task pattern (a single
    asyncio task started on server startup, cancelled on shutdown, sleeping
    for `interval_seconds` between runs) but targets different storage debris
    with a different staleness signal:

    - `LocalBlobStorage.put()` (`supernote/server/services/blob.py`) stages
      every blob write at `<storage_root>/temp/<hex>.tmp` before an atomic
      rename to its final path. If the process crashes or is killed mid-write
      (before the rename), the `.tmp` file is orphaned with no DB record
      anywhere.
    - Chunked uploads (`handle_oss_upload_part` in
      `supernote/server/routes/oss.py`) store each chunk of a multi-part
      upload as its own blob (`<name>.part.<n>`). These are only merged and
      deleted once the final chunk arrives -- an interrupted or abandoned
      upload leaves them behind forever.

    Neither location has a DB record to check for staleness, so both are
    judged purely by filesystem mtime; `ttl_seconds` should be generous
    enough to never catch a write or upload that is merely slow or still in
    progress.
    """

    def __init__(
        self,
        file_service: FileService,
        ttl_seconds: int,
        interval_seconds: int,
        enabled: bool = True,
    ) -> None:
        self.file_service = file_service
        self.ttl_seconds = ttl_seconds
        self.interval_seconds = interval_seconds
        self.enabled = enabled
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the background cleanup loop, if enabled."""
        if not self.enabled:
            logger.info(
                "Temp storage cleanup job disabled "
                "(SUPERNOTE_TEMP_CLEANUP_ENABLED=false)."
            )
            return
        logger.info(
            "Starting temp storage cleanup job "
            f"(ttl_seconds={self.ttl_seconds}, "
            f"interval_seconds={self.interval_seconds})."
        )
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        """Stop the background cleanup loop."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _poll_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self.interval_seconds)
                await self.run_once()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in temp storage cleanup loop: {e}", exc_info=True)

    async def run_once(self) -> TempCleanupResult:
        """Run a single cleanup pass. Returns what was removed."""
        start_time = time.monotonic()
        try:
            cutoff_epoch = time.time() - self.ttl_seconds

            tmp_removed, tmp_bytes = await asyncio.to_thread(
                _scan_and_remove_stale,
                self.file_service.temp_dir,
                _TMP_FILE_RE,
                cutoff_epoch,
            )

            chunk_root = self.file_service.storage_root / USER_DATA_BUCKET
            chunk_removed, chunk_bytes = await asyncio.to_thread(
                _scan_and_remove_stale, chunk_root, _CHUNK_FILE_RE, cutoff_epoch
            )

            result = TempCleanupResult(
                tmp_files_removed=tmp_removed,
                chunk_files_removed=chunk_removed,
                bytes_freed=tmp_bytes + chunk_bytes,
            )
        except Exception:
            TEMP_CLEANUP_RUNS_TOTAL.labels(status="failure").inc()
            raise
        finally:
            TEMP_CLEANUP_DURATION_SECONDS.observe(time.monotonic() - start_time)

        TEMP_CLEANUP_RUNS_TOTAL.labels(status="success").inc()
        if result.tmp_files_removed:
            TEMP_CLEANUP_FILES_REMOVED_TOTAL.labels(kind="tmp").inc(
                result.tmp_files_removed
            )
        if result.chunk_files_removed:
            TEMP_CLEANUP_FILES_REMOVED_TOTAL.labels(kind="chunk").inc(
                result.chunk_files_removed
            )
        if result.bytes_freed:
            TEMP_CLEANUP_BYTES_FREED_TOTAL.inc(result.bytes_freed)

        if result.total_removed:
            logger.info(
                "Temp storage cleanup removed "
                f"{result.tmp_files_removed} orphaned temp file(s) and "
                f"{result.chunk_files_removed} orphaned upload chunk(s), "
                f"freeing {result.bytes_freed} byte(s)."
            )
        else:
            logger.debug("Temp storage cleanup found nothing to remove.")

        return result

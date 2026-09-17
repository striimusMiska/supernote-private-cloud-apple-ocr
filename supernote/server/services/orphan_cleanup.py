import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..constants import CACHE_BUCKET, USER_DATA_BUCKET
from ..db.models.file import UserFileDO
from ..db.models.note_processing import NotePageContentDO, SystemTaskDO
from ..db.models.summary import SummaryDO
from ..db.session import DatabaseSessionManager
from ..metrics import (
    ORPHAN_CLEANUP_DURATION_SECONDS,
    ORPHAN_CLEANUP_ITEMS_REMOVED_TOTAL,
    ORPHAN_CLEANUP_RUNS_TOTAL,
)
from ..utils.paths import get_page_png_path
from .blob import BlobStorage

logger = logging.getLogger(__name__)

# Root of the virtual filesystem. Not a real `UserFileDO` row, but every
# user's top-level files/folders point at it via `directory_id`.
ROOT_DIRECTORY_ID = 0


@dataclass
class OrphanCleanupStats:
    """Counts produced by a single orphan cleanup run.

    `*_found` counts everything currently unreachable from an active root
    (regardless of the retention window); `*_removed` counts what was
    actually past `retention_days` and permanently deleted this run.
    """

    stale_files_found: int = 0
    stale_folders_found: int = 0
    files_removed: int = 0
    folders_removed: int = 0
    note_page_content_removed: int = 0
    system_tasks_removed: int = 0
    summaries_removed: int = 0
    png_blobs_removed: int = 0
    source_blobs_removed: int = 0

    @property
    def total_removed(self) -> int:
        """Total rows/blobs removed across every table/artifact type."""
        return (
            self.files_removed
            + self.folders_removed
            + self.note_page_content_removed
            + self.system_tasks_removed
            + self.summaries_removed
            + self.png_blobs_removed
            + self.source_blobs_removed
        )


class OrphanCleanupService:
    """Periodically garbage-collects data no longer reachable from an active
    file tree.

    Two kinds of `UserFileDO` rows are considered stale:
      1. Rows with `is_active != "Y"` (soft-deleted files/folders).
      2. Rows that are individually active, but sit inside a folder chain
         where some ancestor is inactive (e.g. a bug or a direct DB edit left
         a folder inactive without cascading to its children).

    For stale `.note` files, associated derived data (`f_note_page_content`,
    `f_system_task`, `f_summary`, cached PNG conversion blobs) and the row
    itself are only *permanently* removed once the item has been unreachable
    for at least `retention_days`, to avoid racing with sync edge cases. The
    file's original source blob is deleted at the same time, unless another
    row still references the same `storage_key` (e.g. a copy).

    Mirrors the `RecycleBinCleanupService` asyncio poll-loop pattern: a
    single background task started on server startup, cancelled on shutdown,
    that sleeps for `interval_seconds` between runs. This is a separate,
    independent job from `RecycleBinCleanupService` -- that job purges
    recycle-bin entries after their retention window; this one is a
    reachability-based safety net that also catches stale data never routed
    through the recycle bin at all.
    """

    def __init__(
        self,
        session_manager: DatabaseSessionManager,
        blob_storage: BlobStorage,
        retention_days: int,
        interval_seconds: int,
        enabled: bool = True,
    ) -> None:
        self.session_manager = session_manager
        self.blob_storage = blob_storage
        self.retention_days = retention_days
        self.interval_seconds = interval_seconds
        self.enabled = enabled
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the background cleanup loop, if enabled."""
        if not self.enabled:
            logger.info(
                "Orphan cleanup job disabled (SUPERNOTE_ORPHAN_CLEANUP_ENABLED=false)."
            )
            return
        logger.info(
            "Starting orphan cleanup job "
            f"(retention_days={self.retention_days}, "
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
                logger.error(f"Error in orphan cleanup loop: {e}", exc_info=True)

    async def run_once(self) -> OrphanCleanupStats:
        """Run a single cleanup pass. Returns counts of what was removed."""
        start_time = time.monotonic()
        try:
            stats = await self._cleanup_stale_orphans()
        except Exception:
            ORPHAN_CLEANUP_RUNS_TOTAL.labels(status="failure").inc()
            raise
        finally:
            ORPHAN_CLEANUP_DURATION_SECONDS.observe(time.monotonic() - start_time)

        ORPHAN_CLEANUP_RUNS_TOTAL.labels(status="success").inc()

        for artifact_type, count in (
            ("file", stats.files_removed),
            ("folder", stats.folders_removed),
            ("note_page_content", stats.note_page_content_removed),
            ("system_task", stats.system_tasks_removed),
            ("summary", stats.summaries_removed),
            ("png_blob", stats.png_blobs_removed),
            ("source_blob", stats.source_blobs_removed),
        ):
            if count:
                ORPHAN_CLEANUP_ITEMS_REMOVED_TOTAL.labels(
                    artifact_type=artifact_type
                ).inc(count)

        if stats.total_removed:
            logger.info(
                "Orphan cleanup removed: "
                f"{stats.files_removed} file(s), "
                f"{stats.folders_removed} folder(s), "
                f"{stats.note_page_content_removed} note_page_content row(s), "
                f"{stats.system_tasks_removed} system_task row(s), "
                f"{stats.summaries_removed} summary row(s), "
                f"{stats.png_blobs_removed} PNG blob(s), "
                f"{stats.source_blobs_removed} source blob(s) "
                f"(found {stats.stale_files_found} stale file(s) and "
                f"{stats.stale_folders_found} stale folder(s) total, "
                f"retention_days={self.retention_days})."
            )
        else:
            logger.debug(
                "Orphan cleanup found "
                f"{stats.stale_files_found} stale file(s) and "
                f"{stats.stale_folders_found} stale folder(s), "
                "none past the retention window."
            )

        return stats

    async def _cleanup_stale_orphans(self) -> OrphanCleanupStats:
        """Find and permanently remove data unreachable from an active root.

        Runs across all users in one pass, using `directory_id` to build a
        reachability closure per user starting from the (shared, implicit)
        root directory ID `0`.
        """
        stats = OrphanCleanupStats()
        cutoff_ms = int(time.time() * 1000) - (self.retention_days * 86400 * 1000)

        async with self.session_manager.session() as session:
            result = await session.execute(select(UserFileDO))
            all_nodes = list(result.scalars().all())

            nodes_by_id: dict[int, UserFileDO] = {n.id: n for n in all_nodes}

            active_children_by_parent: dict[int, list[UserFileDO]] = defaultdict(list)
            for node in all_nodes:
                if node.is_active == "Y":
                    active_children_by_parent[node.directory_id].append(node)

            reachable: set[int] = set()
            queue: deque[int] = deque([ROOT_DIRECTORY_ID])
            while queue:
                parent_id = queue.popleft()
                for child in active_children_by_parent.get(parent_id, []):
                    if child.id in reachable:
                        continue
                    reachable.add(child.id)
                    queue.append(child.id)

            stale_nodes = [n for n in all_nodes if n.id not in reachable]
            stats.stale_files_found = sum(1 for n in stale_nodes if n.is_folder != "Y")
            stats.stale_folders_found = sum(
                1 for n in stale_nodes if n.is_folder == "Y"
            )

            eligible = [
                n for n in stale_nodes if _stale_since_ms(n, nodes_by_id) < cutoff_ms
            ]
            eligible_ids = {n.id for n in eligible}
            eligible_files = [n for n in eligible if n.is_folder != "Y"]
            eligible_folders = [n for n in eligible if n.is_folder == "Y"]

            for node in eligible_files:
                await self._remove_stale_file(session, node, eligible_ids, stats)

            for node in eligible_folders:
                await session.delete(node)
                stats.folders_removed += 1

            await session.commit()

        return stats

    async def _remove_stale_file(
        self,
        session: AsyncSession,
        node: UserFileDO,
        eligible_ids: set[int],
        stats: OrphanCleanupStats,
    ) -> None:
        """Delete derived data, blobs, and the row itself for one stale file."""
        # Count via a SELECT first (rather than relying on the DELETE
        # statement's `rowcount`, which some async DB-API drivers/stub
        # signatures don't expose reliably) so the returned stats are exact.
        page_ids_result = await session.execute(
            select(NotePageContentDO.page_id).where(
                NotePageContentDO.file_id == node.id
            )
        )
        note_page_content_rows = list(page_ids_result.scalars().all())
        page_ids = [pid for pid in note_page_content_rows if pid]
        stats.note_page_content_removed += len(note_page_content_rows)
        await session.execute(
            delete(NotePageContentDO).where(NotePageContentDO.file_id == node.id)
        )

        task_ids_result = await session.execute(
            select(SystemTaskDO.id).where(SystemTaskDO.file_id == node.id)
        )
        stats.system_tasks_removed += len(task_ids_result.scalars().all())
        await session.execute(
            delete(SystemTaskDO).where(SystemTaskDO.file_id == node.id)
        )

        summary_ids_result = await session.execute(
            select(SummaryDO.id).where(SummaryDO.file_id == node.id)
        )
        stats.summaries_removed += len(summary_ids_result.scalars().all())
        await session.execute(delete(SummaryDO).where(SummaryDO.file_id == node.id))

        for page_id in page_ids:
            png_path = get_page_png_path(node.id, page_id)
            try:
                await self.blob_storage.delete(CACHE_BUCKET, png_path)
                stats.png_blobs_removed += 1
            except Exception as e:
                logger.warning(
                    f"Failed to delete cached PNG for {node.id} page {page_id}: {e}"
                )

        if node.storage_key:
            still_referenced_stmt = (
                select(UserFileDO.id)
                .where(
                    UserFileDO.storage_key == node.storage_key,
                    UserFileDO.id.notin_(eligible_ids),
                )
                .limit(1)
            )
            still_referenced = await session.execute(still_referenced_stmt)
            if still_referenced.first() is None:
                try:
                    await self.blob_storage.delete(USER_DATA_BUCKET, node.storage_key)
                    stats.source_blobs_removed += 1
                except Exception as e:
                    logger.warning(
                        f"Failed to delete source blob for {node.id} "
                        f"({node.storage_key}): {e}"
                    )

        await session.delete(node)
        stats.files_removed += 1


def _stale_since_ms(node: UserFileDO, nodes_by_id: dict[int, UserFileDO]) -> int:
    """Best-effort timestamp (epoch ms) for when `node` became unreachable.

    For an individually-inactive node, this is its own `update_time` (set
    when `is_active` flips to "N"). For a node that's still active but sits
    under an inactive ancestor, this walks up the folder chain to the
    nearest inactive ancestor and uses *its* `update_time` instead, since
    that's when the chain actually stopped being reachable. Falls back to
    the node's own `update_time` if the chain is broken (e.g. a missing
    ancestor row), which is the more conservative (later) choice in
    practice.
    """
    if node.is_active != "Y":
        return node.update_time

    visited: set[int] = set()
    current = nodes_by_id.get(node.directory_id)
    while current is not None and current.id not in visited:
        if current.is_active != "Y":
            return current.update_time
        visited.add(current.id)
        current = nodes_by_id.get(current.directory_id)

    return node.update_time

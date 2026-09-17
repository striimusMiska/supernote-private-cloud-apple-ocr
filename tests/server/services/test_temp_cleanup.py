import asyncio
import os
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from supernote.server.constants import USER_DATA_BUCKET
from supernote.server.services.file import FileService
from supernote.server.services.temp_cleanup import TempStorageCleanupService
from supernote.server.utils.paths import get_file_chunk_path

# A generous TTL so "fresh" fixtures (written moments ago by the test) are
# never mistaken for orphaned/abandoned.
FRESH_TTL_SECONDS = 3600
# A TTL of 0 means "anything not written in the future is stale", used to
# exercise the removal path without needing to fake old mtimes.
STALE_TTL_SECONDS = 0


def _make_file_service(storage_root: Path) -> FileService:
    """Build a `FileService` backed by a real filesystem `storage_root`.

    The cleanup service only reads `temp_dir`/`storage_root` off of
    `FileService`; the DB/blob/user collaborators are irrelevant here, so
    they're mocked out.
    """
    return FileService(
        storage_root=storage_root,
        blob_storage=MagicMock(),
        user_service=MagicMock(),
        session_manager=MagicMock(),
    )


def _age_file(path: Path, age_seconds: float) -> None:
    """Backdate a file's mtime (and atime) by `age_seconds`."""
    old_time = time.time() - age_seconds
    os.utime(path, (old_time, old_time))


async def test_removes_orphaned_tmp_file_older_than_ttl(tmp_path: Path) -> None:
    file_service = _make_file_service(tmp_path)
    orphan = file_service.temp_dir / "deadbeefdeadbeef.tmp"
    orphan.write_bytes(b"partial write")
    _age_file(orphan, age_seconds=STALE_TTL_SECONDS + 10)

    service = TempStorageCleanupService(
        file_service, ttl_seconds=STALE_TTL_SECONDS, interval_seconds=86400
    )
    result = await service.run_once()

    assert result.tmp_files_removed == 1
    assert result.chunk_files_removed == 0
    assert result.bytes_freed == len(b"partial write")
    assert not orphan.exists()


async def test_leaves_fresh_tmp_file_alone(tmp_path: Path) -> None:
    file_service = _make_file_service(tmp_path)
    fresh = file_service.temp_dir / "cafebabecafebabe.tmp"
    fresh.write_bytes(b"still uploading")

    service = TempStorageCleanupService(
        file_service, ttl_seconds=FRESH_TTL_SECONDS, interval_seconds=86400
    )
    result = await service.run_once()

    assert result.tmp_files_removed == 0
    assert result.bytes_freed == 0
    assert fresh.exists()


async def test_removes_orphaned_chunk_blob_older_than_ttl(tmp_path: Path) -> None:
    file_service = _make_file_service(tmp_path)

    object_name = "some-inner-name"
    chunk_key = get_file_chunk_path(object_name, 1)
    chunk_path = (
        file_service.storage_root / USER_DATA_BUCKET / chunk_key[:2] / chunk_key
    )
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_path.write_bytes(b"abandoned chunk bytes")
    _age_file(chunk_path, age_seconds=STALE_TTL_SECONDS + 10)

    service = TempStorageCleanupService(
        file_service, ttl_seconds=STALE_TTL_SECONDS, interval_seconds=86400
    )
    result = await service.run_once()

    assert result.chunk_files_removed == 1
    assert result.tmp_files_removed == 0
    assert result.bytes_freed == len(b"abandoned chunk bytes")
    assert not chunk_path.exists()


async def test_leaves_fresh_chunk_and_finished_blob_alone(tmp_path: Path) -> None:
    file_service = _make_file_service(tmp_path)

    object_name = "in-progress-name"
    chunk_key = get_file_chunk_path(object_name, 2)
    chunk_path = (
        file_service.storage_root / USER_DATA_BUCKET / chunk_key[:2] / chunk_key
    )
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_path.write_bytes(b"still uploading")

    # A regular (non-chunk, non-temp) blob living in the same bucket must
    # never be touched by this job, no matter how old it is.
    finished_key = "finished-file"
    finished_path = (
        file_service.storage_root / USER_DATA_BUCKET / finished_key[:2] / finished_key
    )
    finished_path.parent.mkdir(parents=True, exist_ok=True)
    finished_path.write_bytes(b"done")
    _age_file(finished_path, age_seconds=STALE_TTL_SECONDS + 999999)

    service = TempStorageCleanupService(
        file_service, ttl_seconds=FRESH_TTL_SECONDS, interval_seconds=86400
    )
    result = await service.run_once()

    assert result.chunk_files_removed == 0
    assert result.total_removed == 0
    assert chunk_path.exists()
    assert finished_path.exists()


async def test_start_noop_when_disabled(tmp_path: Path) -> None:
    file_service = _make_file_service(tmp_path)
    service = TempStorageCleanupService(
        file_service, ttl_seconds=86400, interval_seconds=86400, enabled=False
    )

    await service.start()

    assert service._task is None
    await asyncio.sleep(0.05)

    # stop() is a no-op when nothing was started.
    await service.stop()


async def test_poll_loop_runs_periodically_and_stops_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # interval_seconds=0 keeps the loop tight (cooperatively yielding via
    # asyncio.sleep(0) each iteration) so the test doesn't need real delays.
    file_service = _make_file_service(tmp_path)
    service = TempStorageCleanupService(
        file_service, ttl_seconds=86400, interval_seconds=0, enabled=True
    )

    scan_calls = 0

    def counting_scan(*args: object, **kwargs: object) -> tuple[int, int]:
        nonlocal scan_calls
        scan_calls += 1
        return (0, 0)

    monkeypatch.setattr(
        "supernote.server.services.temp_cleanup._scan_and_remove_stale",
        counting_scan,
    )

    await service.start()
    try:
        await asyncio.sleep(0.05)
        # Each run_once() does two scans (temp dir + chunk bucket).
        assert scan_calls >= 4
    finally:
        await service.stop()

    calls_after_stop = scan_calls
    await asyncio.sleep(0.05)
    assert scan_calls == calls_after_stop


async def test_run_once_still_counts_failure_before_reraising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    file_service = _make_file_service(tmp_path)
    service = TempStorageCleanupService(
        file_service, ttl_seconds=86400, interval_seconds=86400
    )

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "supernote.server.services.temp_cleanup._scan_and_remove_stale", boom
    )

    with pytest.raises(RuntimeError, match="boom"):
        await service.run_once()

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select

from supernote.server.constants import CACHE_BUCKET, USER_DATA_BUCKET
from supernote.server.db.models.file import UserFileDO
from supernote.server.db.models.note_processing import NotePageContentDO, SystemTaskDO
from supernote.server.db.models.summary import SummaryDO
from supernote.server.db.session import DatabaseSessionManager
from supernote.server.services.blob import BlobStorage
from supernote.server.services.orphan_cleanup import (
    OrphanCleanupService,
    OrphanCleanupStats,
)
from supernote.server.services.vfs import VirtualFileSystem
from supernote.server.utils.paths import get_page_png_path

ONE_DAY_MS = 86400 * 1000


@pytest.fixture
def orphan_cleanup_service(
    session_manager: DatabaseSessionManager,
    blob_storage: BlobStorage,
) -> OrphanCleanupService:
    return OrphanCleanupService(
        session_manager,
        blob_storage,
        retention_days=30,
        interval_seconds=86400,
    )


async def _make_inactive_folder_with_active_child_note(
    session_manager: DatabaseSessionManager,
    blob_storage: BlobStorage,
    user_id: int,
    *,
    folder_inactive_days_ago: int,
) -> tuple[int, int, str, str]:
    """Create `folder/note.note` where the folder is inactive but the note
    itself is still individually active (simulating a cascade miss, e.g. a
    bug or a direct DB edit that didn't cascade `is_active` to children).

    Returns (folder_id, note_id, note_storage_key, note_page_id).
    """
    storage_key = f"orphan-note-{user_id}"
    await blob_storage.put(USER_DATA_BUCKET, storage_key, b"note bytes")

    page_id = "page-0"

    async with session_manager.session() as session:
        vfs = VirtualFileSystem(session)
        folder = await vfs.create_directory(user_id, 0, "Notebooks")
        note = await vfs.create_or_update_file(
            user_id,
            folder.id,
            "note.note",
            size=10,
            md5="h",
            storage_key=storage_key,
        )
        folder_id = folder.id
        note_id = note.id

    png_path = get_page_png_path(note_id, page_id)
    await blob_storage.put(CACHE_BUCKET, png_path, b"png bytes")

    async with session_manager.session() as session:
        session.add(
            NotePageContentDO(
                file_id=note_id, page_index=0, page_id=page_id, content_hash="h1"
            )
        )
        session.add(
            SystemTaskDO(
                file_id=note_id, task_type="OCR", key=page_id, status="COMPLETED"
            )
        )
        session.add(
            SummaryDO(
                user_id=user_id,
                file_id=note_id,
                unique_identifier=f"summary-{note_id}",
                content="summary text",
            )
        )
        await session.commit()

    # Flip the folder inactive directly (bypassing `vfs.delete_node`'s
    # cascade), and backdate it, to simulate the reachability-break case the
    # scheduled job is meant to catch.
    async with session_manager.session() as session:
        result = await session.execute(
            select(UserFileDO).where(UserFileDO.id == folder_id)
        )
        folder_do = result.scalar_one()
        folder_do.is_active = "N"
        folder_do.update_time = int(time.time() * 1000) - (
            folder_inactive_days_ago * ONE_DAY_MS
        )
        await session.commit()

    return folder_id, note_id, storage_key, page_id


async def test_cleanup_removes_active_file_under_inactive_parent_folder(
    orphan_cleanup_service: OrphanCleanupService,
    session_manager: DatabaseSessionManager,
    blob_storage: BlobStorage,
    create_test_user: None,
    test_user_id: int,
) -> None:
    """Acceptance criterion: an active file under an inactive parent folder
    is treated as stale and its derived data + row + blobs are removed once
    past the retention window."""
    (
        folder_id,
        note_id,
        storage_key,
        page_id,
    ) = await _make_inactive_folder_with_active_child_note(
        session_manager,
        blob_storage,
        test_user_id,
        folder_inactive_days_ago=31,
    )

    stats = await orphan_cleanup_service.run_once()

    assert stats.files_removed == 1
    assert stats.folders_removed == 1
    assert stats.note_page_content_removed == 1
    assert stats.system_tasks_removed == 1
    assert stats.summaries_removed == 1
    assert stats.png_blobs_removed == 1
    assert stats.source_blobs_removed == 1
    assert stats.total_removed == 7

    async with session_manager.session() as session:
        assert (
            await session.execute(select(UserFileDO).where(UserFileDO.id == note_id))
        ).scalar_one_or_none() is None
        assert (
            await session.execute(select(UserFileDO).where(UserFileDO.id == folder_id))
        ).scalar_one_or_none() is None
        assert (
            await session.execute(
                select(NotePageContentDO).where(NotePageContentDO.file_id == note_id)
            )
        ).scalar_one_or_none() is None
        assert (
            await session.execute(
                select(SystemTaskDO).where(SystemTaskDO.file_id == note_id)
            )
        ).scalar_one_or_none() is None
        assert (
            await session.execute(select(SummaryDO).where(SummaryDO.file_id == note_id))
        ).scalar_one_or_none() is None

    assert await blob_storage.exists(USER_DATA_BUCKET, storage_key) is False
    assert (
        await blob_storage.exists(CACHE_BUCKET, get_page_png_path(note_id, page_id))
        is False
    )


async def test_cleanup_respects_retention_window(
    orphan_cleanup_service: OrphanCleanupService,
    session_manager: DatabaseSessionManager,
    blob_storage: BlobStorage,
    create_test_user: None,
    test_user_id: int,
) -> None:
    """A stale file/folder that hasn't been unreachable long enough is left
    alone (found, but not removed) -- this is the retention/safety window."""
    (
        folder_id,
        note_id,
        storage_key,
        page_id,
    ) = await _make_inactive_folder_with_active_child_note(
        session_manager,
        blob_storage,
        test_user_id,
        folder_inactive_days_ago=1,
    )

    stats = await orphan_cleanup_service.run_once()

    assert stats.stale_files_found == 1
    assert stats.stale_folders_found == 1
    assert stats.total_removed == 0

    async with session_manager.session() as session:
        assert (
            await session.execute(select(UserFileDO).where(UserFileDO.id == note_id))
        ).scalar_one_or_none() is not None
        assert (
            await session.execute(
                select(NotePageContentDO).where(NotePageContentDO.file_id == note_id)
            )
        ).scalar_one_or_none() is not None

    assert await blob_storage.exists(USER_DATA_BUCKET, storage_key) is True


async def test_cleanup_does_not_touch_fully_active_tree(
    orphan_cleanup_service: OrphanCleanupService,
    session_manager: DatabaseSessionManager,
    blob_storage: BlobStorage,
    create_test_user: None,
    test_user_id: int,
) -> None:
    """Visible/active files and folders are never touched."""
    storage_key = "active-note-key"
    await blob_storage.put(USER_DATA_BUCKET, storage_key, b"bytes")

    async with session_manager.session() as session:
        vfs = VirtualFileSystem(session)
        folder = await vfs.create_directory(test_user_id, 0, "Active Folder")
        note = await vfs.create_or_update_file(
            test_user_id,
            folder.id,
            "active.note",
            size=10,
            md5="h",
            storage_key=storage_key,
        )
        note_id = note.id
        session.add(
            NotePageContentDO(
                file_id=note_id, page_index=0, page_id="p0", content_hash="h1"
            )
        )
        await session.commit()

    stats = await orphan_cleanup_service.run_once()

    assert stats.stale_files_found == 0
    assert stats.stale_folders_found == 0
    assert stats.total_removed == 0

    async with session_manager.session() as session:
        assert (
            await session.execute(
                select(NotePageContentDO).where(NotePageContentDO.file_id == note_id)
            )
        ).scalar_one_or_none() is not None

    assert await blob_storage.exists(USER_DATA_BUCKET, storage_key) is True


async def test_cleanup_is_idempotent(
    orphan_cleanup_service: OrphanCleanupService,
    session_manager: DatabaseSessionManager,
    blob_storage: BlobStorage,
    create_test_user: None,
    test_user_id: int,
) -> None:
    """Running the cleanup job twice in a row is harmless."""
    await _make_inactive_folder_with_active_child_note(
        session_manager,
        blob_storage,
        test_user_id,
        folder_inactive_days_ago=31,
    )

    first = await orphan_cleanup_service.run_once()
    assert first.total_removed == 7

    second = await orphan_cleanup_service.run_once()
    assert second.total_removed == 0
    assert second.stale_files_found == 0
    assert second.stale_folders_found == 0


async def test_cleanup_keeps_blob_shared_with_an_active_copy(
    orphan_cleanup_service: OrphanCleanupService,
    session_manager: DatabaseSessionManager,
    blob_storage: BlobStorage,
    create_test_user: None,
    test_user_id: int,
) -> None:
    """A stale file's source blob survives if another (active) row still
    references the same `storage_key` (e.g. a copy made via `copy_node`)."""
    shared_key = "shared-orphan-key"
    await blob_storage.put(USER_DATA_BUCKET, shared_key, b"shared bytes")

    async with session_manager.session() as session:
        vfs = VirtualFileSystem(session)
        stale_note = await vfs.create_or_update_file(
            test_user_id, 0, "stale.note", size=10, md5="h", storage_key=shared_key
        )
        active_copy = await vfs.create_or_update_file(
            test_user_id,
            0,
            "active-copy.note",
            size=10,
            md5="h",
            storage_key=shared_key,
        )
        stale_note_id = stale_note.id
        active_copy_id = active_copy.id

    async with session_manager.session() as session:
        result = await session.execute(
            select(UserFileDO).where(UserFileDO.id == stale_note_id)
        )
        node = result.scalar_one()
        node.is_active = "N"
        node.update_time = int(time.time() * 1000) - (31 * ONE_DAY_MS)
        await session.commit()

    stats = await orphan_cleanup_service.run_once()

    assert stats.files_removed == 1
    assert stats.source_blobs_removed == 0
    assert await blob_storage.exists(USER_DATA_BUCKET, shared_key) is True

    async with session_manager.session() as session:
        assert (
            await session.execute(
                select(UserFileDO).where(UserFileDO.id == active_copy_id)
            )
        ).scalar_one_or_none() is not None


def _mock_session_manager() -> AsyncMock:
    return AsyncMock()


async def test_start_noop_when_disabled() -> None:
    service = OrphanCleanupService(
        _mock_session_manager(),
        AsyncMock(),
        retention_days=30,
        interval_seconds=86400,
        enabled=False,
    )

    await service.start()

    assert service._task is None
    await asyncio.sleep(0.05)

    await service.stop()


async def test_poll_loop_runs_periodically_and_stops_cleanly() -> None:
    service = OrphanCleanupService(
        _mock_session_manager(),
        AsyncMock(),
        retention_days=30,
        interval_seconds=0,
        enabled=True,
    )

    with patch.object(
        service,
        "_cleanup_stale_orphans",
        new=AsyncMock(return_value=OrphanCleanupStats()),
    ) as mock_cleanup:
        await service.start()
        try:
            await asyncio.sleep(0.05)
            assert mock_cleanup.await_count >= 2
        finally:
            await service.stop()

        call_count_after_stop = mock_cleanup.await_count
        await asyncio.sleep(0.05)
        assert mock_cleanup.await_count == call_count_after_stop


async def test_run_once_still_counts_failure_before_reraising(
    orphan_cleanup_service: OrphanCleanupService,
) -> None:
    with patch.object(
        orphan_cleanup_service,
        "_cleanup_stale_orphans",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            await orphan_cleanup_service.run_once()

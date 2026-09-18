"""Shared helpers for building synthetic .spd files in tests.

No real or binary .spd fixture files are checked in; every test builds a
minimal synthetic sqlite database on the fly.
"""

import io
import sqlite3
from pathlib import Path

from PIL import Image

from supernote.spd.reader import TILE_ID_ORIGIN, TILE_ID_STRIDE, TILE_SIZE

__all__ = ["make_tile_png", "tid_for", "write_spd_file"]


def make_tile_png(color: tuple[int, int, int, int] = (255, 0, 0, 255)) -> bytes:
    """Return PNG-encoded bytes for a solid-color TILE_SIZE x TILE_SIZE RGBA tile."""
    img = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def tid_for(row: int, col: int) -> int:
    """Compute a tid whose tile_id_to_row_col() decodes back to (row, col).

    This is the inverse of the row/col formula: `col` may be negative (to
    exercise the wraparound branch in tile_id_to_row_col) as long as it stays
    within (-TILE_ID_STRIDE // 2, TILE_ID_STRIDE // 2).
    """
    return TILE_ID_ORIGIN + row * TILE_ID_STRIDE + col


def write_spd_file(
    path: Path,
    fmt_ver: str | None = "2",
    width: str | None = None,
    height: str | None = None,
    layers: dict[int, dict[int, bytes]] | None = None,
    with_config_table: bool = True,
) -> Path:
    """Create a synthetic .spd sqlite file at `path` and return it.

    Parameters
    ----------
    fmt_ver : str | None
        Value stored for the `fmt_ver` config key; omitted entirely if None.
    width, height : str | None
        Values stored for `surface.width` / `surface.height`; omitted if None.
    layers : dict[int, dict[int, bytes]] | None
        Maps layer_id -> {tid: png_bytes} used to create `surface_<id>` tables.
    with_config_table : bool
        When False, no `config` table is created at all.
    """
    conn = sqlite3.connect(str(path))
    try:
        if with_config_table:
            conn.execute("CREATE TABLE config (name TEXT, value TEXT)")
            rows = []
            if fmt_ver is not None:
                rows.append(("fmt_ver", fmt_ver))
            if width is not None:
                rows.append(("surface.width", width))
            if height is not None:
                rows.append(("surface.height", height))
            conn.executemany("INSERT INTO config (name, value) VALUES (?, ?)", rows)

        for layer_id, tiles in (layers or {}).items():
            conn.execute(f'CREATE TABLE "surface_{layer_id}" (tid INTEGER, tile BLOB)')
            conn.executemany(
                f'INSERT INTO "surface_{layer_id}" (tid, tile) VALUES (?, ?)',
                list(tiles.items()),
            )
        conn.commit()
    finally:
        conn.close()
    return path

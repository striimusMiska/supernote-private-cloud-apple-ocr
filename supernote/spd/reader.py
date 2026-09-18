"""Parser for Supernote Atelier .spd drawing files.

A .spd file is a SQLite database with a ``config`` table of metadata and zero
or more ``surface_<layer_id>`` tables, each holding the PNG-encoded 128x128
ink tiles for one drawing layer.
"""

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .exceptions import CorruptSpdFile, UnsupportedSpdFormat

#: Pixel size (both dimensions) of a single ink tile.
TILE_SIZE = 128

#: Number of tile ids per grid row before wrapping into the next row.
TILE_ID_STRIDE = 4096

#: Empirically-determined offset that maps a tile id to grid row/col 0.
TILE_ID_ORIGIN = 7976857

#: Layer id reserved for the background tracing template, if present. When
#: this layer table exists and has rows, it must be composited bottommost.
REFERENCE_LAYER_ID = 9999

#: Default canvas dimensions used when the config metadata doesn't specify
#: a usable width/height.
DEFAULT_WIDTH = 1404
DEFAULT_HEIGHT = 1872

_SUPPORTED_FORMAT_VERSION = "2"
_SURFACE_TABLE_RE = re.compile(r"^surface_(\d+)$")


def tile_id_to_row_col(tid: int) -> tuple[int, int]:
    """Convert a tile id into its (row, col) position in the tile grid."""
    offset = tid - TILE_ID_ORIGIN
    col = offset % TILE_ID_STRIDE
    row = offset // TILE_ID_STRIDE
    if col >= TILE_ID_STRIDE // 2:
        col -= TILE_ID_STRIDE
        row += 1
    return row, col


@dataclass
class SpdLayer:
    """One drawing layer: its numeric id and its tiles, keyed by tile id."""

    layer_id: int
    tiles: dict[int, bytes]


@dataclass
class SpdDrawing:
    """A fully-parsed .spd drawing, with layers ordered bottom-to-top."""

    layers: list[SpdLayer]
    width: int
    height: int


def load_spd(path: str | Path) -> SpdDrawing:
    """Parse a .spd file from a filesystem path.

    A real filesystem path is required (not bytes/BytesIO) because sqlite3
    needs to open the file directly.

    Raises
    ------
    UnsupportedSpdFormat
        If the config table's ``fmt_ver`` value is missing or not ``"2"``.
    CorruptSpdFile
        If the file isn't a valid sqlite database, or lacks the ``config``
        table.
    """
    conn = sqlite3.connect(str(path))
    # The `value` column of `config` has TEXT affinity but can hold raw
    # bytes (e.g. a JPEG thumbnail), so rows must be fetched as raw bytes
    # and only the specific keys we care about decoded as UTF-8 text.
    conn.text_factory = bytes
    try:
        config = _read_config(conn)

        fmt_ver = config.get("fmt_ver")
        if fmt_ver is None:
            raise UnsupportedSpdFormat("missing fmt_ver in .spd config table")
        if fmt_ver != _SUPPORTED_FORMAT_VERSION:
            raise UnsupportedSpdFormat(f"unsupported .spd format version: {fmt_ver!r}")

        width = _parse_dimension(config.get("surface.width"), DEFAULT_WIDTH)
        height = _parse_dimension(config.get("surface.height"), DEFAULT_HEIGHT)

        layers = _read_layers(conn)
    finally:
        conn.close()

    return SpdDrawing(layers=layers, width=width, height=height)


def _decode_text(value: bytes) -> str | None:
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _read_config(conn: sqlite3.Connection) -> dict[str, str]:
    try:
        rows = conn.execute("SELECT name, value FROM config").fetchall()
    except sqlite3.DatabaseError as err:
        raise CorruptSpdFile(f"not a valid .spd file: {err}") from err

    config: dict[str, str] = {}
    for name, value in rows:
        if not isinstance(name, bytes) or not isinstance(value, bytes):
            continue
        key = _decode_text(name)
        if key is None:
            continue
        text_value = _decode_text(value)
        if text_value is None:
            # Not text (e.g. a thumbnail blob) - not one of the keys we read.
            continue
        config[key] = text_value
    return config


def _parse_dimension(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(float(value))
    except (ValueError, OverflowError):
        return default
    return parsed if parsed > 0 else default


def _list_surface_layer_ids(conn: sqlite3.Connection) -> list[int]:
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    except sqlite3.DatabaseError as err:
        raise CorruptSpdFile(f"could not list tables in .spd file: {err}") from err

    layer_ids = []
    for (name,) in rows:
        if not isinstance(name, bytes):
            continue
        table_name = _decode_text(name)
        if table_name is None:
            continue
        match = _SURFACE_TABLE_RE.match(table_name)
        if match:
            layer_ids.append(int(match.group(1)))
    return layer_ids


def _read_layer_tiles(conn: sqlite3.Connection, layer_id: int) -> dict[int, bytes]:
    try:
        rows = conn.execute(f'SELECT tid, tile FROM "surface_{layer_id}"').fetchall()
    except sqlite3.DatabaseError as err:
        raise CorruptSpdFile(
            f"could not read layer table surface_{layer_id}: {err}"
        ) from err

    tiles: dict[int, bytes] = {}
    for tid, tile in rows:
        if tile is None:
            continue
        tiles[int(tid)] = bytes(tile)
    return tiles


def _read_layers(conn: sqlite3.Connection) -> list[SpdLayer]:
    layer_ids = _list_surface_layer_ids(conn)
    # The reference/tracing-template layer (9999), when present, is always
    # composited bottommost; every other layer is ordered ascending by id
    # (higher id drawn later, i.e. on top).
    ordered_ids = sorted(layer_ids, key=lambda lid: (lid != REFERENCE_LAYER_ID, lid))
    return [
        SpdLayer(layer_id=lid, tiles=_read_layer_tiles(conn, lid))
        for lid in ordered_ids
    ]

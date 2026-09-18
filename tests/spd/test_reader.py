"""Tests for supernote.spd.reader."""

from pathlib import Path

import pytest

from supernote.spd.exceptions import CorruptSpdFile, UnsupportedSpdFormat
from supernote.spd.reader import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    TILE_ID_STRIDE,
    load_spd,
    tile_id_to_row_col,
)

from .conftest import make_tile_png, tid_for, write_spd_file


def test_load_spd_parses_metadata_and_single_layer(tmp_path: Path) -> None:
    tile = make_tile_png()
    tid = tid_for(row=2, col=3)
    path = write_spd_file(
        tmp_path / "single_layer.spd",
        width="200.0",
        height="300.0",
        layers={1: {tid: tile}},
    )

    drawing = load_spd(path)

    assert drawing.width == 200
    assert drawing.height == 300
    assert len(drawing.layers) == 1
    layer = drawing.layers[0]
    assert layer.layer_id == 1
    assert layer.tiles == {tid: tile}


def test_load_spd_defaults_dimensions_when_zero_or_missing(tmp_path: Path) -> None:
    # width/height explicitly "0" -> default
    path_zero = write_spd_file(tmp_path / "zero.spd", width="0.0", height="0.0")
    drawing_zero = load_spd(path_zero)
    assert drawing_zero.width == DEFAULT_WIDTH
    assert drawing_zero.height == DEFAULT_HEIGHT

    # width/height missing entirely -> default
    path_missing = write_spd_file(tmp_path / "missing.spd")
    drawing_missing = load_spd(path_missing)
    assert drawing_missing.width == DEFAULT_WIDTH
    assert drawing_missing.height == DEFAULT_HEIGHT


def test_load_spd_orders_reference_layer_bottommost(tmp_path: Path) -> None:
    tile = make_tile_png()
    path = write_spd_file(
        tmp_path / "multi_layer.spd",
        layers={
            5: {tid_for(0, 0): tile},
            9999: {tid_for(0, 0): tile},
            1: {tid_for(0, 0): tile},
        },
    )

    drawing = load_spd(path)

    assert [layer.layer_id for layer in drawing.layers] == [9999, 1, 5]


def test_load_spd_zero_tiles_is_not_an_error(tmp_path: Path) -> None:
    path = write_spd_file(tmp_path / "blank.spd", width="500.0", height="500.0")

    drawing = load_spd(path)

    assert drawing.layers == []
    assert drawing.width == 500
    assert drawing.height == 500


def test_load_spd_defaults_dimensions_when_non_finite(tmp_path: Path) -> None:
    # "inf" parses as a float but int(float("inf")) raises OverflowError, not
    # ValueError - must still fall back to the default rather than propagating.
    path = write_spd_file(tmp_path / "infinite.spd", width="inf", height="-inf")

    drawing = load_spd(path)

    assert drawing.width == DEFAULT_WIDTH
    assert drawing.height == DEFAULT_HEIGHT


def test_load_spd_missing_fmt_ver_raises(tmp_path: Path) -> None:
    path = write_spd_file(tmp_path / "no_fmt_ver.spd", fmt_ver=None)

    with pytest.raises(UnsupportedSpdFormat):
        load_spd(path)


def test_load_spd_wrong_fmt_ver_raises(tmp_path: Path) -> None:
    path = write_spd_file(tmp_path / "wrong_fmt_ver.spd", fmt_ver="1")

    with pytest.raises(UnsupportedSpdFormat):
        load_spd(path)


def test_load_spd_missing_config_table_raises_corrupt(tmp_path: Path) -> None:
    path = write_spd_file(tmp_path / "no_config.spd", with_config_table=False)

    with pytest.raises(CorruptSpdFile):
        load_spd(path)


def test_load_spd_not_a_sqlite_database_raises_corrupt(tmp_path: Path) -> None:
    path = tmp_path / "garbage.spd"
    path.write_bytes(b"this is not a sqlite database, just some random bytes")

    with pytest.raises(CorruptSpdFile):
        load_spd(path)


def test_tile_id_to_row_col_negative_wraparound() -> None:
    tid = tid_for(row=3, col=-100)

    row, col = tile_id_to_row_col(tid)

    assert (row, col) == (3, -100)


def test_tile_id_to_row_col_positive_no_wraparound() -> None:
    tid = tid_for(row=7, col=TILE_ID_STRIDE // 2 - 1)

    row, col = tile_id_to_row_col(tid)

    assert (row, col) == (7, TILE_ID_STRIDE // 2 - 1)

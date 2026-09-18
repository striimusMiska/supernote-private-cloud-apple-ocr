"""Tests for supernote.spd.converter."""

from pathlib import Path

import pytest

from supernote.spd.converter import (
    convert_to_pdf_bytes,
    convert_to_png_bytes,
    render_image,
)
from supernote.spd.exceptions import CorruptSpdFile
from supernote.spd.reader import TILE_SIZE, SpdDrawing, SpdLayer

from .conftest import make_tile_png, tid_for, write_spd_file

WHITE = (255, 255, 255)


def test_render_image_single_layer_canvas_grows_to_fit_content() -> None:
    tile = make_tile_png((255, 0, 0, 255))
    tid = tid_for(row=0, col=0)
    # Metadata size is smaller than the tile content, so the canvas must grow
    # to fit it exactly, and the sole tile lands at the origin (no slack to
    # center within).
    drawing = SpdDrawing(
        layers=[SpdLayer(layer_id=1, tiles={tid: tile})],
        width=10,
        height=10,
    )

    image = render_image(drawing)

    assert image.size == (TILE_SIZE, TILE_SIZE)
    assert image.mode == "RGB"
    assert image.getpixel((1, 1)) == (255, 0, 0)
    assert image.getpixel((TILE_SIZE - 2, TILE_SIZE - 2)) == (255, 0, 0)


def test_render_image_single_layer_centers_content_in_larger_canvas() -> None:
    tile = make_tile_png((255, 0, 0, 255))
    tid = tid_for(row=0, col=0)
    drawing = SpdDrawing(
        layers=[SpdLayer(layer_id=1, tiles={tid: tile})],
        width=500,
        height=500,
    )

    image = render_image(drawing)

    assert image.size == (500, 500)
    assert image.mode == "RGB"

    # The one 128x128 tile is centered within the larger metadata canvas.
    offset = (500 - TILE_SIZE) // 2
    assert image.getpixel((offset + 1, offset + 1)) == (255, 0, 0)
    # Well outside the tile, the canvas stays white.
    assert image.getpixel((0, 0)) == WHITE


def test_render_image_higher_id_layer_overlays_lower_id() -> None:
    tid = tid_for(row=0, col=0)
    red_tile = make_tile_png((255, 0, 0, 255))
    blue_tile = make_tile_png((0, 0, 255, 255))
    drawing = SpdDrawing(
        layers=[
            SpdLayer(layer_id=1, tiles={tid: red_tile}),
            SpdLayer(layer_id=2, tiles={tid: blue_tile}),
        ],
        width=TILE_SIZE,
        height=TILE_SIZE,
    )

    image = render_image(drawing)

    assert image.getpixel((64, 64)) == (0, 0, 255)


def test_render_image_reference_layer_9999_is_bottommost() -> None:
    tid = tid_for(row=0, col=0)
    reference_tile = make_tile_png((255, 0, 0, 255))
    real_layer_tile = make_tile_png((0, 0, 255, 255))
    # Layers list intentionally mirrors reader's bottom-to-top ordering,
    # where 9999 is placed first despite being numerically largest.
    drawing = SpdDrawing(
        layers=[
            SpdLayer(layer_id=9999, tiles={tid: reference_tile}),
            SpdLayer(layer_id=1, tiles={tid: real_layer_tile}),
        ],
        width=TILE_SIZE,
        height=TILE_SIZE,
    )

    image = render_image(drawing)

    assert image.getpixel((64, 64)) == (0, 0, 255)


def test_render_image_blank_drawing_is_white_canvas_at_configured_size() -> None:
    drawing = SpdDrawing(layers=[], width=1404, height=1872)

    image = render_image(drawing)

    assert image.size == (1404, 1872)
    assert image.mode == "RGB"
    assert image.getpixel((0, 0)) == WHITE
    assert image.getpixel((1403, 1871)) == WHITE


def test_render_image_negative_column_wraparound_tile_placement() -> None:
    tile_a = make_tile_png((0, 255, 0, 255))  # row=0, col=0
    tile_b = make_tile_png((0, 0, 255, 255))  # row=0, col=-1 (wraparound branch)
    tid_a = tid_for(row=0, col=0)
    tid_b = tid_for(row=0, col=-1)

    drawing = SpdDrawing(
        layers=[SpdLayer(layer_id=1, tiles={tid_a: tile_a, tid_b: tile_b})],
        width=300,
        height=300,
    )

    image = render_image(drawing)

    # Content bbox: x in [0, 128), y in [-128, 128) -> width 128, height 256.
    # canvas 300x300 -> offset_x = (300-128)//2 - 0 = 86
    #                    offset_y = (300-256)//2 - (-128) = 150
    assert image.getpixel((86 + 1, 150 + 1)) == (0, 255, 0)
    assert image.getpixel((86 + 1, 22 + 1)) == (0, 0, 255)


def test_convert_to_png_bytes(tmp_path: Path) -> None:
    tile = make_tile_png()
    tid = tid_for(row=0, col=0)
    path = write_spd_file(
        tmp_path / "drawing.spd",
        width="256.0",
        height="256.0",
        layers={1: {tid: tile}},
    )

    png_bytes = convert_to_png_bytes(path)

    assert isinstance(png_bytes, bytes)
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")


def test_convert_to_pdf_bytes(tmp_path: Path) -> None:
    tile = make_tile_png()
    tid = tid_for(row=0, col=0)
    path = write_spd_file(
        tmp_path / "drawing.spd",
        width="256.0",
        height="256.0",
        layers={1: {tid: tile}},
    )

    pdf_bytes = convert_to_pdf_bytes(path)

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")


def test_render_image_corrupt_tile_raises_corrupt_spd_file() -> None:
    tid = tid_for(row=0, col=0)
    drawing = SpdDrawing(
        layers=[SpdLayer(layer_id=1, tiles={tid: b"not a real png"})],
        width=TILE_SIZE,
        height=TILE_SIZE,
    )

    with pytest.raises(CorruptSpdFile):
        render_image(drawing)


def test_convert_to_png_bytes_blank_drawing(tmp_path: Path) -> None:
    path = write_spd_file(tmp_path / "blank.spd")

    png_bytes = convert_to_png_bytes(path)

    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")

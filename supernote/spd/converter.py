"""Render .spd drawings into raster images and PDFs."""

from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from reportlab.pdfgen import canvas

from .exceptions import CorruptSpdFile
from .reader import TILE_SIZE, SpdDrawing, load_spd, tile_id_to_row_col

_WHITE_RGB = (255, 255, 255)
_WHITE_RGBA = (255, 255, 255, 255)


def render_image(drawing: SpdDrawing) -> Image.Image:
    """Composite all layers of a .spd drawing into a single RGB image.

    The canvas is white and sized to the larger of the drawing's own
    width/height metadata and the pixel bounding box of all tiles across all
    layers. When the canvas is larger than the tile content, the content is
    centered within it. Layers are composited in the order they appear on
    ``drawing.layers`` (already bottom-to-top).
    """
    bounds = _tile_pixel_bounds(drawing)
    if bounds is None:
        return Image.new("RGB", (drawing.width, drawing.height), _WHITE_RGB)

    min_x, min_y, max_x, max_y = bounds
    content_width = max_x - min_x
    content_height = max_y - min_y

    canvas_width = max(drawing.width, content_width)
    canvas_height = max(drawing.height, content_height)

    offset_x = (canvas_width - content_width) // 2 - min_x
    offset_y = (canvas_height - content_height) // 2 - min_y

    composited = Image.new("RGBA", (canvas_width, canvas_height), _WHITE_RGBA)
    for layer in drawing.layers:
        for tid, png_bytes in layer.tiles.items():
            row, col = tile_id_to_row_col(tid)
            x = row * TILE_SIZE + offset_x
            y = col * TILE_SIZE + offset_y
            try:
                tile_image = Image.open(BytesIO(png_bytes)).convert("RGBA")
            except UnidentifiedImageError as err:
                raise CorruptSpdFile(
                    f"could not decode ink tile {tid} in .spd file: {err}"
                ) from err
            composited.alpha_composite(tile_image, (x, y))

    return composited.convert("RGB")


def _tile_pixel_bounds(drawing: SpdDrawing) -> tuple[int, int, int, int] | None:
    """Return (min_x, min_y, max_x, max_y) pixel bounds of all tiles, or None."""
    min_row: int | None = None
    max_row: int | None = None
    min_col: int | None = None
    max_col: int | None = None

    for layer in drawing.layers:
        for tid in layer.tiles:
            row, col = tile_id_to_row_col(tid)
            min_row = row if min_row is None else min(min_row, row)
            max_row = row if max_row is None else max(max_row, row)
            min_col = col if min_col is None else min(min_col, col)
            max_col = col if max_col is None else max(max_col, col)

    if min_row is None or max_row is None or min_col is None or max_col is None:
        return None

    # Per the reverse-engineered layout: pixel x is derived from the tile's
    # row, and pixel y from its col.
    min_x = min_row * TILE_SIZE
    max_x = (max_row + 1) * TILE_SIZE
    min_y = min_col * TILE_SIZE
    max_y = (max_col + 1) * TILE_SIZE
    return min_x, min_y, max_x, max_y


def convert_to_png_bytes(path: str | Path) -> bytes:
    """Load a .spd file and render it to PNG-encoded bytes."""
    drawing = load_spd(path)
    image = render_image(drawing)
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def convert_to_pdf_bytes(path: str | Path) -> bytes:
    """Load a .spd file and render it to a single-page PDF's bytes.

    The PDF page is sized to the rendered image's own pixel dimensions (used
    directly as points), so there's no rescaling/distortion. A .spd file is
    always a single page, unlike notebook/converter.py's PdfConverter which
    targets fixed A4 pages.
    """
    drawing = load_spd(path)
    image = render_image(drawing)

    buf = BytesIO()
    pdf_canvas = canvas.Canvas(buf, pagesize=(image.width, image.height))
    pdf_canvas.drawInlineImage(image, 0, 0, width=image.width, height=image.height)
    pdf_canvas.showPage()
    pdf_canvas.save()
    return buf.getvalue()

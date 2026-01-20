"""
Image Annotation Module

Handles drawing bounding boxes and confidence labels on detection images.
"""

from io import BytesIO
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from .detector import Detection


def annotate_image(
    image: Image.Image,
    detections: list[Detection],
    box_color: str = "#00FF00",  # Lime green
    box_width: int = 3,
    font_size: int = 16,
) -> Image.Image:
    """
    Draw bounding boxes and confidence labels on an image.

    Args:
        image: PIL Image to annotate
        detections: List of Detection objects with bbox and confidence
        box_color: Color for bounding boxes (hex or name)
        box_width: Line width for bounding boxes
        font_size: Font size for confidence labels

    Returns:
        Annotated PIL Image (copy of original)
    """
    # Work on a copy to avoid modifying the original
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    # Try to load a font, fall back to default if unavailable
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox

        # Draw bounding box
        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline=box_color,
            width=box_width,
        )

        # Draw confidence label with background
        confidence_text = f"{detection.confidence * 100:.0f}%"

        # Get text bounding box
        text_bbox = draw.textbbox((0, 0), confidence_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Position label above the box (or below if too close to top)
        label_padding = 4
        label_x = x1
        label_y = y1 - text_height - label_padding * 2

        if label_y < 0:
            label_y = y2 + label_padding

        # Draw label background
        draw.rectangle(
            [
                (label_x, label_y),
                (label_x + text_width + label_padding * 2, label_y + text_height + label_padding * 2),
            ],
            fill=box_color,
        )

        # Draw text
        draw.text(
            (label_x + label_padding, label_y + label_padding),
            confidence_text,
            fill="black",
            font=font,
        )

    return annotated


def compress_image(
    image: Image.Image,
    max_width: int = 800,
    quality: int = 75,
) -> bytes:
    """
    Resize and compress an image for storage.

    Args:
        image: PIL Image to compress
        max_width: Maximum width in pixels (height scales proportionally)
        quality: JPEG quality (1-100)

    Returns:
        Compressed image as JPEG bytes
    """
    # Convert to RGB if necessary (removes alpha channel)
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")

    # Resize if larger than max_width
    if image.width > max_width:
        ratio = max_width / image.width
        new_height = int(image.height * ratio)
        image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)

    # Compress to JPEG
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    buffer.seek(0)

    return buffer.getvalue()

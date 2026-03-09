"""
Image extraction and region detection from document pages.

Handles isolating images and figure regions from document pages.
"""

from typing import List, Tuple, Optional
import io

from PIL import Image
import numpy as np

from utils.logger import setup_logger
from models.document import BoundingBox

logger = setup_logger(__name__)


class ImageExtractor:
    """
    Extracts and processes images from document pages.

    Handles image detection, cropping, and preprocessing.
    """

    @staticmethod
    def crop_image_from_bytes(
        image_bytes: bytes,
        bbox: BoundingBox,
        page_width: float,
        page_height: float,
    ) -> Optional[bytes]:
        """
        Crop an image region from full page image.

        Args:
            image_bytes: Full page image as bytes.
            bbox: Bounding box for the region to crop.
            page_width: Full page width.
            page_height: Full page height.

        Returns:
            Cropped image bytes or None if crop fails.
        """
        try:
            # Open image
            image = Image.open(io.BytesIO(image_bytes))

            # Convert bounding box coordinates to pixel coordinates
            # Assume bbox is in normalized coordinates (0-1)
            left = int(bbox.x * image.width)
            top = int(bbox.y * image.height)
            right = int((bbox.x + bbox.width) * image.width)
            bottom = int((bbox.y + bbox.height) * image.height)

            # Crop
            cropped = image.crop((left, top, right, bottom))

            # Convert back to bytes
            output = io.BytesIO()
            cropped.save(output, format="PNG")
            return output.getvalue()

        except Exception as e:
            logger.warning(f"Failed to crop image: {str(e)}")
            return None

    @staticmethod
    def detect_figure_regions(
        layout_elements: List[dict],
        min_height: float = 100,
        min_width: float = 100,
    ) -> List[Tuple[str, BoundingBox]]:
        """
        Detect figure/chart regions from layout elements.

        Args:
            layout_elements: List of layout element dictionaries.
            min_height: Minimum height for a figure region (in points).
            min_width: Minimum width for a figure region (in points).

        Returns:
            List of tuples (element_id, bounding_box) for detected figures.
        """
        figures = []

        try:
            for idx, element in enumerate(layout_elements):
                if element.get("type") == "figure" and element.get("bounding_box"):
                    bbox = element["bounding_box"]

                    # Check size constraints
                    if (
                        bbox.height >= min_height
                        and bbox.width >= min_width
                    ):
                        element_id = f"figure_{idx}"
                        figures.append((element_id, bbox))

            logger.debug(f"Detected {len(figures)} figure regions")
            return figures

        except Exception as e:
            logger.warning(f"Failed to detect figure regions: {str(e)}")
            return []

    @staticmethod
    def extract_color_regions(
        image_bytes: bytes,
        target_color: Tuple[int, int, int],
        tolerance: int = 30,
    ) -> List[BoundingBox]:
        """
        Extract regions of a specific color from an image.

        Useful for identifying color-coded regions in documents.

        Args:
            image_bytes: Image data as bytes.
            target_color: RGB color tuple to search for.
            tolerance: Tolerance for color matching.

        Returns:
            List of bounding boxes for matching regions.
        """
        regions = []

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            pixels = np.array(image)

            # Calculate color distance
            target = np.array(target_color)
            distances = np.sqrt(np.sum((pixels - target) ** 2, axis=2))

            # Create mask for matching colors
            mask = distances <= tolerance
            mask_uint8 = mask.astype(np.uint8) * 255

            # Find contiguous regions
            from scipy import ndimage

            labeled, num_features = ndimage.label(mask)

            # Extract bounding boxes for each region
            for region_id in range(1, num_features + 1):
                positions = np.where(labeled == region_id)
                if len(positions[0]) > 0:
                    y_min, y_max = positions[0].min(), positions[0].max()
                    x_min, x_max = positions[1].min(), positions[1].max()

                    bbox = BoundingBox(
                        x=x_min / image.width,
                        y=y_min / image.height,
                        width=(x_max - x_min) / image.width,
                        height=(y_max - y_min) / image.height,
                    )
                    regions.append(bbox)

            logger.debug(f"Found {len(regions)} color regions")
            return regions

        except Exception as e:
            logger.warning(f"Failed to extract color regions: {str(e)}")
            return []

    @staticmethod
    def resize_image(
        image_bytes: bytes,
        max_width: int = 1024,
        max_height: int = 1024,
    ) -> bytes:
        """
        Resize image while maintaining aspect ratio.

        Args:
            image_bytes: Image data as bytes.
            max_width: Maximum width in pixels.
            max_height: Maximum height in pixels.

        Returns:
            Resized image bytes.
        """
        try:
            image = Image.open(io.BytesIO(image_bytes))

            # Calculate scaling factor
            scale = min(
                max_width / image.width,
                max_height / image.height,
                1.0,  # Don't upscale
            )

            if scale < 1.0:
                new_size = (int(image.width * scale), int(image.height * scale))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            # Convert back to bytes
            output = io.BytesIO()
            image.save(output, format="PNG")
            return output.getvalue()

        except Exception as e:
            logger.warning(f"Failed to resize image: {str(e)}")
            return image_bytes

    @staticmethod
    def extract_text_from_image(image_bytes: bytes) -> str:
        """
        Extract text from image using OCR.

        Placeholder for OCR integration (e.g., pytesseract).
        In production, this would call Tesseract or similar.

        Args:
            image_bytes: Image data as bytes.

        Returns:
            Extracted text or empty string if OCR is not available.
        """
        try:
            # Placeholder - requires pytesseract and Tesseract installation
            # import pytesseract
            # image = Image.open(io.BytesIO(image_bytes))
            # return pytesseract.image_to_string(image)

            logger.debug("OCR extraction not configured")
            return ""

        except Exception as e:
            logger.warning(f"Failed to extract text from image: {str(e)}")
            return ""

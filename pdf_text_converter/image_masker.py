"""
Image masker for hiding table regions
Masks detected tables with white rectangles
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple


class ImageMasker:
    """Masks table regions in images"""
    
    def __init__(self, expand_px: int = 0, mask_color: Tuple[int, int, int] = (255, 255, 255)):
        """
        Initialize masker
        
        Args:
            expand_px: Expand bounding boxes by N pixels
            mask_color: RGB color for masking (default: white)
        """
        self.expand_px = expand_px
        self.mask_color = mask_color
    
    def mask_regions(self, image: np.ndarray, tables: List[Dict]) -> np.ndarray:
        """
        Mask table regions in image
        
        Args:
            image: Input image (RGB)
            tables: List of table dicts with 'bbox' key [x1, y1, x2, y2]
            
        Returns:
            Masked image
        """
        if not tables:
            return image.copy()
        
        masked = image.copy()
        h, w = image.shape[:2]
        
        for table in tables:
            bbox = table['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Expand bbox
            if self.expand_px > 0:
                x1 = max(0, x1 - self.expand_px)
                y1 = max(0, y1 - self.expand_px)
                x2 = min(w, x2 + self.expand_px)
                y2 = min(h, y2 + self.expand_px)
            
            # Ensure valid coordinates
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Mask the region
            if x2 > x1 and y2 > y1:
                masked[y1:y2, x1:x2] = self.mask_color
        
        return masked
    
    def get_masked_region(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """
        Extract a specific region from image
        
        Args:
            image: Input image
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Cropped region
        """
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        
        # Ensure valid coordinates
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros((10, 10, 3), dtype=np.uint8)
        
        return image[y1:y2, x1:x2].copy()
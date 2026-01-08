"""
Google Cloud Vision API text extraction
Handles full-page and region-specific text extraction
"""

import numpy as np
from typing import List, Tuple
from pathlib import Path
import io


class VisionExtractor:
    """Extract text using Google Cloud Vision API"""
    
    def __init__(self, credentials_path: str):
        """
        Initialize Vision API client
        
        Args:
            credentials_path: Path to Google Cloud credentials JSON
        """
        self.credentials_path = credentials_path
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Google Cloud Vision client"""
        try:
            from google.cloud import vision
            import os
            
            # Set credentials
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.credentials_path
            
            # Create client
            self.client = vision.ImageAnnotatorClient()
            
            print("✅ Google Vision API initialized")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Google Vision API: {e}")
    
    def extract_text(self, image: np.ndarray) -> str:
        """
        Extract all text from image
        
        Args:
            image: Input image (RGB numpy array)
            
        Returns:
            Extracted text
        """
        from google.cloud import vision
        
        # Convert numpy array to bytes
        image_bytes = self._numpy_to_bytes(image)
        
        # Create Vision API image
        vision_image = vision.Image(content=image_bytes)
        
        # Perform text detection
        response = self.client.text_detection(image=vision_image)
        
        # Check for errors
        if response.error.message:
            raise Exception(f"Vision API error: {response.error.message}")
        
        # Extract text
        texts = response.text_annotations
        
        if texts:
            # First annotation contains all text
            return texts[0].description
        
        return ""
    
    def extract_text_from_bbox(self, image: np.ndarray, bbox: List[float]) -> str:
        """
        Extract text from a specific region
        
        Args:
            image: Input image (RGB numpy array)
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Extracted text from region
        """
        # Crop region
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        
        # Ensure valid coordinates
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        if x2 <= x1 or y2 <= y1:
            return ""
        
        cropped = image[y1:y2, x1:x2]
        
        # Extract text from crop
        return self.extract_text(cropped)
    
    def _numpy_to_bytes(self, image: np.ndarray) -> bytes:
        """Convert numpy array to bytes for Vision API"""
        from PIL import Image
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        # Save to bytes
        byte_io = io.BytesIO()
        pil_image.save(byte_io, format='PNG')
        
        return byte_io.getvalue()
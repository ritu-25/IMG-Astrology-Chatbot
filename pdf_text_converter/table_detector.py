"""
Table detection using multiple backends
Supports: Table Transformer, PaddleOCR, Hybrid mode
"""

import numpy as np
from typing import List, Dict
from abc import ABC, abstractmethod


class BaseTableDetector(ABC):
    """Base class for table detectors"""
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect tables in image
        
        Args:
            image: Input image (RGB numpy array)
            
        Returns:
            List of detected tables with format:
            {
                'bbox': [x1, y1, x2, y2],
                'confidence': float,
                'backend': str
            }
        """
        pass


class TableTransformerDetector(BaseTableDetector):
    """Table detection using Microsoft Table Transformer"""
    
    def __init__(self, model_name: str = "microsoft/table-transformer-detection", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._load_model()
    
    def _load_model(self):
        """Load Table Transformer model"""
        try:
            from transformers import AutoImageProcessor, TableTransformerForObjectDetection
            import torch
            
            print(f"Loading Table Transformer model: {self.model_name}")
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = TableTransformerForObjectDetection.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self.torch = torch
            
            print(f"✅ Table Transformer loaded on {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Table Transformer: {e}")
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """Detect tables using Table Transformer"""
        from PIL import Image
        
        # Convert to PIL
        pil_image = Image.fromarray(image)
        
        # Prepare inputs
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process
        target_sizes = self.torch.tensor([pil_image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, 
            threshold=0.5, 
            target_sizes=target_sizes
        )[0]
        
        # Convert to our format
        tables = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            x1, y1, x2, y2 = box.cpu().numpy()
            tables.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(score),
                'backend': 'table-transformer'
            })
        
        return tables


class PaddleTableDetector(BaseTableDetector):
    """Table detection using PaddleOCR"""
    
    def __init__(self, lang: str = 'en', use_gpu: bool = False):
        self.lang = lang
        self.use_gpu = use_gpu
        self._load_model()
    
    def _load_model(self):
        """Load PaddleOCR model"""
        try:
            from paddleocr import PPStructure
            
            print(f"Loading PaddleOCR structure model (lang={self.lang}, gpu={self.use_gpu})")
            self.engine = PPStructure(
                lang=self.lang,
                use_gpu=self.use_gpu,
                show_log=False,
                table=True,
                ocr=False,  # We only need structure detection
                layout=False
            )
            
            print("✅ PaddleOCR loaded")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load PaddleOCR: {e}")
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """Detect tables using PaddleOCR"""
        # Convert RGB to BGR for PaddleOCR
        bgr_image = image[:, :, ::-1].copy()
        
        # Run detection
        results = self.engine(bgr_image)
        
        # Extract tables
        tables = []
        for item in results:
            if item['type'] == 'table':
                bbox = item['bbox']
                # bbox format: [x1, y1, x2, y2]
                tables.append({
                    'bbox': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                    'confidence': float(item.get('score', 0.9)),
                    'backend': 'paddle'
                })
        
        return tables


class HybridTableDetector(BaseTableDetector):
    """Hybrid detector using both Table Transformer and PaddleOCR"""
    
    def __init__(self, nms_threshold: float = 0.5):
        self.nms_threshold = nms_threshold
        self.transformer = TableTransformerDetector()
        self.paddle = PaddleTableDetector()
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """Detect tables using both backends and merge results"""
        # Get detections from both
        tables_transformer = self.transformer.detect(image)
        tables_paddle = self.paddle.detect(image)
        
        # Combine
        all_tables = tables_transformer + tables_paddle
        
        # Apply NMS to remove duplicates
        if len(all_tables) > 1:
            all_tables = self._non_maximum_suppression(all_tables)
        
        return all_tables
    
    def _non_maximum_suppression(self, tables: List[Dict]) -> List[Dict]:
        """Apply Non-Maximum Suppression to remove overlapping detections"""
        if not tables:
            return []
        
        # Sort by confidence
        tables = sorted(tables, key=lambda x: x['confidence'], reverse=True)
        
        kept = []
        
        for table in tables:
            # Check if overlaps with any kept table
            should_keep = True
            
            for kept_table in kept:
                iou = self._calculate_iou(table['bbox'], kept_table['bbox'])
                if iou > self.nms_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                kept.append(table)
        
        return kept
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union (IoU)"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


class TableDetector:
    """Factory class for creating table detectors"""
    
    @staticmethod
    def create(backend: str = 'table-transformer', **kwargs) -> BaseTableDetector:
        """
        Create a table detector
        
        Args:
            backend: 'table-transformer', 'paddle', or 'hybrid'
            **kwargs: Additional arguments for specific backends
            
        Returns:
            Table detector instance
        """
        if backend == 'table-transformer':
            return TableTransformerDetector(**kwargs)
        
        elif backend == 'paddle':
            return PaddleTableDetector(**kwargs)
        
        elif backend == 'hybrid':
            return HybridTableDetector(**kwargs)
        
        else:
            raise ValueError(f"Unknown backend: {backend}. Choose from: table-transformer, paddle, hybrid")
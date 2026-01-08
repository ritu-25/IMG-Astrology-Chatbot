# from pathlib import Path

# class Config:
#     # Base paths
#     BASE_DIR = Path(__file__).parent
#     OUTPUT_DIR = BASE_DIR / "output"

#     # ========= PDF SETTINGS =========
#     PDF_DPI = 300
#     POPPLER_PATH = r"C:\poppler-25.12.0\Library\bin"

#     # ========= OCR SETTINGS =========
#     # PaddleOCR will be used (no Tesseract needed)
    
#     # ========= GOOGLE VISION =========
#     VISION_CREDENTIALS = (
#         r"C:\Users\saini\OneDrive\Desktop\pdf_text_converter"
#         r"\google_credentials 1.json"
#     )

#     # ========= TABLE DETECTION =========
#     TABLE_CONFIDENCE_THRESHOLD = 0.6
#     EXPAND_TABLE_BBOX = 10
    
#     # Use Table Transformer model
#     TABLE_MODEL = "microsoft/table-transformer-detection"

#     # ========= OUTPUT OPTIONS =========
#     SAVE_JSON = True
#     SAVE_TEXT = True
#     SAVE_VISUALIZATIONS = True
#     SAVE_MASKED_IMAGES = True

"""
Configuration file for BPHS PDF Text Extractor
Centralized settings for all pipeline components
"""

from pathlib import Path


class Config:
    """Configuration class for the BPHS pipeline"""
    
    def __init__(self):
        # ============================================
        # GOOGLE CLOUD VISION API
        # ============================================
        self.GOOGLE_CREDENTIALS = r"C:\Users\saini\OneDrive\Desktop\pdf_text_converter\google_credentials 1.json"
        
        # ============================================
        # PDF CONVERSION
        # ============================================
        self.PDF_DPI = 300  # DPI for PDF to image conversion (higher = better quality, slower)
        
        # ============================================
        # TABLE DETECTION
        # ============================================
        # Available backends: 'table-transformer', 'paddle', 'hybrid'
        self.TABLE_DETECTION_BACKEND = 'table-transformer'
        
        # Confidence threshold for table detection (0.0 - 1.0)
        self.TABLE_CONFIDENCE_THRESHOLD = 0.5
        
        # Expand detected table bounding boxes by N pixels (helps catch borders)
        self.EXPAND_TABLE_BBOX = 10
        
        # ============================================
        # TABLE MASKING
        # ============================================
        # Whether to mask (skip) detected tables during text extraction
        self.SKIP_TABLES = True
        
        # Masking color (RGB) - white by default
        self.MASK_COLOR = (255, 255, 255)
        
        # ============================================
        # OUTPUT SETTINGS
        # ============================================
        self.OUTPUT_DIR = "output"
        
        # What to save
        self.SAVE_TEXT = True              # Save extracted text to .txt
        self.SAVE_JSON = True              # Save results to .json
        self.SAVE_VISUALIZATIONS = True    # Save visualization images
        self.SAVE_MASKED_IMAGES = True     # Save masked page images
        
        # ============================================
        # BACKEND SPECIFIC SETTINGS
        # ============================================
        
        # Table Transformer settings
        self.TABLE_TRANSFORMER_MODEL = "microsoft/table-transformer-detection"
        self.TABLE_TRANSFORMER_DEVICE = "cpu"  # or "cuda" if GPU available
        
        # PaddleOCR settings
        self.PADDLE_LANG = "en"
        self.PADDLE_USE_GPU = False
        
        # Hybrid mode settings (uses both backends)
        self.HYBRID_NMS_THRESHOLD = 0.5  # IoU threshold for Non-Maximum Suppression
        
    def validate(self):
        """Validate configuration settings"""
        errors = []
        
        # Check Google credentials
        if not Path(self.GOOGLE_CREDENTIALS).exists():
            errors.append(f"Google credentials file not found: {self.GOOGLE_CREDENTIALS}")
        
        # Check backend
        valid_backends = ['table-transformer', 'paddle', 'hybrid']
        if self.TABLE_DETECTION_BACKEND not in valid_backends:
            errors.append(f"Invalid backend. Choose from: {valid_backends}")
        
        # Check threshold
        if not 0.0 <= self.TABLE_CONFIDENCE_THRESHOLD <= 1.0:
            errors.append("TABLE_CONFIDENCE_THRESHOLD must be between 0.0 and 1.0")
        
        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(errors))
        
        return True
    
    def __str__(self):
        """String representation of config"""
        return f"""
BPHS Pipeline Configuration
============================
Google Credentials: {self.GOOGLE_CREDENTIALS}
PDF DPI: {self.PDF_DPI}
Detection Backend: {self.TABLE_DETECTION_BACKEND}
Confidence Threshold: {self.TABLE_CONFIDENCE_THRESHOLD}
Skip Tables: {self.SKIP_TABLES}
Output Directory: {self.OUTPUT_DIR}
============================
"""
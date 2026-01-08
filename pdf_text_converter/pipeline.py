"""
Modified pipeline with GUI-based manual review
Complete pipeline.py with all methods
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict
from pdf2image import convert_from_path

from config import Config
from table_detector import TableDetector
from image_masker import ImageMasker
from vision_extractor import VisionExtractor
from manual_reviewer import GUITableReviewer, QuickReviewer


class BPHSPipeline:
    def __init__(self, config: Config = None, review_mode: str = 'gui'):
        """
        Initialize pipeline with GUI review
        
        Args:
            config: Configuration object
            review_mode: 'gui' (visual review), 'quick' (command line with images), 'auto' (no review)
        """
        self.config = config or Config()
        self.review_mode = review_mode
        
        # Create output directory
        self.output_dir = Path(self.config.OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        print(f"Initializing {self.config.TABLE_DETECTION_BACKEND} backend...")
        self.table_detector = TableDetector.create(self.config.TABLE_DETECTION_BACKEND)
        self.masker = ImageMasker(expand_px=self.config.EXPAND_TABLE_BBOX)
        
        print("Initializing Google Vision API...")
        self.vision = VisionExtractor(credentials_path=self.config.GOOGLE_CREDENTIALS)
        
        # Initialize reviewer based on mode
        if review_mode == 'gui':
            print("Manual review: GUI MODE (Interactive visual review)")
            self.reviewer = GUITableReviewer(self.config.OUTPUT_DIR)
        elif review_mode == 'quick':
            print("Manual review: QUICK MODE (Command line with preview images)")
            self.reviewer = QuickReviewer(self.config.OUTPUT_DIR)
        else:
            print("Manual review: DISABLED (Auto mode)")
            self.reviewer = None
        
        print(f"Pipeline initialized with {self.config.TABLE_DETECTION_BACKEND} backend")
    
    def pdf_to_images(self, pdf_path: str, start_page: int = None, end_page: int = None) -> List[np.ndarray]:
        print(f"\nConverting PDF to images at {self.config.PDF_DPI} DPI...")
        
        images = convert_from_path(
            pdf_path, 
            dpi=self.config.PDF_DPI,
            first_page=start_page,
            last_page=end_page
        )
        
        np_images = [np.array(img) for img in images]
        print(f"Converted {len(np_images)} pages")
        return np_images
    
    def process_pdf(self, pdf_path: str, start_page: int = None, 
                   end_page: int = None) -> List[Dict]:
        """
        Process PDF with GUI review workflow
        """
        print("\n" + "="*60)
        print("PDF PROCESSING WITH MANUAL REVIEW")
        print("="*60)
        print(f"PDF: {pdf_path}")
        print(f"Pages: {start_page or 'start'} to {end_page or 'end'}")
        print(f"Review mode: {self.review_mode.upper()}")
        print("="*60)
        
        # Convert PDF to images
        images = self.pdf_to_images(pdf_path, start_page, end_page)
        actual_start = start_page or 1
        
        # PHASE 1: DETECT ALL TABLES
        print("\n" + "="*60)
        print("PHASE 1: DETECTING TABLES")
        print("="*60)
        
        all_page_data = []  # Store data for all pages
        pages_for_review = []  # Pages to show in reviewer
        
        for i, image in enumerate(images):
            page_num = actual_start + i
            print(f"\nScanning {i+1}/{len(images)} (Page {page_num})...")
            
            # Detect tables
            tables = self.table_detector.detect(image)
            tables_filtered = [
                t for t in tables
                if t["confidence"] >= self.config.TABLE_CONFIDENCE_THRESHOLD
            ]
            
            # Store all page data
            page_data = {
                'page_num': page_num,
                'image': image,
                'tables_all': tables,
                'tables_filtered': tables_filtered
            }
            all_page_data.append(page_data)
            
            # Add ALL pages to review (even without tables)
            pages_for_review.append({
                'page_num': page_num,
                'image': image,
                'tables': tables_filtered  # Empty list if no tables
            })
            
            if tables_filtered:
                print(f"  ✓ Found {len(tables_filtered)} table(s)")
            else:
                print(f"  ⊘ No tables detected (you can still draw manually)")
        
        # PHASE 2: MANUAL REVIEW
        approved_by_page = {}
        
        if self.reviewer:
            print("\n" + "="*60)
            print("PHASE 2: MANUAL REVIEW")
            print("="*60)
            print(f"Total pages to review: {len(pages_for_review)}")
            pages_with_auto_tables = len([p for p in pages_for_review if p['tables']])
            pages_without_tables = len(pages_for_review) - pages_with_auto_tables
            print(f"  Pages with auto-detected tables: {pages_with_auto_tables}")
            print(f"  Pages without tables (can draw manually): {pages_without_tables}")
            print(f"Total auto-detected tables: {sum(len(p['tables']) for p in pages_for_review)}")
            
            approved_by_page = self.reviewer.review_all_pages(pages_for_review)
        
        else:  # Auto mode
            print("\n⊘ Auto mode: All detected tables will be masked")
            for page_data in pages_for_review:
                if page_data['tables']:
                    approved_by_page[page_data['page_num']] = page_data['tables']
        
        # PHASE 3: PROCESS ALL PAGES WITH APPROVED DECISIONS
        print("\n" + "="*60)
        print("PHASE 3: PROCESSING PAGES")
        print("="*60)
        
        results = []
        for page_data in all_page_data:
            page_num = page_data['page_num']
            image = page_data['image']
            tables_filtered = page_data['tables_filtered']
            
            print(f"\nProcessing Page {page_num}...")
            
            # Get approved tables for this page
            tables_to_mask = approved_by_page.get(page_num, [])
            
            # Process the page
            result = self._process_single_page(
                image=image,
                page_num=page_num,
                tables_detected=tables_filtered,
                tables_to_mask=tables_to_mask
            )
            
            results.append(result)
        
        # SAVE RESULTS
        print(f"\n" + "="*60)
        print("SAVING RESULTS")
        print("="*60)
        
        if self.config.SAVE_JSON:
            self._save_json(results)
        
        if self.config.SAVE_TEXT:
            self._save_text(results)
            self._save_masked_table_text(results)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _process_single_page(self, image: np.ndarray, page_num: int,
                            tables_detected: List[Dict], 
                            tables_to_mask: List[Dict]) -> Dict:
        """
        Process a single page with pre-approved masking decisions
        
        Args:
            image: Page image
            page_num: Page number
            tables_detected: All tables detected
            tables_to_mask: Tables approved for masking
        """
        try:
            # 1️⃣ Extract table text BEFORE masking
            table_texts = []
            for t in tables_to_mask:
                try:
                    table_text = self.vision.extract_text_from_bbox(image, t["bbox"])
                    table_texts.append({
                        "page_num": page_num,
                        "bbox": t["bbox"],
                        "backend": t.get("backend", "unknown"),
                        "text": table_text
                    })
                except Exception as e:
                    print(f"  ⚠ Error extracting table text: {e}")

            # 2️⃣ Mask tables
            if self.config.SKIP_TABLES and tables_to_mask:
                masked_image = self.masker.mask_regions(image, tables_to_mask)
                print(f"  ✓ Masked {len(tables_to_mask)} table(s)")
            else:
                masked_image = image
                if tables_to_mask:
                    print(f"  ⊘ SKIP_TABLES=False, not masking")
                else:
                    print(f"  ⊘ No tables to mask")

            # 3️⃣ Extract PAGE text (after masking)
            print(f"  📄 Extracting text with Google Vision API...")
            text = self.vision.extract_text(masked_image)
            text_length = len(text)
            print(f"  ✓ Extracted {text_length} characters")

            # 4️⃣ Save artifacts
            page_id = f"page_{page_num:03d}"

            if self.config.SAVE_VISUALIZATIONS:
                self._save_visualization(image, tables_to_mask, page_id)

            if self.config.SAVE_MASKED_IMAGES:
                masked_path = self.output_dir / f"{page_id}_masked.png"
                cv2.imwrite(
                    str(masked_path),
                    cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
                )

            return {
                'page_num': page_num,
                'text': text,
                'tables_detected': len(tables_detected),
                'tables_masked': len(tables_to_mask),
                'table_regions': tables_to_mask,
                'table_texts': table_texts,
                'text_length': text_length,
                'status': 'success'
            }

        except Exception as e:
            print(f"  ❌ Error processing page {page_num}: {e}")
            import traceback
            traceback.print_exc()

            return {
                'page_num': page_num,
                'text': '',
                'tables_detected': 0,
                'tables_masked': 0,
                'table_regions': [],
                'table_texts': [],
                'text_length': 0,
                'status': 'error',
                'error': str(e)
            }

    def _save_visualization(self, image: np.ndarray, tables: List[Dict], page_id: str):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            
            fig, ax = plt.subplots(1, figsize=(12, 16))
            ax.imshow(image)
            
            for i, table in enumerate(tables):
                bbox = table['bbox']
                x1, y1, x2, y2 = bbox
                width, height = x2 - x1, y2 - y1
                
                rect = Rectangle((x1, y1), width, height, 
                               linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                
                ax.text(x1, y1-10, f"MASKED {i+1}", 
                       color='red', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.axis('off')
            plt.title(f'{page_id} - Masked Tables', fontsize=14)
            plt.tight_layout()
            
            viz_path = self.output_dir / f"{page_id}_masked_viz.png"
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            print(f"   Could not save visualization: {e}")
    
    def _save_json(self, results: List[Dict]):
        """Save both content.json and table.json"""
        
        # 1. Save content.json (page-level data)
        content_json_path = self.output_dir / 'content.json'
        
        successful = [r for r in results if r.get('status') == 'success']
        failed = [r for r in results if r.get('status') == 'error']
        
        content_output = {
            'metadata': {
                'total_pages': len(results),
                'successful_pages': len(successful),
                'failed_pages': len(failed),
                'total_tables_detected': sum(r.get('tables_detected', 0) for r in results),
                'total_tables_masked': sum(r.get('tables_masked', 0) for r in results),
                'total_characters_extracted': sum(r.get('text_length', 0) for r in results),
                'review_mode': self.review_mode
            },
            'pages': [
                {
                    'page_num': r['page_num'],
                    'text': r.get('text', ''),
                    'text_length': r.get('text_length', 0),
                    'tables_detected': r.get('tables_detected', 0),
                    'tables_masked': r.get('tables_masked', 0),
                    'status': r.get('status', 'unknown'),
                    'error': r.get('error') if r.get('status') == 'error' else None
                }
                for r in results
            ]
        }
        
        with open(content_json_path, 'w', encoding='utf-8') as f:
            json.dump(content_output, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Saved content JSON: {content_json_path}")
        
        # 2. Save table.json (table-level data)
        table_json_path = self.output_dir / 'table.json'
        
        all_tables = []
        for result in results:
            page_num = result['page_num']
            for table_text in result.get('table_texts', []):
                all_tables.append({
                    'page_num': page_num,
                    'bbox': table_text['bbox'],
                    'backend': table_text['backend'],
                    'text': table_text['text'],
                    'text_length': len(table_text['text'])
                })
        
        table_output = {
            'metadata': {
                'total_tables_masked': len(all_tables),
                'total_pages_with_tables': len([r for r in results if r.get('table_texts')]),
                'review_mode': self.review_mode
            },
            'tables': all_tables
        }
        
        with open(table_json_path, 'w', encoding='utf-8') as f:
            json.dump(table_output, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Saved table JSON: {table_json_path}")
    
    def _save_text(self, results: List[Dict]):
        """Save extracted page text (content) to content.txt"""
        content_path = self.output_dir / 'content.txt'
        
        with open(content_path, 'w', encoding='utf-8') as f:
            f.write("EXTRACTED PAGE CONTENT (Hindi, English, Sanskrit)\n")
            f.write(f"Review Mode: {self.review_mode.upper()}\n")
            f.write("Tables have been masked based on manual review\n")
            f.write("="*60 + "\n\n")
            
            for result in results:
                f.write(f"\n{'='*60}\n")
                f.write(f"PAGE {result['page_num']}")
                
                if result.get('status') == 'error':
                    f.write(f" [ERROR: {result.get('error', 'Unknown')}]")
                else:
                    detected = result.get('tables_detected', 0)
                    masked = result.get('tables_masked', 0)
                    f.write(f" [Tables Detected: {detected}, Masked: {masked}]")
                
                f.write(f"\n{'='*60}\n\n")
                f.write(result.get('text', ''))
                f.write("\n\n")
        
        print(f"✅ Saved content text: {content_path}")
    
    def _save_masked_table_text(self, results: List[Dict]):
        """Save text of MASKED tables to table.txt"""
        table_path = self.output_dir / "table.txt"

        with open(table_path, "w", encoding="utf-8") as f:
            f.write("MASKED TABLE TEXT\n")
            f.write(f"Review Mode: {self.review_mode.upper()}\n")
            f.write("Text extracted from tables before masking\n")
            f.write("="*60 + "\n\n")
            
            for result in results:
                page = result["page_num"]
                table_texts = result.get("table_texts", [])
                
                if table_texts:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"PAGE {page} - {len(table_texts)} Masked Table(s)\n")
                    f.write(f"{'='*60}\n\n")

                    for idx, table in enumerate(table_texts, 1):
                        f.write(f"--- Table {idx} ---\n")
                        f.write(f"Backend: {table['backend']}\n")
                        f.write(f"Bounding Box: {table['bbox']}\n")
                        f.write(f"Text:\n{table['text']}\n")
                        f.write("\n" + "-"*40 + "\n\n")

        print(f"✅ Saved table text: {table_path}")
    
    def _print_summary(self, results: List[Dict]):
        successful = [r for r in results if r.get('status') == 'success']
        failed = [r for r in results if r.get('status') == 'error']
        
        total_detected = sum(r.get('tables_detected', 0) for r in results)
        total_masked = sum(r.get('tables_masked', 0) for r in results)

        print("\n" + "="*60)
        print("✅ EXTRACTION COMPLETE!")
        print("="*60)
        print(f"Total pages: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        print(f"Total tables detected: {total_detected}")
        print(f"Total tables masked: {total_masked}")
        if total_detected > 0:
            kept = total_detected - total_masked
            print(f"Tables kept (not masked): {kept}")
            print(f"Masking rate: {100*total_masked/total_detected:.1f}%")
        print(f"Total characters extracted: {sum(r.get('text_length', 0) for r in results):,}")
        
        if failed:
            print(f"\n⚠️ Failed pages:")
            for r in failed:
                print(f"  - Page {r['page_num']}: {r.get('error', 'Unknown')}")
        
        print(f"\n📁 Output directory: {self.output_dir}/")
        print("="*60)
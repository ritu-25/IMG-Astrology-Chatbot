"""
GUI-Based Manual Review System with Manual Drawing Capability
Review auto-detected tables + Draw your own boxes for missed tables
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple


class GUITableReviewer:
    """Interactive GUI for reviewing detected tables AND manually drawing new ones"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.review_dir = self.output_dir / "review_gui"
        self.review_dir.mkdir(exist_ok=True)
        
        self.all_decisions = {}  # page_num -> list of decisions
        
        # Colors
        self.COLOR_MASK = (0, 0, 255)      # Red for tables to MASK
        self.COLOR_KEEP = (0, 255, 0)      # Green for tables to KEEP
        self.COLOR_PENDING = (255, 165, 0) # Orange for pending review
        self.COLOR_MANUAL = (255, 0, 255)  # Magenta for manually drawn
        
        # Drawing state
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.current_rect = None
        
    def review_all_pages(self, pages_data: List[Dict]) -> Dict[int, List[Dict]]:
        """
        Review all pages with detected tables using GUI
        
        Args:
            pages_data: List of dicts with 'image', 'tables', 'page_num'
                       (tables can be empty list for pages with no auto-detection)
            
        Returns:
            Dict mapping page_num -> list of tables to mask
        """
        if not pages_data:
            print("No pages to review")
            return {}
        
        print("\n" + "="*70)
        print("GUI TABLE REVIEW SYSTEM - ALL PAGES MODE")
        print("="*70)
        print("\n🎮 CONTROLS:")
        print("  === REVIEW AUTO-DETECTED TABLES ===")
        print("  MOUSE CLICK: Click on a table box to select it")
        print("  M: Mark selected table as MASK (Red)")
        print("  K: Mark selected table as KEEP (Green)")
        print("  A: Mark ALL tables as MASK")
        print("  N: Mark NONE (Keep all tables)")
        print()
        print("  === DRAW YOUR OWN TABLES ===")
        print("  MOUSE DRAG: Click & drag to draw a box around missed tables")
        print("  T: Save drawn box as TABLE to MASK (Red)")
        print("  R: Save drawn box as REGION to KEEP (Green)")
        print("  C: Cancel current drawing")
        print()
        print("  === OTHER ===")
        print("  D: Delete last decision/drawing")
        print("  SPACE/ENTER: Save & Next page (or skip if nothing to review)")
        print("  S: Skip page without changes")
        print("  Q/ESC: Save & Quit")
        print("="*70)
        print(f"\n📄 Reviewing ALL {len(pages_data)} pages")
        print(f"   (You can draw boxes on pages with no auto-detected tables)")
        print("="*70)
        
        cv2.namedWindow('Table Review', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Table Review', 1400, 900)
        
        for idx, page_data in enumerate(pages_data):
            page_num = page_data['page_num']
            image = page_data['image']
            tables = page_data['tables']  # Can be empty list
            
            if tables:
                print(f"\n📄 Page {page_num} ({idx+1}/{len(pages_data)}) - {len(tables)} auto-detected table(s)")
            else:
                print(f"\n📄 Page {page_num} ({idx+1}/{len(pages_data)}) - No auto-detected tables")
                print(f"   💡 You can draw boxes manually if needed")
            
            # Review this page
            decisions = self._review_page_gui(image, tables, page_num, idx+1, len(pages_data))
            
            # Only save if there are decisions (auto or manual)
            if decisions:
                self.all_decisions[page_num] = decisions
        
        cv2.destroyAllWindows()
        
        # Save review log
        self._save_review_log()
        
        # Convert to tables_to_mask format
        result = {}
        for page_num, decisions in self.all_decisions.items():
            masked_tables = [d['table'] for d in decisions if d['decision'] == 'mask']
            if masked_tables:
                result[page_num] = masked_tables
        
        return result
    
    def _review_page_gui(self, image: np.ndarray, tables: List[Dict], page_num: int,
                        current_page: int = 1, total_pages: int = 1) -> List[Dict]:
        """Review tables on a single page using GUI with drawing capability"""
        
        # Initialize decisions for auto-detected tables
        page_decisions = []
        next_manual_id = 1  # For naming manually drawn tables
        
        for idx, table in enumerate(tables):
            # IMPORTANT: Convert bbox to integers here
            bbox = table['bbox']
            bbox_int = [int(x) for x in bbox]
            
            page_decisions.append({
                'table_idx': idx,
                'table': table,
                'decision': 'pending',
                'bbox': bbox_int,  # Use integer bbox
                'source': 'auto',  # auto-detected
                'confidence': table.get('confidence', 0.0)
            })
        
        # Create display image
        display_img = image.copy()
        temp_img = image.copy()  # For drawing preview
        selected_idx = None
        
        # Mouse callback state
        mouse_state = {
            'selected_idx': None,
            'drawing': False,
            'ix': -1,
            'iy': -1,
            'temp_rect': None
        }
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal display_img, temp_img
            
            if event == cv2.EVENT_LBUTTONDOWN:
                # Check if clicking on existing box
                clicked_on_box = False
                for idx, decision in enumerate(page_decisions):
                    x1, y1, x2, y2 = decision['bbox']
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        mouse_state['selected_idx'] = idx
                        clicked_on_box = True
                        print(f"  ✓ Selected {decision['source'].upper()} Table {idx+1}")
                        break
                
                # If not clicking on box, start drawing new box
                if not clicked_on_box:
                    mouse_state['drawing'] = True
                    mouse_state['ix'] = x
                    mouse_state['iy'] = y
                    mouse_state['selected_idx'] = None
            
            elif event == cv2.EVENT_MOUSEMOVE:
                if mouse_state['drawing']:
                    # Draw preview rectangle
                    temp_img = display_img.copy()
                    cv2.rectangle(temp_img, 
                                (mouse_state['ix'], mouse_state['iy']), 
                                (x, y), 
                                self.COLOR_MANUAL, 3)
                    cv2.putText(temp_img, "Drawing... (Press T=Mask, R=Keep, C=Cancel)", 
                              (10, image.shape[0] - 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_MANUAL, 2)
                    cv2.imshow('Table Review', temp_img)
            
            elif event == cv2.EVENT_LBUTTONUP:
                if mouse_state['drawing']:
                    mouse_state['drawing'] = False
                    x1, y1 = min(mouse_state['ix'], x), min(mouse_state['iy'], y)
                    x2, y2 = max(mouse_state['ix'], x), max(mouse_state['iy'], y)
                    
                    # Only save if box is large enough
                    if (x2 - x1) > 20 and (y2 - y1) > 20:
                        mouse_state['temp_rect'] = (x1, y1, x2, y2)
                        print(f"  📐 Box drawn: ({x1}, {y1}) to ({x2}, {y2})")
                        print(f"     Press T to MASK, R to KEEP, or C to CANCEL")
                    else:
                        print(f"  ⚠ Box too small, ignored")
                        mouse_state['temp_rect'] = None
                        temp_img = display_img.copy()
        
        cv2.setMouseCallback('Table Review', mouse_callback)
        
        while True:
            # Redraw display
            display_img = image.copy()
            
            # Draw all table boxes with their current status
            for idx, decision in enumerate(page_decisions):
                x1, y1, x2, y2 = decision['bbox']
                
                # Ensure integers (extra safety)
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Choose color based on decision and source
                if decision['decision'] == 'mask':
                    color = self.COLOR_MASK
                    label = "MASK"
                elif decision['decision'] == 'keep':
                    color = self.COLOR_KEEP
                    label = "KEEP"
                else:  # pending
                    if decision['source'] == 'manual':
                        color = self.COLOR_MANUAL
                        label = "MANUAL"
                    else:
                        color = self.COLOR_PENDING
                        label = "PENDING"
                
                # Draw thicker box if selected
                thickness = 5 if mouse_state['selected_idx'] == idx else 2
                cv2.rectangle(display_img, (x1, y1), (x2, y2), color, thickness)
                
                # Add label with background
                source_prefix = "M" if decision['source'] == 'manual' else f"{idx+1}"
                label_text = f"[{source_prefix}] {label}"
                if decision['source'] == 'auto':
                    conf_text = f" {decision['confidence']:.2f}"
                    label_text += conf_text
                
                # Background for text
                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(display_img, (x1, y1-30), (x1 + text_size[0] + 10, y1), color, -1)
                
                cv2.putText(display_img, label_text, (x1+5, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw temporary rectangle if exists
            if mouse_state['temp_rect']:
                x1, y1, x2, y2 = mouse_state['temp_rect']
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(display_img, (x1, y1), (x2, y2), self.COLOR_MANUAL, 3)
                cv2.putText(display_img, "New box (T=Mask, R=Keep, C=Cancel)", 
                          (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_MANUAL, 2)
            
            # Add instructions overlay
            self._draw_instructions(display_img, page_num, page_decisions, mouse_state)
            
            # Show image
            if not mouse_state['drawing']:
                cv2.imshow('Table Review', display_img)
            
            key = cv2.waitKey(1) & 0xFF
            
            # === SAVE DRAWN BOX AS TABLE TO MASK ===
            if key == ord('t'):
                if mouse_state['temp_rect']:
                    bbox = mouse_state['temp_rect']
                    new_table = {
                        'bbox': [int(x) for x in bbox],  # Ensure integers
                        'confidence': 1.0,
                        'backend': 'manual',
                        'manual': True
                    }
                    page_decisions.append({
                        'table_idx': len(page_decisions),
                        'table': new_table,
                        'decision': 'mask',
                        'bbox': [int(x) for x in bbox],  # Ensure integers
                        'source': 'manual',
                        'confidence': 1.0
                    })
                    print(f"  ✓ Manually drawn table added as MASK")
                    mouse_state['temp_rect'] = None
                else:
                    print(f"  ⚠ No box drawn. Click and drag to draw a box first.")
            
            # === SAVE DRAWN BOX AS REGION TO KEEP ===
            elif key == ord('r'):
                if mouse_state['temp_rect']:
                    bbox = mouse_state['temp_rect']
                    new_table = {
                        'bbox': [int(x) for x in bbox],  # Ensure integers
                        'confidence': 1.0,
                        'backend': 'manual',
                        'manual': True
                    }
                    page_decisions.append({
                        'table_idx': len(page_decisions),
                        'table': new_table,
                        'decision': 'keep',
                        'bbox': [int(x) for x in bbox],  # Ensure integers
                        'source': 'manual',
                        'confidence': 1.0
                    })
                    print(f"  ✓ Manually drawn region added as KEEP")
                    mouse_state['temp_rect'] = None
                else:
                    print(f"  ⚠ No box drawn. Click and drag to draw a box first.")
            
            # === CANCEL CURRENT DRAWING ===
            elif key == ord('c'):
                if mouse_state['temp_rect']:
                    mouse_state['temp_rect'] = None
                    mouse_state['drawing'] = False
                    print(f"  ↺ Drawing cancelled")
            
            # === MARK SELECTED AS MASK ===
            elif key == ord('m'):
                if mouse_state['selected_idx'] is not None:
                    idx = mouse_state['selected_idx']
                    page_decisions[idx]['decision'] = 'mask'
                    print(f"  ✓ Table {idx+1} marked as MASK")
                else:
                    print("  ⚠ No table selected. Click on a table first.")
            
            # === MARK SELECTED AS KEEP ===
            elif key == ord('k'):
                if mouse_state['selected_idx'] is not None:
                    idx = mouse_state['selected_idx']
                    page_decisions[idx]['decision'] = 'keep'
                    print(f"  ✓ Table {idx+1} marked as KEEP")
                else:
                    print("  ⚠ No table selected. Click on a table first.")
            
            # === MASK ALL ===
            elif key == ord('a'):
                for decision in page_decisions:
                    decision['decision'] = 'mask'
                print(f"  ✓ All {len(page_decisions)} items marked as MASK")
            
            # === KEEP ALL (NONE) ===
            elif key == ord('n'):
                for decision in page_decisions:
                    decision['decision'] = 'keep'
                print(f"  ✓ All {len(page_decisions)} items marked as KEEP")
            
            # === DELETE LAST ===
            elif key == ord('d'):
                if page_decisions:
                    removed = page_decisions.pop()
                    src = removed['source'].upper()
                    dec = removed['decision'].upper()
                    print(f"  ↺ Removed last {src} table ({dec})")
                    mouse_state['selected_idx'] = None
                else:
                    print(f"  ⚠ No tables to delete")
            
            # === SKIP PAGE ===
            elif key == ord('s'):
                print(f"  ⊘ Page {page_num} skipped (no changes)")
                # Return empty if nothing drawn, or return pending items as 'keep'
                for d in page_decisions:
                    if d['decision'] == 'pending':
                        d['decision'] = 'keep'
                return page_decisions if page_decisions else []
            
            # === NEXT PAGE ===
            elif key in [ord(' '), 13]:  # Space or Enter
                # If no decisions at all (no auto tables, nothing drawn), allow quick skip
                if not page_decisions:
                    print(f"  → Page {page_num} - No tables (skipped)")
                    return []
                
                # Check if all tables have been reviewed
                pending = [d for d in page_decisions if d['decision'] == 'pending']
                if pending:
                    print(f"  ⚠ Warning: {len(pending)} item(s) still pending. Press again to confirm.")
                    key2 = cv2.waitKey(0) & 0xFF
                    if key2 not in [ord(' '), 13]:
                        continue
                    # Mark pending as keep
                    for d in pending:
                        d['decision'] = 'keep'
                
                auto_count = len([d for d in page_decisions if d['source'] == 'auto'])
                manual_count = len([d for d in page_decisions if d['source'] == 'manual'])
                masked_count = len([d for d in page_decisions if d['decision'] == 'mask'])
                kept_count = len([d for d in page_decisions if d['decision'] == 'keep'])
                
                if page_decisions:
                    print(f"  ✓ Page {page_num} complete:")
                    print(f"    Auto-detected: {auto_count}, Manually drawn: {manual_count}")
                    print(f"    Decision: {masked_count} MASK, {kept_count} KEEP")
                else:
                    print(f"  → Page {page_num} - No changes")
                break
            
            # === QUIT ===
            elif key in [ord('q'), 27]:  # Q or ESC
                print("\n⚠ Quitting review. Remaining tables will be kept (not masked).")
                # Mark all pending as keep
                for d in page_decisions:
                    if d['decision'] == 'pending':
                        d['decision'] = 'keep'
                return page_decisions
        
        return page_decisions
    
    def _draw_instructions(self, img: np.ndarray, page_num: int, 
                          decisions: List[Dict], mouse_state: Dict):
        """Draw instructions overlay on the image"""
        h, w = img.shape[:2]
        
        # Semi-transparent overlay at top
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (w-10, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Count decisions
        auto_count = len([d for d in decisions if d['source'] == 'auto'])
        manual_count = len([d for d in decisions if d['source'] == 'manual'])
        mask_count = len([d for d in decisions if d['decision'] == 'mask'])
        keep_count = len([d for d in decisions if d['decision'] == 'keep'])
        pending_count = len([d for d in decisions if d['decision'] == 'pending'])
        
        # Draw text
        y = 30
        cv2.putText(img, f"Page {page_num} | Auto: {auto_count} | Manual: {manual_count}", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 30
        
        # Show status differently if no tables
        if auto_count == 0 and manual_count == 0:
            cv2.putText(img, "No tables detected - Draw boxes manually or press SPACE to skip", 
                       (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)
        else:
            cv2.putText(img, f"MASK: {mask_count} | KEEP: {keep_count} | PENDING: {pending_count}", 
                       (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        y += 30
        cv2.putText(img, "REVIEW: Click box->M=Mask K=Keep | DRAW: Drag box->T=Mask R=Keep", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 25
        cv2.putText(img, "A=All N=None D=Delete S=Skip | SPACE=Next Q=Quit", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Show drawing status
        if mouse_state.get('drawing') or mouse_state.get('temp_rect'):
            y += 30
            cv2.putText(img, ">>> DRAWING MODE ACTIVE <<<", 
                       (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    def _save_review_log(self):
        """Save all review decisions to JSON"""
        log_path = self.review_dir / 'gui_review_log.json'
        
        # Calculate summary
        all_decisions_flat = []
        for page_num, decisions in self.all_decisions.items():
            for d in decisions:
                all_decisions_flat.append({
                    'page_num': page_num,
                    'table_idx': d['table_idx'] + 1,
                    'bbox': d['bbox'],
                    'confidence': d.get('confidence', 1.0),
                    'backend': d['table'].get('backend', 'unknown'),
                    'source': d['source'],  # 'auto' or 'manual'
                    'decision': d['decision']
                })
        
        summary = {
            'total_tables': len(all_decisions_flat),
            'auto_detected': len([d for d in all_decisions_flat if d['source'] == 'auto']),
            'manually_drawn': len([d for d in all_decisions_flat if d['source'] == 'manual']),
            'masked': len([d for d in all_decisions_flat if d['decision'] == 'mask']),
            'kept': len([d for d in all_decisions_flat if d['decision'] == 'keep']),
        }
        
        output = {
            'summary': summary,
            'decisions': all_decisions_flat
        }
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Review log saved: {log_path}")
        print(f"📊 Summary:")
        print(f"   Total: {summary['total_tables']} tables")
        print(f"   Auto-detected: {summary['auto_detected']}, Manual: {summary['manually_drawn']}")
        print(f"   Decision: {summary['masked']} masked, {summary['kept']} kept")


class QuickReviewer:
    """
    Simpler non-GUI reviewer for command line
    Shows image paths and gets keyboard input
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.review_dir = self.output_dir / "review_quick"
        self.review_dir.mkdir(exist_ok=True)
        self.decisions = []
    
    def review_all_pages(self, pages_data: List[Dict]) -> Dict[int, List[Dict]]:
        """Quick command-line review with saved preview images"""
        
        if not pages_data:
            return {}
        
        print("\n" + "="*70)
        print("QUICK REVIEW MODE")
        print("="*70)
        
        result = {}
        
        for idx, page_data in enumerate(pages_data):
            page_num = page_data['page_num']
            image = page_data['image']
            tables = page_data['tables']
            
            print(f"\n📄 Page {page_num} ({idx+1}/{len(pages_data)}) - {len(tables)} table(s)")
            
            # Save preview image
            preview_path = self._save_preview(image, tables, page_num)
            print(f"📷 Preview saved: {preview_path}")
            
            # Review options
            print("\nOptions:")
            print("  [a]ll - Mask ALL tables on this page")
            print("  [n]one - Keep ALL tables (no masking)")
            print("  [i]ndividual - Review each table")
            print("  [s]kip - Skip this page, keep all tables")
            
            while True:
                choice = input("Your choice: ").lower().strip()
                
                if choice in ['a', 'all']:
                    result[page_num] = tables
                    for t in tables:
                        self._log_decision(page_num, t, 'mask')
                    print(f"✓ All {len(tables)} tables will be MASKED")
                    break
                
                elif choice in ['n', 'none', 's', 'skip']:
                    for t in tables:
                        self._log_decision(page_num, t, 'keep')
                    print(f"✓ All {len(tables)} tables will be KEPT")
                    break
                
                elif choice in ['i', 'individual']:
                    masked = self._review_individual(image, tables, page_num)
                    if masked:
                        result[page_num] = masked
                    break
                
                else:
                    print("Invalid choice. Please enter a, n, i, or s")
        
        self._save_log()
        return result
    
    def _review_individual(self, image: np.ndarray, tables: List[Dict], 
                          page_num: int) -> List[Dict]:
        """Review each table individually"""
        masked_tables = []
        
        for idx, table in enumerate(tables, 1):
            # Save individual crop
            crop_path = self._save_crop(image, table, page_num, idx)
            
            print(f"\n  Table {idx}/{len(tables)}")
            print(f"  Confidence: {table['confidence']:.2f}")
            print(f"  Backend: {table.get('backend', 'unknown')}")
            print(f"  Crop: {crop_path}")
            
            while True:
                decision = input("  Mask this? [y/n]: ").lower().strip()
                
                if decision in ['y', 'yes']:
                    masked_tables.append(table)
                    self._log_decision(page_num, table, 'mask')
                    print("  ✓ Will be MASKED")
                    break
                elif decision in ['n', 'no']:
                    self._log_decision(page_num, table, 'keep')
                    print("  ✓ Will be KEPT")
                    break
        
        return masked_tables
    
    def _save_preview(self, image: np.ndarray, tables: List[Dict], 
                     page_num: int) -> Path:
        """Save preview with all tables outlined"""
        preview = image.copy()
        
        for idx, table in enumerate(tables, 1):
            x1, y1, x2, y2 = map(int, table['bbox'])
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(preview, f"T{idx}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        path = self.review_dir / f"page_{page_num:03d}_preview.png"
        cv2.imwrite(str(path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
        return path
    
    def _save_crop(self, image: np.ndarray, table: Dict, 
                   page_num: int, table_idx: int) -> Path:
        """Save cropped table image"""
        x1, y1, x2, y2 = map(int, table['bbox'])
        crop = image[y1:y2, x1:x2]
        
        path = self.review_dir / f"page_{page_num:03d}_table_{table_idx}.png"
        cv2.imwrite(str(path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        return path
    
    def _log_decision(self, page_num: int, table: Dict, decision: str):
        """Log decision"""
        self.decisions.append({
            'page_num': page_num,
            'bbox': table['bbox'],
            'confidence': table['confidence'],
            'backend': table.get('backend', 'unknown'),
            'decision': decision
        })
    
    def _save_log(self):
        """Save review log"""
        log_path = self.review_dir / 'quick_review_log.json'
        
        summary = {
            'total': len(self.decisions),
            'masked': len([d for d in self.decisions if d['decision'] == 'mask']),
            'kept': len([d for d in self.decisions if d['decision'] == 'keep'])
        }
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump({'summary': summary, 'decisions': self.decisions}, 
                     f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Review log saved: {log_path}")
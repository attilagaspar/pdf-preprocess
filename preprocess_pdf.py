"""
PDF Preprocessor for OCR
Processes double-page book scans: 
1. Detects the rectangular/trapezoid page in dark background
2. Applies perspective correction (trapezoid -> rectangle)
3. Splits corrected page in half
4. Reorganizes into separate PDFs
"""

import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d
import fitz  # PyMuPDF - much faster than pdf2image
from PIL import Image
import img2pdf
from typing import List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PDFPreprocessor:
    """Handles preprocessing of double-page book scans for OCR"""
    
    def __init__(self, input_folder: str, output_folder: str, save_intermediate: bool = True):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.save_intermediate = save_intermediate
        self.intermediate_folder = Path(output_folder).parent / "intermediate"
        
        logger.info(f"Initializing PDF Preprocessor")
        logger.info(f"  Input folder: {self.input_folder}")
        logger.info(f"  Output folder: {self.output_folder}")
        logger.info(f"  Save intermediate images: {self.save_intermediate}")
        
        if self.save_intermediate:
            self.intermediate_folder.mkdir(parents=True, exist_ok=True)
            logger.info(f"  Intermediate folder: {self.intermediate_folder}")              
        
    def process_all_pdfs(self):
        """Recursively process all PDFs in the input folder"""
        pdf_files = list(self.input_folder.rglob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.input_folder}")
            return
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        logger.info("="*60)
        
        for idx, pdf_path in enumerate(pdf_files, 1):
            try:
                logger.info(f"\n[{idx}/{len(pdf_files)}] Starting to process: {pdf_path.name}")
                self.process_single_pdf(pdf_path)
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {str(e)}", exc_info=True)
    
    def process_single_pdf(self, pdf_path: Path):
        """Process a single PDF file using three-phase approach:
        Phase 1: Detect trapezoids on all pages
        Phase 2: Compute average trapezoid from non-edge-touching pages
        Phase 3: Apply average trapezoid to all pages
        """
        logger.info(f"Processing: {pdf_path}")
        
        # Calculate relative path and create output directory structure
        rel_path = pdf_path.relative_to(self.input_folder)
        output_dir = self.output_folder / rel_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"  Output directory: {output_dir}")
        
        # Create intermediate subfolder for this PDF if enabled
        if self.save_intermediate:
            intermediate_pdf_dir = self.intermediate_folder / rel_path.parent / pdf_path.stem
            intermediate_pdf_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"  Intermediate snapshots: {intermediate_pdf_dir}")
        
        # Check PDF file size
        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        logger.info(f"  PDF file size: {file_size_mb:.2f} MB")
        
        # Convert PDF to images using PyMuPDF (much faster than pdf2image)
        logger.info(f"  Opening PDF and converting pages to images...")
        
        try:
            start_time = time.time()
            
            # Open PDF with PyMuPDF
            pdf_document = fitz.open(str(pdf_path))
            page_count = pdf_document.page_count
            logger.info(f"  PDF has {page_count} page(s)")
            
            # Convert each page to image and grayscale
            images = []
            gray_images = []
            for page_num in range(page_count):
                logger.info(f"  Loading page {page_num + 1}/{page_count}...")
                page = pdf_document[page_num]
                
                # Render page to image at 300 DPI (zoom factor 300/72 = 4.167)
                zoom = 300 / 72  # 300 DPI
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
                
                # Convert to grayscale for processing
                cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                gray_images.append(gray)
            
            pdf_document.close()
            
            elapsed_time = time.time() - start_time
            logger.info(f"  ✓ Conversion complete in {elapsed_time:.1f} seconds")
            logger.info(f"  Converted to {len(images)} page(s)")
        except Exception as e:
            logger.error(f"  Failed to convert PDF: {str(e)}")
            logger.error(f"  Make sure PyMuPDF is installed: pip install PyMuPDF")
            raise
        
        # ==============================================
        # PHASE 1: Detect trapezoids on all pages
        # ==============================================
        logger.info(f"\n  ========================================")
        logger.info(f"  PHASE 1: Detecting trapezoids on all pages")
        logger.info(f"  ========================================")
        
        trapezoids = []
        edge_margin = 50  # pixels
        
        for idx, gray in enumerate(gray_images):
            logger.info(f"\n  Page {idx + 1}/{len(gray_images)}: Detecting trapezoid...")
            
            # Save grayscale snapshot
            if self.save_intermediate:
                snapshot_path = intermediate_pdf_dir / f"scan_{idx + 1:03d}_01_grayscale.png"
                cv2.imwrite(str(snapshot_path), gray)
                logger.info(f"    Saved: {snapshot_path.name}")
            
            # Detect trapezoid
            trapezoid = self.detect_page_trapezoid_simple(gray, intermediate_pdf_dir if self.save_intermediate else None, idx + 1)
            
            if trapezoid is None:
                logger.warning(f"    ⚠ Could not detect trapezoid")
                trapezoids.append(None)
                continue
            
            # Check if trapezoid touches image edges
            touches_edge = self.touches_image_edge(trapezoid, gray.shape, edge_margin)
            
            trapezoids.append({
                'contour': trapezoid,
                'touches_edge': touches_edge,
                'page_idx': idx
            })
            
            status = "⚠ TOUCHES EDGE (excluded)" if touches_edge else "✓ Valid (included)"
            logger.info(f"    {status}")
            corners = trapezoid.reshape(-1, 2)
            logger.info(f"    Corners: {corners.tolist()}")
            
            # Save contour detection snapshot
            if self.save_intermediate:
                debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                color = (0, 0, 255) if touches_edge else (0, 255, 0)  # Red if edge-touching, green if valid
                cv2.drawContours(debug_img, [trapezoid], -1, color, 5)
                snapshot_path = intermediate_pdf_dir / f"scan_{idx + 1:03d}_02_detected_trapezoid.png"
                cv2.imwrite(str(snapshot_path), debug_img)
                logger.info(f"    Saved: {snapshot_path.name}")
        
        # ==============================================
        # PHASE 2: Compute average trapezoid
        # ==============================================
        logger.info(f"\n  ========================================")
        logger.info(f"  PHASE 2: Computing average trapezoid")
        logger.info(f"  ========================================")
        
        valid_trapezoids = [t['contour'] for t in trapezoids if t is not None and not t['touches_edge']]
        
        logger.info(f"  Valid trapezoids (non-edge-touching): {len(valid_trapezoids)}/{len(trapezoids)}")
        
        if len(valid_trapezoids) == 0:
            logger.error(f"  ✗ No valid trapezoids found! Cannot proceed.")
            logger.error(f"  All detected trapezoids touch the image edges.")
            return
        
        average_trapezoid = self.compute_average_trapezoid(valid_trapezoids)
        
        logger.info(f"  ✓ Average trapezoid computed:")
        avg_corners = average_trapezoid.reshape(-1, 2)
        logger.info(f"    Corners: {avg_corners.tolist()}")
        
        # ==============================================
        # PHASE 3: Apply average trapezoid to all pages
        # ==============================================
        logger.info(f"\n  ========================================")
        logger.info(f"  PHASE 3: Applying average trapezoid to all pages")
        logger.info(f"  ========================================")
        
        page_number = 1
        for idx, gray in enumerate(gray_images):
            logger.info(f"\n  Processing scan page {idx + 1}/{len(gray_images)}")
            logger.info(f"    Image size: {gray.shape[1]}x{gray.shape[0]} pixels")
            logger.info(f"    Using average trapezoid for perspective correction")
            
            # STEP 1: Apply perspective correction using average trapezoid
            logger.info(f"    Step 1: Applying perspective correction...")
            corrected_page = self.apply_perspective_correction_to_full_page(gray, average_trapezoid)
            
            if corrected_page is None:
                logger.warning(f"    ⚠ Perspective correction failed, skipping this scan")
                continue
            
            logger.info(f"      ✓ Corrected dimensions: {corrected_page.shape[1]}x{corrected_page.shape[0]} pixels")
            
            # Save perspective corrected snapshot
            if self.save_intermediate:
                snapshot_path = intermediate_pdf_dir / f"scan_{idx + 1:03d}_03_perspective_corrected.png"
                cv2.imwrite(str(snapshot_path), corrected_page)
                logger.info(f"      Saved intermediate: {snapshot_path.name}")
            
            # STEP 2: Cut the corrected page in half (middle split)
            logger.info(f"    Step 2: Splitting corrected page in half...")
            left_page, right_page = self.split_corrected_page(corrected_page)
            
            # STEP 3: Save each half as a separate PDF
            for side_idx, page_img in enumerate([left_page, right_page]):
                side_name = "left" if side_idx == 0 else "right"
                logger.info(f"    Processing {side_name} page (output page {page_number})...")
                logger.info(f"      Page dimensions: {page_img.shape[1]}x{page_img.shape[0]} pixels")
                
                # Save split page snapshot
                if self.save_intermediate:
                    snapshot_path = intermediate_pdf_dir / f"scan_{idx + 1:03d}_04_split_{side_name}.png"
                    cv2.imwrite(str(snapshot_path), page_img)
                    logger.info(f"      Saved intermediate: {snapshot_path.name}")
                
                # Save as single-page PDF
                output_filename = f"{pdf_path.stem}_page_{page_number:04d}.pdf"
                output_path = output_dir / output_filename
                logger.info(f"      Saving as: {output_filename}")
                self.save_as_pdf(page_img, output_path)
                
                page_number += 1
        
        logger.info(f"\n  ✓ Completed: {pdf_path.name} -> {page_number - 1} pages generated")
        logger.info("="*60)
    
    def detect_page_trapezoid_simple(self, gray_image: np.ndarray, debug_dir: Optional[Path] = None, scan_num: int = 0) -> Optional[np.ndarray]:
        """Simple trapezoid detection using morphological operations + contour approximation.
        Returns a 4-point contour representing the page trapezoid (with perspective distortion).
        """
        height, width = gray_image.shape
        logger.info(f"    Analyzing image ({width}x{height})...")
        
        # Threshold to separate page (bright) from background (dark)
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        logger.info(f"    Applied Otsu threshold")
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        logger.info(f"    Applied morphological closing")
        
        # Save threshold image
        if debug_dir is not None:
            snapshot_path = debug_dir / f"scan_{scan_num:03d}_01b_threshold.png"
            cv2.imwrite(str(snapshot_path), binary)
            logger.info(f"    Saved: {snapshot_path.name}")
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logger.info(f"    Found {len(contours)} contours")
        
        if len(contours) == 0:
            logger.warning(f"    No contours found")
            return None
        
        # Find largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        logger.info(f"    Largest contour area: {area:.0f} pixels")
        
        # Approximate contour to a polygon to find the quadrilateral (trapezoid)
        # Use a small epsilon to preserve the actual shape
        perimeter = cv2.arcLength(largest_contour, True)
        epsilon = 0.02 * perimeter  # 2% approximation
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        logger.info(f"    Approximated to {len(approx)} corners")
        
        # If we don't get exactly 4 corners, use convex hull
        if len(approx) != 4:
            hull = cv2.convexHull(largest_contour)
            perimeter_hull = cv2.arcLength(hull, True)
            epsilon_hull = 0.02 * perimeter_hull
            approx = cv2.approxPolyDP(hull, epsilon_hull, True)
            logger.info(f"    Used convex hull, approximated to {len(approx)} corners")
        
        # If still not 4 corners, try more aggressive approximation
        if len(approx) != 4:
            epsilon = 0.05 * perimeter  # 5% approximation
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            logger.info(f"    Aggressive approximation: {len(approx)} corners")
        
        # Last resort: use the 4 extreme points
        if len(approx) != 4:
            logger.info(f"    Using extreme points as fallback")
            # Find extreme points
            leftmost = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
            rightmost = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])
            topmost = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
            bottommost = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])
            
            # Create trapezoid from extreme points
            approx = np.array([[leftmost], [topmost], [rightmost], [bottommost]], dtype=np.int32)
        
        trapezoid = approx
        corners = trapezoid.reshape(-1, 2)
        
        logger.info(f"    Detected trapezoid corners: {corners.tolist()}")
        logger.info(f"    ✓ Trapezoid detected")
        
        return trapezoid
    
    def touches_image_edge(self, trapezoid: np.ndarray, image_shape: tuple, margin: int = 50) -> bool:
        """Check if trapezoid touches the image edge within the specified margin.
        Returns True if any corner is within margin pixels of the edge.
        """
        height, width = image_shape
        corners = trapezoid.reshape(-1, 2)
        
        for corner in corners:
            x, y = corner
            if x < margin or x > width - margin or y < margin or y > height - margin:
                return True
        
        return False
    
    def compute_average_trapezoid(self, trapezoids: list) -> np.ndarray:
        """Compute average trapezoid from a list of trapezoids.
        Uses median of each corner coordinate to be robust to outliers.
        """
        # Extract all corners
        all_corners = [t.reshape(-1, 2) for t in trapezoids]
        
        # Order all trapezoids consistently (top-left, top-right, bottom-right, bottom-left)
        ordered_corners = [self.order_points(corners) for corners in all_corners]
        
        # Stack all corners
        stacked = np.array(ordered_corners)  # Shape: (n_trapezoids, 4, 2)
        
        # Compute median for each corner position
        median_corners = np.median(stacked, axis=0)  # Shape: (4, 2)
        
        # Round to integers
        median_corners = np.int0(median_corners)
        
        # Reshape to contour format
        average_trapezoid = median_corners.reshape(-1, 1, 2)
        
        logger.info(f"  Computed from {len(trapezoids)} trapezoids")
        logger.info(f"  Median corners: {median_corners.tolist()}")
        
        return average_trapezoid
    
    def detect_page_contour(self, gray_image: np.ndarray, debug_dir: Optional[Path] = None, scan_num: int = 0) -> Optional[np.ndarray]:
        """
        STEP 1: Detect the page using projection-based robust boundary detection.
        Finds content concentration, ignoring fingers and edge artifacts.
        Returns the contour of the page with 4 corner points.
        """
        height, width = gray_image.shape
        logger.info(f"      Analyzing image for page detection ({width}x{height})...")
        
        # Threshold to separate page (bright) from background (dark)
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        logger.info(f"      Applied Otsu threshold")
        
        # Find all contours first
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logger.info(f"      Found {len(contours)} initial contours")
        
        # Filter contours: keep only those with linear, well-defined boundaries (rectangles)
        # Discard irregular shapes like hands/fingers
        filtered_binary = np.zeros_like(binary)
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 1000:  # Skip tiny specks
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            bbox_area = w * h
            
            # Get perimeter
            perimeter = cv2.arcLength(contour, True)
            
            # Calculate rectangularity: ratio of contour area to bounding box area
            # Rectangle = 1.0, irregular shape < 0.7
            rectangularity = area / bbox_area if bbox_area > 0 else 0
            
            # Calculate compactness: ratio of area to perimeter^2
            # More compact = more rectangular
            compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Approximate contour to polygon to check for straight edges
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            num_corners = len(approx)
            
            # Keep contours that are rectangular-like:
            # - High rectangularity (> 0.6)
            # - Reasonable number of corners (4-20 for pages with text)
            # - Not too irregular (compactness check)
            if rectangularity > 0.6 and 4 <= num_corners <= 20:
                cv2.drawContours(filtered_binary, [contour], -1, 255, -1)
                logger.info(f"      Kept contour {i}: area={area:.0f}, rect={rectangularity:.3f}, corners={num_corners}, w={w}, h={h}")
            else:
                logger.info(f"      Filtered out contour {i}: area={area:.0f}, rect={rectangularity:.3f}, corners={num_corners} (irregular shape)")
        
        binary = filtered_binary
        logger.info(f"      Applied shape-based filtering to remove irregular boundaries")
        
        # Save filtered threshold image
        if debug_dir is not None:
            snapshot_path = debug_dir / f"scan_{scan_num:03d}_01b_threshold.png"
            cv2.imwrite(str(snapshot_path), binary)
            logger.info(f"      Saved intermediate: {snapshot_path.name}")
        
        # Step 2: Apply histogram analysis to find page plateau + slopes
        # Vertical histogram (sum along columns)
        vertical_hist = np.sum(binary, axis=0) / 255
        # Horizontal histogram (sum along rows)
        horizontal_hist = np.sum(binary, axis=1) / 255
        
        logger.info(f"      Analyzing histograms for main content region...")
        
        # Find main content region in vertical histogram
        # Use a low threshold (20th percentile) to include slopes
        # But find largest gap-free region to exclude scattered fingers
        v_nonzero = vertical_hist[vertical_hist > 0]
        if len(v_nonzero) > 0:
            v_threshold = np.percentile(v_nonzero, 20)  # Low threshold to include slopes
            v_above = np.where(vertical_hist >= v_threshold)[0]
            
            if len(v_above) > 0:
                # Find the largest continuous region (page) vs scattered regions (fingers)
                # Look for gaps larger than 100 pixels
                gaps = np.diff(v_above)
                large_gaps = np.where(gaps > 100)[0]  # Gaps larger than 100 pixels = separate objects
                
                if len(large_gaps) > 0:
                    # Multiple segments exist - take the widest one (should be page)
                    segments = np.split(v_above, large_gaps + 1)
                    widest_segment = max(segments, key=len)
                    left_bound = widest_segment[0]
                    right_bound = widest_segment[-1]
                    logger.info(f"      Found {len(segments)} vertical segments, using widest ({len(widest_segment)} columns)")
                else:
                    # No large gaps, use full range
                    left_bound = v_above[0]
                    right_bound = v_above[-1]
            else:
                left_bound, right_bound = 0, width - 1
        else:
            left_bound, right_bound = 0, width - 1
        
        # Find main content region in horizontal histogram
        h_nonzero = horizontal_hist[horizontal_hist > 0]
        if len(h_nonzero) > 0:
            h_threshold = np.percentile(h_nonzero, 20)  # Low threshold to include slopes
            h_above = np.where(horizontal_hist >= h_threshold)[0]
            
            if len(h_above) > 0:
                # Find the tallest continuous region
                gaps = np.diff(h_above)
                large_gaps = np.where(gaps > 100)[0]
                
                if len(large_gaps) > 0:
                    segments = np.split(h_above, large_gaps + 1)
                    tallest_segment = max(segments, key=len)
                    top_bound = tallest_segment[0]
                    bottom_bound = tallest_segment[-1]
                    logger.info(f"      Found {len(segments)} horizontal segments, using tallest ({len(tallest_segment)} rows)")
                else:
                    top_bound = h_above[0]
                    bottom_bound = h_above[-1]
            else:
                top_bound, bottom_bound = 0, height - 1
        else:
            top_bound, bottom_bound = 0, height - 1
        
        logger.info(f"      Rectangle from histogram (longest continuous region):")
        logger.info(f"        Left: x={left_bound}, Right: x={right_bound}, Width: {right_bound - left_bound}")
        logger.info(f"        Top: y={top_bound}, Bottom: y={bottom_bound}, Height: {bottom_bound - top_bound}")
        
        # Create axis-aligned rectangular contour (parallel to screen)
        approx = np.array([
            [[left_bound, top_bound]],
            [[right_bound, top_bound]],
            [[right_bound, bottom_bound]],
            [[left_bound, bottom_bound]]
        ], dtype=np.int32)
        
        logger.info(f"      Rectangle corners: {approx.reshape(-1, 2).tolist()}")
        
        # Save visualization
        if debug_dir is not None:
            vis_img = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(vis_img, [approx], 0, (0, 255, 0), 5)
            snapshot_path = debug_dir / f"scan_{scan_num:03d}_01c_detected_bounds.png"
            cv2.imwrite(str(snapshot_path), vis_img)
            logger.info(f"      Saved intermediate: {snapshot_path.name}")
        
        logger.info(f"      ✓ Detected page using histogram (longest continuous region) → axis-aligned rectangle")
        return approx
    
    def _fallback_detection(self, gray_image: np.ndarray) -> Optional[np.ndarray]:
        """Fallback to simple bounding box when detection fails"""
        height, width = gray_image.shape
        
        # Use a conservative inset (10% from each edge)
        margin_x = int(width * 0.1)
        margin_y = int(height * 0.1)
        
        approx = np.array([
            [[margin_x, margin_y]],
            [[width - margin_x, margin_y]],
            [[width - margin_x, height - margin_y]],
            [[margin_x, height - margin_y]]
        ], dtype=np.int32)
        
        logger.info(f"      Using fallback rectangular detection with 10% margins")
        return approx
    
    def apply_perspective_correction_to_full_page(self, image: np.ndarray, contour: np.ndarray) -> Optional[np.ndarray]:
        """
        STEP 2: Apply perspective correction to transform trapezoid to rectangle.
        Takes the 4-point contour and warps it to a proper rectangle.
        """
        if contour is None or len(contour) != 4:
            logger.warning("      ⚠ Invalid contour for perspective correction")
            return None
        
        # Extract corner points and convert to float32
        points = contour.reshape(4, 2).astype(np.float32)
        
        # Order points: top-left, top-right, bottom-right, bottom-left
        points = self.order_points(points)
        
        logger.info(f"      Ordering corners:")
        logger.info(f"        Top-left: ({points[0][0]:.0f}, {points[0][1]:.0f})")
        logger.info(f"        Top-right: ({points[1][0]:.0f}, {points[1][1]:.0f})")
        logger.info(f"        Bottom-right: ({points[2][0]:.0f}, {points[2][1]:.0f})")
        logger.info(f"        Bottom-left: ({points[3][0]:.0f}, {points[3][1]:.0f})")
        
        # Calculate the width of the corrected image (max of top and bottom edge lengths)
        width_top = np.linalg.norm(points[0] - points[1])
        width_bottom = np.linalg.norm(points[3] - points[2])
        max_width = int(max(width_top, width_bottom))
        
        # Calculate the height of the corrected image (max of left and right edge lengths)
        height_left = np.linalg.norm(points[0] - points[3])
        height_right = np.linalg.norm(points[1] - points[2])
        max_height = int(max(height_left, height_right))
        
        logger.info(f"      Calculating output dimensions:")
        logger.info(f"        Width: top={width_top:.0f}, bottom={width_bottom:.0f} -> using {max_width}")
        logger.info(f"        Height: left={height_left:.0f}, right={height_right:.0f} -> using {max_height}")
        
        # Define destination points for the corrected rectangle
        dst_points = np.array([
            [0, 0],                          # top-left
            [max_width - 1, 0],              # top-right
            [max_width - 1, max_height - 1], # bottom-right
            [0, max_height - 1]              # bottom-left
        ], dtype=np.float32)
        
        # Calculate perspective transform matrix
        matrix = cv2.getPerspectiveTransform(points, dst_points)
        
        # Apply perspective transformation
        corrected = cv2.warpPerspective(image, matrix, (max_width, max_height))
        
        logger.info(f"      ✓ Perspective correction applied -> {max_width}x{max_height} pixels")
        
        return corrected
    
    def split_corrected_page(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        STEP 3: Split the corrected rectangle in half vertically at the spine/gutter.
        Finds the spine pattern: white gutter → dark spine → white gutter.
        In histogram: high values (white) → low values (dark spine) → high values (white).
        Returns (left_page, right_page).
        """
        height, width = image.shape
        center_x = width // 2
        
        # Search for spine in middle 40% of image (30%-70% of width)
        search_start = int(width * 0.3)
        search_end = int(width * 0.7)
        search_region = image[:, search_start:search_end]
        
        # Compute vertical histogram: sum of pixel intensities along each column
        # High values = white (gutter), low values = dark (spine or text)
        vertical_hist = np.sum(search_region, axis=0).astype(float)
        
        # Normalize histogram to 0-1 range for easier threshold calculation
        hist_min = np.min(vertical_hist)
        hist_max = np.max(vertical_hist)
        if hist_max > hist_min:
            vertical_hist_norm = (vertical_hist - hist_min) / (hist_max - hist_min)
        else:
            vertical_hist_norm = vertical_hist
        
        logger.info(f"      Searching for spine pattern (x={search_start}-{search_end})")
        
        # Smooth histogram to reduce noise
        smoothed_hist = uniform_filter1d(vertical_hist_norm, size=10)
        
        # Find valleys (low points) in the histogram - these are dark regions (spine or text columns)
        # But we want the valley that's surrounded by high values (white gutters)
        
        # Define threshold: values below 0.5 are considered "dark" (spine candidate)
        # values above 0.7 are considered "white" (gutter)
        dark_threshold = 0.5
        white_threshold = 0.7
        
        # Find dark regions (potential spine)
        dark_mask = smoothed_hist < dark_threshold
        
        # Find contiguous dark regions
        dark_regions = []
        in_dark = False
        start_idx = 0
        
        for i, is_dark in enumerate(dark_mask):
            if is_dark and not in_dark:
                start_idx = i
                in_dark = True
            elif not is_dark and in_dark:
                dark_regions.append((start_idx, i))
                in_dark = False
        if in_dark:
            dark_regions.append((start_idx, len(dark_mask)))
        
        logger.info(f"      Found {len(dark_regions)} dark regions")
        
        # Score each dark region based on:
        # 1. How much white (high values) surround it on both sides
        # 2. How close it is to center
        best_score = -1
        best_region = None
        
        for start, end in dark_regions:
            region_center = (start + end) // 2
            region_center_abs = search_start + region_center
            
            # Check whiteness on left side (20 pixels before)
            left_check_start = max(0, start - 20)
            left_check_end = start
            left_whiteness = np.mean(smoothed_hist[left_check_start:left_check_end]) if left_check_end > left_check_start else 0
            
            # Check whiteness on right side (20 pixels after)
            right_check_start = end
            right_check_end = min(len(smoothed_hist), end + 20)
            right_whiteness = np.mean(smoothed_hist[right_check_start:right_check_end]) if right_check_end > right_check_start else 0
            
            # Both sides should be white (gutter)
            whiteness_score = min(left_whiteness, right_whiteness)
            
            # Distance from center (closer is better)
            distance_from_center = abs(region_center_abs - center_x)
            distance_score = 1.0 / (1.0 + distance_from_center / 100.0)
            
            # Combined score
            score = whiteness_score * 0.7 + distance_score * 0.3
            
            logger.info(f"        Region {start}-{end} (x={search_start + start}-{search_start + end}): whiteness={whiteness_score:.2f}, distance={distance_from_center}px, score={score:.2f}")
            
            if score > best_score:
                best_score = score
                best_region = (start, end)
        
        if best_region is not None:
            start, end = best_region
            spine_x = search_start + (start + end) // 2
            logger.info(f"      Selected spine region: x={search_start + start}-{search_start + end}, cutting at x={spine_x}")
        else:
            # Fallback: use center
            spine_x = center_x
            logger.info(f"      No spine pattern found, using center: x={spine_x}")
        
        # Split at the detected spine
        left_page = image[:, :spine_x]
        right_page = image[:, spine_x:]
        
        logger.info(f"      Split: left={left_page.shape[1]}x{left_page.shape[0]}, right={right_page.shape[1]}x{right_page.shape[0]}")
        
        return left_page, right_page
    
    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Order points in the correct sequence:
        [0] = top-left, [1] = top-right, [2] = bottom-right, [3] = bottom-left
        
        Strategy: 
        - Sum coordinates: top-left has smallest sum, bottom-right has largest sum
        - Diff coordinates: top-right has smallest diff (x-y), bottom-left has largest diff
        """
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Sum of coordinates
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left (smallest sum)
        rect[2] = pts[np.argmax(s)]  # bottom-right (largest sum)
        
        # Difference of coordinates
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right (smallest difference)
        rect[3] = pts[np.argmax(diff)]  # bottom-left (largest difference)
        
        return rect
    
    def save_as_pdf(self, image: np.ndarray, output_path: Path):
        """Save image as a single-page PDF"""
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        # Convert PIL image to bytes in memory
        from io import BytesIO
        img_bytes = BytesIO()
        pil_image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Convert to PDF
        with open(output_path, 'wb') as f:
            pdf_bytes = img2pdf.convert(img_bytes.read())
            f.write(pdf_bytes)
        
        file_size_kb = output_path.stat().st_size / 1024
        logger.info(f"        ✓ Saved: {output_path.name} ({file_size_kb:.1f} KB)")


def main():
    """Main entry point"""
    # Define input and output folders
    script_dir = Path(__file__).parent
    input_folder = script_dir / "input"
    output_folder = script_dir / "output"
    
    # Toggle for saving intermediate processing snapshots (set to False to disable)
    SAVE_INTERMEDIATE_SNAPSHOTS = True
    
    # Check if input folder exists
    if not input_folder.exists():
        logger.error(f"Input folder does not exist: {input_folder}")
        logger.info("Creating input folder. Please add PDF files to process.")
        input_folder.mkdir(parents=True, exist_ok=True)
        return
    
    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output folder ready: {output_folder}\n")
    
    # Create preprocessor and run
    preprocessor = PDFPreprocessor(str(input_folder), str(output_folder), 
                                   save_intermediate=SAVE_INTERMEDIATE_SNAPSHOTS)
    preprocessor.process_all_pdfs()
    
    logger.info("\n" + "="*60)
    logger.info("✓ All processing complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()

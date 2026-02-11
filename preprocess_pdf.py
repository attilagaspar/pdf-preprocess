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
        """Process a single PDF file"""
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
            
            # Convert each page to image
            images = []
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
            
            pdf_document.close()
            
            elapsed_time = time.time() - start_time
            logger.info(f"  ✓ Conversion complete in {elapsed_time:.1f} seconds")
            logger.info(f"  Converted to {len(images)} page(s)")
        except Exception as e:
            logger.error(f"  Failed to convert PDF: {str(e)}")
            logger.error(f"  Make sure PyMuPDF is installed: pip install PyMuPDF")
            raise
        
        page_number = 1
        for idx, img in enumerate(images):
            logger.info(f"\n  Processing scan page {idx + 1}/{len(images)}")
            logger.info(f"    Image size: {img.size[0]}x{img.size[1]} pixels")
            
            # Convert PIL image to OpenCV format
            logger.info(f"    Converting to OpenCV format...")
            cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            logger.info(f"    Converting to grayscale...")
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Save grayscale snapshot
            if self.save_intermediate:
                snapshot_path = intermediate_pdf_dir / f"scan_{idx + 1:03d}_01_grayscale.png"
                cv2.imwrite(str(snapshot_path), gray)
                logger.info(f"      Saved intermediate: {snapshot_path.name}")
            
            # STEP 1: Detect the rectangular/trapezoid page shape in the dark background
            logger.info(f"    Step 1: Detecting page contour...")
            page_contour = self.detect_page_contour(gray)
            
            if page_contour is None:
                logger.warning(f"    ⚠ Could not detect page contour, skipping this scan")
                continue
            
            # Save contour detection snapshot
            if self.save_intermediate:
                debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(debug_img, [page_contour], -1, (0, 255, 0), 5)
                snapshot_path = intermediate_pdf_dir / f"scan_{idx + 1:03d}_02_detected_contour.png"
                cv2.imwrite(str(snapshot_path), debug_img)
                logger.info(f"      Saved intermediate: {snapshot_path.name}")
            
            # STEP 2: Apply perspective correction (trapezoid -> rectangle)
            logger.info(f"    Step 2: Applying perspective correction...")
            corrected_page = self.apply_perspective_correction_to_full_page(gray, page_contour)
            
            if corrected_page is None:
                logger.warning(f"    ⚠ Perspective correction failed, skipping this scan")
                continue
            
            logger.info(f"      ✓ Corrected dimensions: {corrected_page.shape[1]}x{corrected_page.shape[0]} pixels")
            
            # Save perspective corrected snapshot
            if self.save_intermediate:
                snapshot_path = intermediate_pdf_dir / f"scan_{idx + 1:03d}_03_perspective_corrected.png"
                cv2.imwrite(str(snapshot_path), corrected_page)
                logger.info(f"      Saved intermediate: {snapshot_path.name}")
            
            # STEP 3: Cut the corrected page in half (middle split)
            logger.info(f"    Step 3: Splitting corrected page in half...")
            left_page, right_page = self.split_corrected_page(corrected_page)
            
            # STEP 4: Save each half as a separate PDF
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
    
    def detect_page_contour(self, gray_image: np.ndarray) -> Optional[np.ndarray]:
        """
        STEP 1: Detect the rectangular/trapezoid page shape in the mostly black background.
        Returns the contour of the page with 4 corner points.
        """
        height, width = gray_image.shape
        logger.info(f"      Analyzing image for page detection ({width}x{height})...")
        
        # Apply binary threshold - invert so page (light) becomes white, background (dark) becomes black
        # Using Otsu's method for automatic threshold
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        logger.info(f"      Applied binary threshold (Otsu)")
        
        # Morphological operations to clean up noise and connect page edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        logger.info(f"      Applied morphological operations")
        
        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logger.info(f"      Found {len(contours)} contours")
        
        if len(contours) == 0:
            logger.warning("      ⚠ No contours found")
            return None
        
        # Find the largest contour (should be the page)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        area_percent = (area / (width * height)) * 100
        logger.info(f"      Largest contour: {area:.0f} pixels ({area_percent:.1f}% of image)")
        
        # The page should be a significant portion of the image (at least 20%)
        if area_percent < 20:
            logger.warning(f"      ⚠ Contour too small (expected >20% of image)")
            return None
        
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        logger.info(f"      Approximated contour to {len(approx)} points")
        
        # If we don't have exactly 4 points, try with different epsilon values
        if len(approx) != 4:
            for eps_multiplier in [0.01, 0.03, 0.04, 0.05]:
                epsilon = eps_multiplier * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                if len(approx) == 4:
                    logger.info(f"      ✓ Found 4 corners with epsilon={eps_multiplier}")
                    break
        
        if len(approx) != 4:
            # If still not 4 points, use bounding box as fallback
            logger.warning(f"      ⚠ Could not approximate to 4 corners (got {len(approx)}), using bounding box")
            x, y, w, h = cv2.boundingRect(largest_contour)
            approx = np.array([
                [[x, y]],
                [[x + w, y]],
                [[x + w, y + h]],
                [[x, y + h]]
            ], dtype=np.int32)
        
        logger.info(f"      ✓ Detected page contour with 4 corners")
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
        STEP 3: Split the corrected rectangle in half vertically.
        Returns (left_page, right_page).
        """
        height, width = image.shape
        mid_x = width // 2
        
        left_page = image[:, :mid_x]
        right_page = image[:, mid_x:]
        
        logger.info(f"      Split at x={mid_x}: left={left_page.shape[1]}x{left_page.shape[0]}, right={right_page.shape[1]}x{right_page.shape[0]}")
        
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

# PDF Preprocessor for OCR

A tool for preprocessing double-page book scans to prepare them for OCR.

## Features

- **Recursive Processing**: Processes all PDFs in the input folder and subfolders
- **Color to B&W Conversion**: High-quality grayscale conversion with adaptive thresholding
- **Intelligent Page Detection**: Automatically detects page regions and removes blank areas and unwanted objects
- **Double-Page Splitting**: Finds the center of double-page scans and splits them into two separate pages
- **Perspective Correction**: Applies trapezoid correction to straighten skewed pages
- **Folder Structure Preservation**: Maintains the same folder structure in the output folder
- **Intermediate Snapshots**: Saves processing snapshots in `intermediate` folder for debugging (toggleable)

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

Note: This tool uses PyMuPDF (fitz) which is much faster than pdf2image for large PDFs and doesn't require Poppler installation.

## Usage

1. Place your PDF files in the `input` folder (you can create subfolders)
2. (Optional) Edit `preprocess_pdf.py` and set `SAVE_INTERMEDIATE_SNAPSHOTS = True` (default) or `False`
3. Run the preprocessor:
```bash
python preprocess_pdf.py
```
4. Processed files will be saved in the `output` folder with the same folder structure
5. If enabled, intermediate processing snapshots will be in the `intermediate` folder

## Intermediate Snapshots

When `SAVE_INTERMEDIATE_SNAPSHOTS = True` (default), the tool saves PNG snapshots at each processing stage:
- `01_grayscale.png` - Initial grayscale conversion
- `02_split_left/right.png` - Split pages after detecting double pages
- `03_corrected_left/right.png` - After perspective correction
- `04_bw_left/right.png` - Final black & white result

This helps verify the processing is working correctly without waiting for the entire PDF to complete. Set to `False` to disable and speed up processing.

## How It Works

For each PDF file:
1. Converts each page to a high-resolution image (300 DPI)
2. Converts to grayscale
3. Detects page regions using contour detection
4. Removes blank areas and unwanted objects
5. Splits double pages at the center
6. Applies perspective correction to each page
7. Converts to high-quality black and white using adaptive thresholding
8. Saves each page as a separate single-page PDF

## Output

Each input PDF generates multiple output PDFs:
- Original: `input/book.pdf`
- Output: `output/book_page_0001.pdf`, `output/book_page_0002.pdf`, etc.

## Troubleshooting

- If pages aren't detected correctly, check that the scan quality is good and pages have clear borders
- If perspective correction fails, the original (uncorrected) page will be saved
- Check the console output for detailed logging information

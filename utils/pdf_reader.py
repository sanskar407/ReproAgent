"""
PDF text extraction utilities.
Supports multiple extraction methods with fallbacks.
"""

from pathlib import Path
from typing import Optional


def extract_text_pypdf(pdf_path: str) -> str:
    """
    Extract text using PyPDF2 (faster, less accurate).
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted text
    """
    try:
        import PyPDF2
        
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        return f"Error extracting PDF with PyPDF2: {str(e)}"


def extract_text_pdfplumber(pdf_path: str, max_pages: int = 10) -> str:
    """
    Extract text using pdfplumber (slower, more accurate).
    
    Args:
        pdf_path: Path to PDF file
        max_pages: Maximum pages to extract (for speed)
        
    Returns:
        Extracted text
    """
    try:
        import pdfplumber
        
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages[:max_pages]):
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error extracting PDF with pdfplumber: {str(e)}"


def extract_text(
    pdf_path: str, 
    method: str = "auto", 
    max_pages: int = 10
) -> str:
    """
    Extract text from PDF using specified method.
    
    Args:
        pdf_path: Path to PDF file
        method: 'pypdf', 'pdfplumber', or 'auto' (try both)
        max_pages: Max pages to extract
        
    Returns:
        Extracted text
    """
    if not Path(pdf_path).exists():
        return f"Error: File not found: {pdf_path}"
    
    if method == "auto":
        # Try pdfplumber first (more accurate)
        text = extract_text_pdfplumber(pdf_path, max_pages)
        
        if not text.startswith("Error"):
            return text
        
        # Fallback to PyPDF2
        print("⚠️  pdfplumber failed, trying PyPDF2...")
        text = extract_text_pypdf(pdf_path)
        
        if not text.startswith("Error"):
            return text
        
        return "Error: All PDF extraction methods failed"
    
    elif method == "pypdf":
        return extract_text_pypdf(pdf_path)
    
    elif method == "pdfplumber":
        return extract_text_pdfplumber(pdf_path, max_pages)
    
    else:
        return f"Error: Unknown method: {method}"


# Test function
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        text = extract_text(pdf_path)
        print(text[:1000])  # First 1000 chars
        print(f"\n... (Total length: {len(text)} characters)")
    else:
        print("Usage: python pdf_reader.py <pdf_path>")

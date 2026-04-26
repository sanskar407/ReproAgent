import fitz  # PyMuPDF

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text from a PDF file using PyMuPDF.
    """
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""
    
    return text

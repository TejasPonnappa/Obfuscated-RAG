"""Turns an uploaded file into plain text. Kept free of Streamlit so it can be tested."""
import io


class NoTextError(ValueError):
    """The file has no extractable text (for a PDF: probably a scan, which would need OCR)."""


def extract_text(filename, data):
    """Text of a .txt or .pdf upload. Raises NoTextError for a PDF without a text layer."""
    if filename.lower().endswith(".pdf"):
        from pypdf import PdfReader                   # imported here so plain-text use needs nothing extra
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                raise NoTextError("This PDF is password-protected. Remove the password first.")
            pages = [(page.extract_text() or "") for page in reader.pages]
        except NoTextError:
            raise
        except Exception as e:                    # a damaged file or one that is not really a PDF
            raise NoTextError(f"Could not read this PDF ({type(e).__name__}). It may be damaged, or not a real PDF.") from e
        text = "\n\n".join(p.strip() for p in pages if p.strip())
        if not text.strip():
            raise NoTextError("This PDF has no selectable text (it may be a scan). Export it as text or run OCR first.")
        return text
    return data.decode("utf-8", errors="replace")

import pytest

pytest.importorskip("pypdf")

from documents import NoTextError, extract_text
from eval.pdf_sample import make_pdf


def test_text_files_are_decoded():
    assert extract_text("a.txt", "caf\u00e9 salary $5,000".encode("utf-8")) == "caf\u00e9 salary $5,000"


def test_pdf_text_is_extracted_across_pages():
    text = "\n".join(f"Line {i}: Aisha Bello earns $97,250 (see note)." for i in range(120))      # forces several pages
    out = extract_text("hr.PDF", make_pdf(text))
    assert "Aisha Bello earns $97,250 (see note)." in out and out.count("Line ") == 120


def test_a_pdf_without_text_raises_a_clear_error():
    with pytest.raises(NoTextError):
        extract_text("scan.pdf", make_pdf(""))


def test_a_file_that_is_not_a_pdf_gives_a_friendly_error_not_a_stack_trace():
    with pytest.raises(NoTextError, match="Could not read this PDF"):
        extract_text("fake.pdf", b"just some text, not a pdf")

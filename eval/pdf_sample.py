"""
A minimal PDF writer (text only, Helvetica, A4, wrapped lines) so the repo can make sample PDFs without a PDF library.
    python -m eval.pdf_sample samples/northwind_hr_records.txt samples/northwind_hr_records.pdf
"""
import sys
import textwrap


def _escape(line):
    return line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(text, width=95, lines_per_page=52):
    lines = []
    for raw in text.splitlines():
        lines.extend(textwrap.wrap(raw, width) or [""])
    pages = [lines[i:i + lines_per_page] for i in range(0, max(len(lines), 1), lines_per_page)]
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"", b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for page in pages:
        stream = "BT /F1 10 Tf 50 800 Td 14 TL\n" + "\n".join(f"({_escape(l.encode('latin-1', 'replace').decode('latin-1'))}) '" for l in page) + "\nET"
        content_id = len(objects) + 2
        kids.append(f"{len(objects) + 1} 0 R")
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents {content_id} 0 R "
                       f"/Resources << /Font << /F1 3 0 R >> >> >>".encode())
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream".encode("latin-1"))
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>".encode()
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    open(dst, "wb").write(make_pdf(open(src, encoding="utf-8").read()))
    print("wrote", dst)

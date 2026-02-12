import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import os
import io

# Explicit Tesseract path (Windows fix)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

PDF_DIR = "data/pdfs"
OUTPUT_DIR = "data/extracted_text"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages_data = []

    for page_no, page in enumerate(doc, start=1):
        text = page.get_text().strip()

        # If very little text, assume scanned page → OCR
        if len(text) < 100:
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr_text = pytesseract.image_to_string(img)
            text = text + "\n" + ocr_text

        pages_data.append({
            "page": page_no,
            "text": text
        })

    return pages_data

def run():
    for file in os.listdir(PDF_DIR):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(PDF_DIR, file)
            pages = extract_text_from_pdf(pdf_path)

            output_file = os.path.join(
                OUTPUT_DIR, file.replace(".pdf", ".txt")
            )

            with open(output_file, "w", encoding="utf-8") as f:
                for p in pages:
                    f.write(f"\n--- Page {p['page']} ---\n")
                    f.write(p["text"])

            print(f"✅ Extracted text from: {file}")

if __name__ == "__main__":
    run()

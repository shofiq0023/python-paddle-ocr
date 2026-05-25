import logging
import os
import re
import threading
from flask import Flask, request, jsonify
from paddleocr import PaddleOCR
import cv2
import numpy as np
import fitz  # PyMuPDF

# Suppress verbose C++ / PaddlePaddle logs
# Must be set BEFORE PaddleOCR is imported
os.environ["GLOG_minloglevel"]    = "3"
os.environ["PADDLE_CPP_LOG_LEVEL"] = "3"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("ppocr").setLevel(logging.WARNING)

app = Flask(__name__)

ocr = PaddleOCR(
    use_doc_orientation_classify=False,  # form is already upright
    use_doc_unwarping=False,             # scanned PDF is flat
    use_textline_orientation=False,      # all lines are normal orientation
    lang="en",
    device="cpu",
)

# Thread safety lock
# PaddleOCR 3.x has a confirmed bug: ocr.predict() is NOT thread-safe.
# Flask handles each HTTP request in a new thread, so without this lock,
# every 2nd request fails with RuntimeError: Unknown exception.
# The lock serialises all inference calls - OCR is CPU-bound anyway,
# so there is no performance loss from single-threading inference.
# Ref: https://github.com/PaddlePaddle/PaddleOCR/issues/16238
_ocr_lock = threading.Lock()

# Label -> snake_case output key
# Keys are lowercase; aliases handle OCR misreads and apostrophe variants.
FIELD_MAP = {
    "customer name":                "customer_name",
    "date of birth":                "date_of_birth",
    "national id":                  "national_id",
    "tin":                          "tin",
    "passport no":                  "passport_number",
    "passport number":              "passport_number",
    "father's name":                "father_name",
    "father name":                  "father_name",
    "mothers name":                 "mother_name",
    "mother's name":                "mother_name",
    "mother name":                  "mother_name",
    "spouse name":                  "spouse_name",
    "loan type":                    "loan_type",
    "applied loan amount":          "applied_loan_amount",
    "monthly income":               "monthly_income",
    "monthly expense":              "monthly_expense",
    "profession":                   "profession",
    "customer's permanent address": "permanent_address",
    "customers permanent address":  "permanent_address",
    "permanent address":            "permanent_address",
    "customer's present address":   "present_address",
    "customers present address":    "present_address",
    "present address":              "present_address",
    "mobile no":                    "mobile_number",
    "mobile number":                "mobile_number",
}

# Sorted longest-first so "customer's permanent address" matches before "address"
_SORTED_LABELS = sorted(FIELD_MAP.keys(), key=len, reverse=True)

# Header row exclusion
# These texts appear in the VALUE column header row ("Insert in Bold letter").
# Filter them out before spatial matching so they never become field values.
_HEADER_TEXTS = {
    "insert in bold letter",
    "insert in bold",
    "in bold letter",
    "particulars",
    "bold letter",
}

# PDF rendering
def pdf_to_image(pdf_bytes: bytes) -> np.ndarray:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    
    # 300 DPI from 72 base
    mat  = fitz.Matrix(300 / 72, 300 / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
    doc.close()

    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)     # PaddleOCR expects BGR


# Image Preprocessing
def preprocess(img: np.ndarray) -> np.ndarray:
    # Light sharpen only
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


# OCR runner
def run_ocr(img: np.ndarray) -> list[dict]:
    """
    Call PaddleOCR 3.x predict() and normalise results into a flat list:
      [{ text, confidence, x, y, width, height }, ...]
    sorted top-to-bottom, left-to-right.

    3.x result structure:
      ocr.predict(img) → list of result objects
      each result["res"] contains:
        dt_polys   : list of 4-point polygons  [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
        rec_texts  : list of recognised strings
        rec_scores : list of confidence floats
    """
    # Acquire lock before calling predict() — see _ocr_lock comment above
    with _ocr_lock:
        raw_results = ocr.predict(img)
    lines = []

    if not raw_results:
        return lines

    for res in raw_results:
        try:
            # res may be a dict or a dict-like object
            r = res["res"] if "res" in res else res
            polys = r["dt_polys"]
            texts = r["rec_texts"]
            scores = r["rec_scores"]
        except (KeyError, TypeError) as e:
            logger.warning("Skipping unexpected result structure: %s — %s", type(res), e)
            continue

        for poly, text, score in zip(polys, texts, scores):
            pts = np.array(poly)
            x = int(pts[:, 0].min())
            y = int(pts[:, 1].min())
            w = int(pts[:, 0].max() - x)
            h = int(pts[:, 1].max() - y)
            lines.append({
                "text": text.strip(),
                "confidence": round(float(score), 3),
                "x": x, "y": y, "width": w, "height": h,
            })

    lines.sort(key=lambda l: (l["y"], l["x"]))
    return lines


# Column-split detection
def detect_column_split(lines: list[dict], page_width: int) -> int:
    """
    Find the x boundary between the label column (left) and value column (right).

    PaddleOCR 3.x returns lines interleaved from both columns.
    We find the largest horizontal gap in x-start positions within the
    central 15%-65% band of the page — that gap IS the column divider.
    """
    band_xs = sorted({l["x"] for l in lines
                      if page_width * 0.15 < l["x"] < page_width * 0.65})

    if len(band_xs) < 2:
        return int(page_width * 0.38)

    best_gap, split_x = 0, int(page_width * 0.38)
    for i in range(1, len(band_xs)):
        gap = band_xs[i] - band_xs[i - 1]
        if gap > best_gap:
            best_gap = gap
            split_x  = (band_xs[i - 1] + band_xs[i]) // 2

    logger.info("Column split detected at x=%d / page_width=%d", split_x, page_width)
    return split_x


# Label matcher
def match_label(raw_text: str) -> str | None:
    """
    Map an OCR label-column string to a snake_case output key.
    Normalises apostrophes and strips punctuation noise before matching.
    Returns None if no known field matches.
    """

    # Normalise: lowercase, strip trailing colon/pipe, collapse apostrophe variants
    clean = raw_text.lower().strip()
    clean = re.sub(r"[''`]", "'", clean)   # curly/backtick apostrophes → straight
    clean = re.sub(r"[:|]+$", "", clean).strip()

    for label in _SORTED_LABELS:
        if label == clean or label in clean or clean in label:
            return FIELD_MAP[label]

    return None


# Form parser
def parse_form(lines: list[dict], page_width: int) -> dict:
    """
    Pure spatial two-column parser.

    Algorithm:
      1. Split all OCR lines into label_col (x < split) and value_col (x >= split).
      2. For each label line that matches a known field:
           a. Find all value-column lines whose y-centre falls within ±1.8x the
              label's height (i.e. same table row).
           b. Join them left→right as the field value.
      3. Noise characters (•, |, -) are stripped from values.

    Why purely spatial instead of text-anchor slicing:
      PaddleOCR 3.x returns lines in reading order per-column, not strictly
      top-to-bottom across both columns. Flattening to a single string causes
      label and value text to interleave, breaking substring slicing.
    """
    split_x = detect_column_split(lines, page_width)

    # Find header row y-position so we can exclude it
    # "Particulars" is the left-column header; "Insert in Bold letter" is the
    # right-column header. They sit in the topmost row of the table.
    # Any value-column line at that y-band must be excluded from data matching.
    header_ys  = {
        l["y"] for l in lines
        if l["text"].lower().strip() in _HEADER_TEXTS
        or "particulars" in l["text"].lower()
        or "insert in bold" in l["text"].lower()
    }

    # Also compute a minimum y-threshold: skip anything above the first data row.
    # The header row height is typically ~40-60px at 300 DPI; add a small buffer.
    min_data_y = (max(header_ys) + 30) if header_ys else 0
    logger.info("Header y-positions: %s  →  min_data_y=%d", header_ys, min_data_y)

    label_lines = [l for l in lines if l["x"] <  split_x and l["y"] > min_data_y]
    value_lines = [
        l for l in lines
        if l["x"] >= split_x
        and l["y"] > min_data_y
        and l["text"].lower().strip() not in _HEADER_TEXTS   # explicit text guard
        and "insert in bold" not in l["text"].lower()
    ]

    logger.info("Label col (%d lines): %s", len(label_lines),
                [l["text"] for l in label_lines])
    logger.info("Value col (%d lines): %s", len(value_lines),
                [l["text"] for l in value_lines])

    result = {}
    used_value_ys = set()   # prevent the same value line from being claimed twice

    for lbl in label_lines:
        key = match_label(lbl["text"])
        if not key or key in result:
            continue

        label_y   = lbl["y"]
        label_h   = max(lbl["height"], 18)
        tolerance = label_h * 1.8

        candidates = [
            v for v in value_lines
            if abs(v["y"] - label_y) <= tolerance
            and v["y"] not in used_value_ys
        ]

        # Widen search if nothing found (some rows have more whitespace)
        if not candidates:
            tolerance  = label_h * 3.5
            candidates = [
                v for v in value_lines
                if abs(v["y"] - label_y) <= tolerance
                and v["y"] not in used_value_ys
            ]

        if not candidates:
            continue

        candidates.sort(key=lambda v: v["x"])
        raw_value = " ".join(v["text"] for v in candidates)

        # Clean noise characters
        value = re.sub(r"^[•·|\-–\s]+", "", raw_value)
        value = re.sub(r"[•·|\-–\s]+$", "", value)
        value = re.sub(r"\s{2,}",        " ", value).strip()

        if value:
            result[key] = value
            for v in candidates:
                used_value_ys.add(v["y"])

    return result


# Flask routes

@app.route("/ocr/form", methods=["POST"])
def ocr_form():
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Multipart field name must be 'file'."}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are accepted."}), 400

    try:
        pdf_bytes = file.read()
        logger.info("Received: %s  (%d bytes)", file.filename, len(pdf_bytes))

        img = pdf_to_image(pdf_bytes)
        page_width = img.shape[1]
        logger.info("Rendered: %dx%d px", page_width, img.shape[0])

        img = preprocess(img)
        lines = run_ocr(img)
        logger.info("OCR lines detected: %d", len(lines))
        logger.info("Raw OCR:\n%s", "\n".join(
            f"  [{l['x']:4d},{l['y']:4d}] {l['text']}" for l in lines))

        extracted = parse_form(lines, page_width)
        logger.info("Extracted fields: %s", extracted)

        return jsonify({
            "success":   True,
            "data":      extracted
        }), 200

    except Exception as e:
        logger.exception("OCR processing failed")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=False)
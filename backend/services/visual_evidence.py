"""
MetroNetra — Visual Evidence Overlay Service
SIH Problem Statement ID: SIH26034

Generates composite visual evidence overlays on product images:
- Draws PaddleOCR text bounding boxes
- Highlights relevant statutory declarations
- Color codes boxes:
    🟢 Green  = PASS
    🟡 Yellow = WARNING / NEEDS_VERIFICATION
    🔴 Red    = POTENTIAL_NON_COMPLIANCE
- Annotates detected text, confidence %, readability status, and font size in mm
- Saves overlay images to uploads/ for web UI serving
"""
from __future__ import annotations
import os
import cv2
import numpy as np
from typing import List, Optional, Dict, Any

from backend.services.ocr import OCRToken
from backend.services.declaration_extractor import ExtractedDeclaration
from backend.services.visual_compliance import VisualComplianceReport


def generate_annotated_evidence_image(
    original_image: np.ndarray,
    tokens: List[OCRToken],
    declarations: List[ExtractedDeclaration],
    visual_compliance: Optional[VisualComplianceReport] = None,
    inspection_id: str = "demo",
    save_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Renders an annotated composite evidence image with bounding boxes, text overlays,
    confidence metrics, and status badges.
    Saves image to disk and returns relative web URL path (e.g. /uploads/...)
    """
    img_bgr = cv2.cvtColor(original_image.copy(), cv2.COLOR_RGB2BGR)
    h_img, w_img = img_bgr.shape[:2]

    # Map field name -> readability status
    readability_map: Dict[str, str] = {}
    font_map: Dict[str, str] = {}
    if visual_compliance:
        for r in visual_compliance.text_readability:
            readability_map[r.declaration_field] = r.readability_status
        for f in visual_compliance.font_dimensions:
            if f.estimated_height_mm:
                font_map[f.declaration_field] = f"{f.estimated_height_mm}mm"

    # Map field name -> extracted declaration
    decl_map: Dict[str, ExtractedDeclaration] = {d.field: d for d in declarations if d.extracted_value}

    # 1. Draw light gray bounding boxes for all raw PaddleOCR tokens first
    for tok in tokens:
        if tok.bbox:
            ymin, xmin, ymax, xmax = tok.bbox
            cv2.rectangle(img_bgr, (xmin, ymin), (xmax, ymax), (200, 200, 200), 1)

    # 2. Draw prominent color-coded bounding boxes for extracted statutory fields
    for decl in declarations:
        if not decl.extracted_value or decl.source_ocr_token_index is None:
            continue
        if decl.source_ocr_token_index < 0 or decl.source_ocr_token_index >= len(tokens):
            continue

        matched_tok = tokens[decl.source_ocr_token_index]
        if not matched_tok.bbox:
            continue

        ymin, xmin, ymax, xmax = matched_tok.bbox
        status = readability_map.get(decl.field, "NEEDS_VERIFICATION")

        # Color coding
        if status == "PASS":
            color_bgr = (40, 200, 80)     # Green
        elif status == "WARNING":
            color_bgr = (0, 215, 255)    # Yellow / Gold
        else:
            color_bgr = (50, 100, 240)    # Red / Crimson

        # Thick highlight box
        cv2.rectangle(img_bgr, (xmin, ymin), (xmax, ymax), color_bgr, 3)

        # Label badge background & text
        conf_pct = int(matched_tok.confidence * 100)
        font_str = f" | {font_map[decl.field]}" if decl.field in font_map else ""
        badge_text = f"{decl.field}: '{matched_tok.text[:20]}' ({conf_pct}%{font_str})"

        # Calculate text badge size
        (text_w, text_h), baseline = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        badge_ymin = max(0, ymin - text_h - 8)
        badge_ymax = max(text_h + 8, ymin)
        badge_xmax = min(w_img, xmin + text_w + 10)

        # Draw filled background rectangle for label legibility
        cv2.rectangle(img_bgr, (xmin, badge_ymin), (badge_xmax, badge_ymax), color_bgr, -1)
        # White text
        cv2.putText(
            img_bgr, badge_text,
            (xmin + 5, badge_ymax - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
        )

    # Save output image
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{inspection_id}_annotated_full_label.jpg"
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, img_bgr)
        return f"/uploads/{filename}"

    return None

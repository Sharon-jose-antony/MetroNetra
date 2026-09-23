"""
MetroNetra — Font Size and Readability Analysis Service
Module: backend.services.font_analysis.service

Performs visual text height estimation and readability assessment on statutory
declarations using existing OCR bounding boxes and lightweight OpenCV image processing.

Key Principles:
1. Reuses existing OCR tokens, bounding boxes, and image arrays (no redundant OCR passes).
2. Physical calibration honesty:
   - When no physical scale/calibration reference exists, physical size is strictly reported
     as "Not calibrated", and visual text height is reported in pixels (e.g. 32 px).
3. Non-punitive evaluation:
   - Poor readability or small visual height does NOT automatically trigger a legal violation.
   - Outputs: PASS, REVIEW, POTENTIAL_NON_COMPLIANCE (with uncertain cases directed to manual review).
4. Readability categories:
   - READABLE, LOW READABILITY, REVIEW.
"""
from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

from backend.services.ocr import OCRToken
from backend.services.declaration_extractor import ExtractedDeclaration


@dataclass
class DeclarationFontAnalysis:
    """Detailed font size and readability analysis for a single statutory declaration."""
    field: str
    text_height_px: Optional[int]
    physical_size: str                   # "Not calibrated" or "X.X mm (calibrated)"
    readability: str                     # "READABLE" | "LOW READABILITY" | "REVIEW"
    confidence: float                    # 0.0 to 1.0 composite confidence
    status: str                          # "PASS" | "REVIEW" | "POTENTIAL_NON_COMPLIANCE"
    ocr_confidence: float                # 0.0 to 1.0
    contrast_score: float                # standard deviation of grayscale crop
    sharpness_score: float               # Laplacian variance of grayscale crop
    bbox: Optional[List[int]] = None     # [ymin, xmin, ymax, xmax]
    notes: Optional[str] = None


@dataclass
class FontReadabilityReport:
    """Aggregated font and readability report for an inspection."""
    items: List[DeclarationFontAnalysis] = field(default_factory=list)
    calibration_status: str = "Not calibrated"
    summary_message: str = ""
    is_calibrated: bool = False
    pixels_per_mm: Optional[float] = None


def _find_matching_token(
    decl: ExtractedDeclaration,
    tokens: List[OCRToken],
) -> Optional[OCRToken]:
    """Finds the most accurate OCRToken for an extracted declaration."""
    # 1. Direct index match if available
    if (
        decl.source_ocr_token_index is not None
        and 0 <= decl.source_ocr_token_index < len(tokens)
    ):
        cand = tokens[decl.source_ocr_token_index]
        if cand.bbox:
            return cand

    # 2. Text overlap match with raw OCR snippet or extracted value
    decl_raw = (decl.raw_ocr_text or decl.extracted_value or "").lower().strip()
    if not decl_raw or not tokens:
        return None

    raw_words = set(decl_raw.split())
    best_token: Optional[OCRToken] = None
    best_score = 0.0

    for tok in tokens:
        if not tok.bbox:
            continue
        tok_text = tok.text.lower().strip()
        if not tok_text:
            continue
        tok_words = set(tok_text.split())
        inter = len(raw_words & tok_words)
        union = len(raw_words | tok_words)
        score = (inter / union) if union > 0 else 0.0

        # Substring bonus
        if tok_text in decl_raw or decl_raw in tok_text:
            score = max(score, 0.6)

        if score > best_score:
            best_score = score
            best_token = tok

    if best_score >= 0.2:
        return best_token

    return None


def analyze_declaration_font(
    decl: ExtractedDeclaration,
    tokens: List[OCRToken],
    gray_image: np.ndarray,
    h_img: int,
    w_img: int,
    pixels_per_mm: Optional[float] = None,
) -> DeclarationFontAnalysis:
    """Analyzes a single declaration for visual font height and readability."""
    if not decl.extracted_value:
        return DeclarationFontAnalysis(
            field=decl.field,
            text_height_px=None,
            physical_size="Not calibrated",
            readability="REVIEW",
            confidence=0.0,
            status="REVIEW",
            ocr_confidence=0.0,
            contrast_score=0.0,
            sharpness_score=0.0,
            bbox=None,
            notes="Declaration not detected in label; visual analysis unavailable.",
        )

    matched_token = _find_matching_token(decl, tokens)
    if not matched_token or not matched_token.bbox:
        return DeclarationFontAnalysis(
            field=decl.field,
            text_height_px=None,
            physical_size="Not calibrated",
            readability="REVIEW",
            confidence=round(float(decl.extraction_confidence or 0.3), 2),
            status="REVIEW",
            ocr_confidence=round(float(decl.extraction_confidence or 0.0), 2),
            contrast_score=0.0,
            sharpness_score=0.0,
            bbox=None,
            notes="Text bounding box could not be isolated for region analysis.",
        )

    ymin, xmin, ymax, xmax = matched_token.bbox
    box_h = max(1, ymax - ymin)
    box_w = max(1, xmax - xmin)

    # Estimate text line count for multi-line boxes
    tok_text = matched_token.text.strip()
    num_lines = max(1, tok_text.count("\n") + 1)
    # Estimate visual font character height (pixels per line)
    est_line_h = box_h / num_lines
    est_height_px = max(1, int(round(est_line_h)))

    # Physical size handling
    if pixels_per_mm and pixels_per_mm > 0:
        est_mm = round(est_height_px / pixels_per_mm, 2)
        physical_size_str = f"{est_mm} mm (calibrated)"
    else:
        physical_size_str = "Not calibrated"

    # Crop text region with slight padding for local contrast and sharpness analysis
    pad_y = max(1, int(box_h * 0.08))
    pad_x = max(1, int(box_w * 0.08))
    crop_ymin = max(0, ymin - pad_y)
    crop_xmin = max(0, xmin - pad_x)
    crop_ymax = min(h_img, ymax + pad_y)
    crop_xmax = min(w_img, xmax + pad_x)

    crop = gray_image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

    if crop.size > 0:
        sharpness = float(cv2.Laplacian(crop, cv2.CV_64F).var())
        contrast = float(crop.std())
    else:
        sharpness = 0.0
        contrast = 0.0

    ocr_conf = float(matched_token.confidence)

    # Composite confidence score (0.0 - 1.0)
    # Factors: OCR confidence (40%), Contrast adequacy (25%), Sharpness (20%), Resolution (15%)
    norm_contrast = min(1.0, contrast / 35.0)
    norm_sharpness = min(1.0, sharpness / 80.0)
    norm_resolution = min(1.0, est_height_px / 20.0)

    composite_conf = (
        (0.40 * ocr_conf) +
        (0.25 * norm_contrast) +
        (0.20 * norm_sharpness) +
        (0.15 * norm_resolution)
    )
    composite_conf = round(float(np.clip(composite_conf, 0.05, 0.99)), 2)

    # Readability classification: READABLE | LOW READABILITY | REVIEW
    # High clarity indicators:
    is_sharp = sharpness >= 40.0
    is_contrasted = contrast >= 18.0
    is_adequate_res = est_height_px >= 14
    is_high_ocr = ocr_conf >= 0.70

    if is_high_ocr and is_contrasted and is_sharp and is_adequate_res:
        readability = "READABLE"
        status = "PASS"
        notes = (
            f"Clear legibility: estimated height {est_height_px}px, contrast std={contrast:.1f}, "
            f"sharpness={sharpness:.1f}, OCR confidence={int(ocr_conf*100)}%."
        )
    elif (ocr_conf >= 0.45 and (is_contrasted or is_sharp) and est_height_px >= 10):
        readability = "READABLE"
        status = "PASS"
        notes = (
            f"Readable: estimated height {est_height_px}px, contrast std={contrast:.1f}, "
            f"sharpness={sharpness:.1f}."
        )
    elif ocr_conf >= 0.30 and est_height_px >= 8:
        readability = "LOW READABILITY"
        status = "REVIEW"
        notes = (
            f"Low readability: text height {est_height_px}px with partial contrast (std={contrast:.1f}) "
            f"or blur ({sharpness:.1f}). Manual inspection recommended."
        )
    else:
        readability = "REVIEW"
        status = "REVIEW"
        notes = (
            f"Sub-optimal visibility: OCR confidence {int(ocr_conf*100)}%, height {est_height_px}px. "
            "Requires manual verification."
        )

    return DeclarationFontAnalysis(
        field=decl.field,
        text_height_px=est_height_px,
        physical_size=physical_size_str,
        readability=readability,
        confidence=composite_conf,
        status=status,
        ocr_confidence=round(ocr_conf, 4),
        contrast_score=round(contrast, 2),
        sharpness_score=round(sharpness, 2),
        bbox=[ymin, xmin, ymax, xmax],
        notes=notes,
    )


def analyze_font_and_readability(
    image_np: np.ndarray,
    tokens: List[OCRToken],
    declarations: List[ExtractedDeclaration],
    pixels_per_mm: Optional[float] = None,
) -> FontReadabilityReport:
    """
    Main entry point for Font Size & Readability Analysis.
    Evaluates all extracted statutory declarations against existing OCR bounding boxes.
    """
    if image_np is None or len(declarations) == 0:
        return FontReadabilityReport(
            items=[],
            calibration_status="Not calibrated",
            summary_message="No statutory declarations or image available for font analysis.",
            is_calibrated=False,
            pixels_per_mm=None,
        )

    h_img, w_img = image_np.shape[:2]
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image_np

    is_calibrated = (pixels_per_mm is not None and pixels_per_mm > 0)
    calibration_str = f"Calibrated ({pixels_per_mm:.1f} px/mm)" if is_calibrated else "Not calibrated"

    items: List[DeclarationFontAnalysis] = []
    for decl in declarations:
        item = analyze_declaration_font(
            decl=decl,
            tokens=tokens,
            gray_image=gray_image,
            h_img=h_img,
            w_img=w_img,
            pixels_per_mm=pixels_per_mm,
        )
        items.append(item)

    readable_count = sum(1 for it in items if it.readability == "READABLE")
    total_applicable = sum(1 for it in items if it.text_height_px is not None)
    total_items = len(items)

    summary = (
        f"Font & Readability: {readable_count}/{total_applicable or total_items} readable declarations. "
        f"Scale: {calibration_str}."
    )

    return FontReadabilityReport(
        items=items,
        calibration_status=calibration_str,
        summary_message=summary,
        is_calibrated=is_calibrated,
        pixels_per_mm=pixels_per_mm,
    )

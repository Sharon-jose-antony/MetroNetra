"""
MetroNetra — Visual Compliance Analysis Service
SIH Problem Statement ID: SIH26034

Implements 3 clearly separated visual compliance checks:
1. Image Quality Assessment (wraps existing image quality engine)
2. Text Readability Assessment (evaluates PaddleOCR tokens per declaration)
3. Font-Size / Character-Dimension Check (Calibrated vs Uncalibrated modes under PCR 2011 Rule 9)
"""
from __future__ import annotations
import math
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

from backend.services.ocr import OCRToken
from backend.services.declaration_extractor import ExtractedDeclaration
from backend.services.image_quality import ImageQualityResult
from backend.config import settings


@dataclass
class TextReadabilityItem:
    declaration_field: str
    extracted_text: Optional[str]
    ocr_confidence: float
    local_contrast_score: float         # Standard deviation of grayscale pixels in text crop
    local_blur_score: float             # Laplacian variance in text crop
    bbox_resolution_px: Tuple[int, int] # (height, width) of bounding box in pixels
    is_edge_clipped: bool               # True if bounding box touches image boundaries
    readability_status: str             # PASS | WARNING | NEEDS_VERIFICATION
    rationale: str
    bbox: Optional[List[int]] = None    # [ymin, xmin, ymax, xmax]


@dataclass
class FontDimensionAnalysis:
    declaration_field: str
    calibration_mode: str               # CALIBRATED | UNCALIBRATED
    estimated_height_mm: Optional[float]
    estimated_width_mm: Optional[float]
    width_height_ratio: Optional[float]
    required_min_height_mm: Optional[float]
    rule_reference: str
    status: str                         # PASS | WARNING | NEEDS_VERIFICATION
    message: str
    bbox: Optional[List[int]] = None


@dataclass
class VisualComplianceReport:
    image_quality: ImageQualityResult
    text_readability: List[TextReadabilityItem]
    font_dimensions: List[FontDimensionAnalysis]
    calibration_mode: str               # CALIBRATED | UNCALIBRATED
    pixels_per_mm: Optional[float]
    summary_message: str


# ── Legal Metrology Rule 9 Minimum Height Tables ─────────────────────────────
# PCR 2011 Rule 9 specifies minimum character heights based on Net Quantity
RULE9_HEIGHT_TABLE = [
    # (max_qty_grams_or_ml, min_height_mm)
    (50.0, 1.0),
    (100.0, 1.5),
    (500.0, 2.0),
    (float("inf"), 4.0),
]


def _get_required_font_height_mm(net_quantity_str: Optional[str] = None) -> float:
    """
    Look up minimum font size in mm according to Legal Metrology Rule 9.
    Defaults to 1.5mm if quantity is missing/unparsed.
    """
    if not net_quantity_str:
        return 1.5

    import re
    match = re.search(r'(\d+(?:\.\d+)?)\s*(k?g|m?l|l|kg)?', net_quantity_str.lower())
    if not match:
        return 1.5

    val = float(match.group(1))
    unit = match.group(2) or "g"

    if unit in ("kg", "l"):
        val *= 1000.0

    for max_qty, min_h in RULE9_HEIGHT_TABLE:
        if val <= max_qty:
            return min_h
    return 2.0


def assess_text_readability(
    image_np: np.ndarray,
    tokens: List[OCRToken],
    declarations: List[ExtractedDeclaration],
) -> List[TextReadabilityItem]:
    """
    Evaluates text readability/legibility for each important statutory declaration:
    - PaddleOCR confidence score
    - Local text/background contrast
    - Blur/sharpness around the text bounding box (Laplacian variance)
    - Text-region resolution (pixel height/width)
    - Edge proximity / clipping check
    Returns status: PASS / WARNING / NEEDS_VERIFICATION.
    Rule: Never flags non-compliance based on low OCR score alone.
    """
    h_img, w_img = image_np.shape[:2]
    gray_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if len(image_np.shape) == 3 else image_np

    readability_results: List[TextReadabilityItem] = []

    for decl in declarations:
        if not decl.extracted_value:
            readability_results.append(TextReadabilityItem(
                declaration_field=decl.field,
                extracted_text=None,
                ocr_confidence=0.0,
                local_contrast_score=0.0,
                local_blur_score=0.0,
                bbox_resolution_px=(0, 0),
                is_edge_clipped=False,
                readability_status="NEEDS_VERIFICATION",
                rationale="Declaration text was not detected in OCR tokens.",
                bbox=None,
            ))
            continue

        # Find best matching OCR token by text overlap
        matched_token: Optional[OCRToken] = None
        if decl.source_ocr_token_index is not None and 0 <= decl.source_ocr_token_index < len(tokens):
            matched_token = tokens[decl.source_ocr_token_index]
        else:
            decl_lower = decl.raw_ocr_text.lower().strip() if decl.raw_ocr_text else ""
            best_overlap = 0.0
            for t in tokens:
                t_lower = t.text.lower().strip()
                if not decl_lower or not t_lower:
                    continue
                words_decl = set(decl_lower.split())
                words_tok = set(t_lower.split())
                overlap = len(words_decl & words_tok) / len(words_decl | words_tok)
                if overlap > best_overlap:
                    best_overlap = overlap
                    matched_token = t

        if not matched_token or not matched_token.bbox:
            readability_results.append(TextReadabilityItem(
                declaration_field=decl.field,
                extracted_text=decl.extracted_value,
                ocr_confidence=decl.extraction_confidence,
                local_contrast_score=0.0,
                local_blur_score=0.0,
                bbox_resolution_px=(0, 0),
                is_edge_clipped=False,
                readability_status="NEEDS_VERIFICATION",
                rationale="No explicit bounding box found for declaration text.",
                bbox=None,
            ))
            continue

        bbox = matched_token.bbox  # [ymin, xmin, ymax, xmax]
        ymin, xmin, ymax, xmax = bbox
        box_h = max(1, ymax - ymin)
        box_w = max(1, xmax - xmin)

        # Check edge clipping (touching image edge within 5px margin)
        is_edge_clipped = (ymin <= 5 or xmin <= 5 or ymax >= h_img - 5 or xmax >= w_img - 5)

        # Extract text crop region
        pad_y = max(2, int(box_h * 0.1))
        pad_x = max(2, int(box_w * 0.1))
        crop_ymin = max(0, ymin - pad_y)
        crop_xmin = max(0, xmin - pad_x)
        crop_ymax = min(h_img, ymax + pad_y)
        crop_xmax = min(w_img, xmax + pad_x)

        crop_gray = gray_img[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

        # Calculate local blur (Laplacian variance) and local contrast (standard deviation)
        if crop_gray.size > 0:
            local_blur = float(cv2.Laplacian(crop_gray, cv2.CV_64F).var())
            local_contrast = float(crop_gray.std())
        else:
            local_blur = 0.0
            local_contrast = 0.0

        ocr_conf = matched_token.confidence

        # Determine readability status
        issues = []
        if ocr_conf < 0.50:
            issues.append(f"Low OCR confidence ({ocr_conf*100:.1f}%)")
        if local_contrast < 18.0:
            issues.append(f"Low local contrast (std={local_contrast:.1f})")
        if local_blur < 45.0:
            issues.append(f"Text region blur (var={local_blur:.1f})")
        if box_h < 12:
            issues.append(f"Very small pixel height ({box_h}px)")
        if is_edge_clipped:
            issues.append("Text box near image border (possible clipping)")

        if not issues and ocr_conf >= 0.70:
            status = "PASS"
            rationale = f"High legibility: clear contrast (std={local_contrast:.1f}), sharp focus (var={local_blur:.1f}), confidence {ocr_conf*100:.1f}%."
        elif ocr_conf >= 0.40 and len(issues) <= 2:
            status = "WARNING"
            rationale = "Marginal readability: " + "; ".join(issues) + ". Manual check recommended."
        else:
            status = "NEEDS_VERIFICATION"
            rationale = "Unreliable readability: " + "; ".join(issues) + ". Requires officer verification."

        readability_results.append(TextReadabilityItem(
            declaration_field=decl.field,
            extracted_text=decl.extracted_value,
            ocr_confidence=round(ocr_conf, 4),
            local_contrast_score=round(local_contrast, 2),
            local_blur_score=round(local_blur, 2),
            bbox_resolution_px=(box_h, box_w),
            is_edge_clipped=is_edge_clipped,
            readability_status=status,
            rationale=rationale,
            bbox=bbox,
        ))

    return readability_results


def assess_font_dimensions(
    tokens: List[OCRToken],
    declarations: List[ExtractedDeclaration],
    calibration_mode: str = "UNCALIBRATED",
    known_reference_mm: Optional[float] = None,
    known_reference_px: Optional[float] = None,
    net_quantity_str: Optional[str] = None,
) -> List[FontDimensionAnalysis]:
    """
    Performs Font-Size and Character-Dimension Analysis in two modes:
    A. CALIBRATED MODE: Uses physical scale (known_reference_mm / known_reference_px)
       to compute px_per_mm and estimate character height/width in mm.
       Compares against PCR 2011 Rule 9 requirements.
    B. UNCALIBRATED MODE: Refuses to make fake millimetre claims.
       Displays: "Physical font size cannot be reliably verified from this image without scale/calibration."
       Marks status as NEEDS_VERIFICATION.
    """
    mode = calibration_mode.upper().strip()
    font_analyses: List[FontDimensionAnalysis] = []

    # Uncalibrated mode
    if mode != "CALIBRATED" or not known_reference_mm or not known_reference_px or known_reference_px <= 0:
        for decl in declarations:
            font_analyses.append(FontDimensionAnalysis(
                declaration_field=decl.field,
                calibration_mode="UNCALIBRATED",
                estimated_height_mm=None,
                estimated_width_mm=None,
                width_height_ratio=None,
                required_min_height_mm=_get_required_font_height_mm(net_quantity_str),
                rule_reference="Rule 9, Legal Metrology (Packaged Commodities) Rules, 2011",
                status="NEEDS_VERIFICATION",
                message="Physical font size cannot be reliably verified from this image without scale/calibration.",
                bbox=None,
            ))
        return font_analyses

    # Calibrated mode
    px_per_mm = float(known_reference_px) / float(known_reference_mm)
    required_min_h = _get_required_font_height_mm(net_quantity_str)

    for decl in declarations:
        matched_token: Optional[OCRToken] = None
        decl_lower = decl.raw_ocr_text.lower().strip() if decl.raw_ocr_text else ""
        best_score = 0.0

        for t in tokens:
            t_lower = t.text.lower().strip()
            if not decl_lower or not t_lower:
                continue
            words_decl = set(decl_lower.split())
            words_tok = set(t_lower.split())
            score = len(words_decl & words_tok) / len(words_decl | words_tok)
            if score > best_score:
                best_score = score
                matched_token = t

        if not matched_token or not matched_token.bbox:
            font_analyses.append(FontDimensionAnalysis(
                declaration_field=decl.field,
                calibration_mode="CALIBRATED",
                estimated_height_mm=None,
                estimated_width_mm=None,
                width_height_ratio=None,
                required_min_height_mm=required_min_h,
                rule_reference="Rule 9, Legal Metrology (Packaged Commodities) Rules, 2011",
                status="NEEDS_VERIFICATION",
                message="Declaration text region not located; font dimension unverified.",
                bbox=None,
            ))
            continue

        bbox = matched_token.bbox  # [ymin, xmin, ymax, xmax]
        ymin, xmin, ymax, xmax = bbox
        box_h_px = max(1, ymax - ymin)
        box_w_px = max(1, xmax - xmin)

        text_str = matched_token.text.strip()
        num_lines = max(1, text_str.count("\n") + 1)
        line_h_px = box_h_px / num_lines

        # Cap-height factor (~0.80 of line box height)
        est_char_h_mm = round((line_h_px * 0.80) / px_per_mm, 2)

        char_count = max(1, len(text_str.replace(" ", "")))
        est_char_w_mm = round((box_w_px / char_count) / px_per_mm, 2)
        aspect_ratio = round(est_char_w_mm / max(0.1, est_char_h_mm), 2)

        rule_ref = "Rule 9, Legal Metrology (Packaged Commodities) Rules, 2011"
        if est_char_h_mm >= required_min_h:
            status = "PASS"
            msg = (
                f"Calibrated estimated font height: {est_char_h_mm} mm "
                f"meets/exceeds statutory minimum requirement of {required_min_h} mm "
                f"(ratio w/h={aspect_ratio})."
            )
        elif est_char_h_mm >= required_min_h * 0.80:
            status = "WARNING"
            msg = (
                f"Calibrated estimated font height: {est_char_h_mm} mm "
                f"is borderline close to statutory minimum requirement of {required_min_h} mm."
            )
        else:
            status = "NEEDS_VERIFICATION"
            msg = (
                f"Calibrated estimated font height: {est_char_h_mm} mm "
                f"appears below statutory minimum requirement of {required_min_h} mm under {rule_ref}. "
                "Manual physical measurement with caliper recommended."
            )

        font_analyses.append(FontDimensionAnalysis(
            declaration_field=decl.field,
            calibration_mode="CALIBRATED",
            estimated_height_mm=est_char_h_mm,
            estimated_width_mm=est_char_w_mm,
            width_height_ratio=aspect_ratio,
            required_min_height_mm=required_min_h,
            rule_reference=rule_ref,
            status=status,
            message=msg,
            bbox=bbox,
        ))

    return font_analyses


def evaluate_visual_compliance(
    image_np: np.ndarray,
    quality_result: ImageQualityResult,
    tokens: List[OCRToken],
    declarations: List[ExtractedDeclaration],
    calibration_mode: str = "UNCALIBRATED",
    known_reference_mm: Optional[float] = None,
    known_reference_px: Optional[float] = None,
    net_quantity_str: Optional[str] = None,
) -> VisualComplianceReport:
    """
    Master function orchestrating Image Quality, Text Readability, and Font Dimensions.
    """
    # 1. Text Readability Analysis
    readability = assess_text_readability(image_np, tokens, declarations)

    # 2. Font Dimensions Analysis
    font_dims = assess_font_dimensions(
        tokens=tokens,
        declarations=declarations,
        calibration_mode=calibration_mode,
        known_reference_mm=known_reference_mm,
        known_reference_px=known_reference_px,
        net_quantity_str=net_quantity_str,
    )

    # Pixels per mm if calibrated
    px_per_mm = (
        round(known_reference_px / known_reference_mm, 2)
        if calibration_mode.upper() == "CALIBRATED" and known_reference_mm and known_reference_px
        else None
    )

    summary = (
        f"Visual Compliance: Quality={quality_result.quality_recommendation}, "
        f"Readability={sum(1 for r in readability if r.readability_status=='PASS')}/{len(readability)} PASS, "
        f"Font Scale={'Calibrated (' + str(px_per_mm) + ' px/mm)' if px_per_mm else 'Uncalibrated'}."
    )

    return VisualComplianceReport(
        image_quality=quality_result,
        text_readability=readability,
        font_dimensions=font_dims,
        calibration_mode=calibration_mode.upper(),
        pixels_per_mm=px_per_mm,
        summary_message=summary,
    )

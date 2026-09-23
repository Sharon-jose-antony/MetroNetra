"""
MetroNetra — Inspection Pipeline Orchestrator
Chains all services in sequence:
  Image Quality → Preprocessing → Multi-Pass OCR → Declaration Extraction
  → Rule Engine → Evidence → Confidence → Assessment

This is the single entry point called by the /analyze API endpoint.
"""
from __future__ import annotations
import os
import time
import uuid
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import Optional, List

from backend.services.image_quality import assess_quality, ImageQualityResult
from backend.services.preprocessing import (
    preprocess, PreprocessResult
)
from backend.services.ocr import (
    OCRProvider, create_ocr_provider, run_multi_pass_ocr, MultiPassOCRResult,
    OCRToken, _deduplicate_tokens
)
from backend.services.declaration_extractor import extract_all_declarations, ExtractedDeclaration
from backend.services.rule_engine import evaluate_rules, compute_overall_status, RuleEvalResult
from backend.services.confidence import compute_confidence, ConfidenceResult
from backend.services.evidence import generate_evidence, EvidenceSlice
from backend.services.font_analysis import (
    analyze_font_and_readability, FontReadabilityReport
)
from backend.services.visual_elements import (
    detect_visual_elements, VisualElementsReport
)
from backend.database.models import ProductCategory, InspectionStatus, DeclarationStatus
from backend.config import settings


@dataclass
class PipelineResult:
    inspection_id: str
    status: InspectionStatus
    overall_assessment: str            # PASS | MANUAL_REVIEW | POTENTIAL_NON_COMPLIANCE
    quality_result: Optional[ImageQualityResult]
    ocr_result: Optional[MultiPassOCRResult]
    declarations: List[ExtractedDeclaration]
    rule_results: List[RuleEvalResult]
    evidence: List[EvidenceSlice]
    confidence: Optional[ConfidenceResult]
    font_analysis: Optional[FontReadabilityReport] = None
    visual_elements: Optional[VisualElementsReport] = None
    processing_time_seconds: float = 0.0
    pipeline_stages: List[str] = field(default_factory=list)
    timing_breakdown: Optional[dict] = None
    error: Optional[str] = None


# Singleton OCR provider (initialized once per process and reused)
_ocr_provider: Optional[OCRProvider] = None


def get_ocr_provider() -> OCRProvider:
    global _ocr_provider
    if _ocr_provider is None:
        _ocr_provider = create_ocr_provider()
    return _ocr_provider


def _rescale_tokens(tokens: List[OCRToken], scale_factor: float) -> List[OCRToken]:
    """Rescale token bboxes and polygons back to original image coordinates."""
    if abs(scale_factor - 1.0) <= 0.02:
        return tokens
    for tok in tokens:
        tok.upscaled_bbox = list(tok.bbox)
        ymin = int(round(tok.bbox[0] / scale_factor))
        xmin = int(round(tok.bbox[1] / scale_factor))
        ymax = int(round(tok.bbox[2] / scale_factor))
        xmax = int(round(tok.bbox[3] / scale_factor))
        tok.bbox = [ymin, xmin, ymax, xmax]
        if tok.polygon:
            tok.polygon = [
                [int(round(p[0] / scale_factor)), int(round(p[1] / scale_factor))]
                for p in tok.polygon
            ]
    return tokens


def run_inspection_pipeline(
    image_path: str,
    product_category: ProductCategory,
    inspection_id: str,
    is_imported: bool = False,
    evidence_save_dir: Optional[str] = None,
) -> PipelineResult:
    """
    Execute the high-speed AI compliance inspection pipeline on a single image:
    1. Fast Image Read & Decode
    2. Lightweight Image Quality Assessment (Laplacian + Glare)
    3. Fast Primary Preprocessing (<10ms resolution normalization to 1280px)
    4. Primary Deep OCR Pass (PaddleOCR singleton)
    5. Context-Aware Declaration Extraction on Pass 1
    6. Conditional Fallback Pass (ONLY if critical statutory declarations missing or low confidence)
    7. Statutory Rule Engine Evaluation (PCR 2011)
    8. Evidence Extraction & Confidence Scoring
    """
    t_start = time.perf_counter()
    stages: list[str] = []
    error: Optional[str] = None
    t_read = 0.0
    t_qual = 0.0
    t_prep = 0.0
    t_ocr1 = 0.0
    t_ocr2 = 0.0
    t_decl = 0.0
    t_rules = 0.0
    t_ev = 0.0
    passes_used: list[str] = []

    # ── 1. Load Image ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        pil_image = Image.open(image_path).convert("RGB")
        img_np = np.array(pil_image)
        t_read = time.perf_counter() - t0
        stages.append("IMAGE_LOADED")
    except Exception as e:
        return PipelineResult(
            inspection_id=inspection_id,
            status=InspectionStatus.ERROR,
            overall_assessment="ERROR",
            quality_result=None,
            ocr_result=None,
            declarations=[],
            rule_results=[],
            evidence=[],
            confidence=None,
            processing_time_seconds=time.perf_counter() - t_start,
            pipeline_stages=["IMAGE_LOAD_FAILED"],
            error=f"Failed to load image: {str(e)}",
        )

    # ── 2. Image Quality Assessment ───────────────────────────────────────────
    t0 = time.perf_counter()
    quality_result = assess_quality(img_np)
    t_qual = time.perf_counter() - t0
    stages.append("IMAGE_QUALITY_ASSESSED")

    # If image is UNUSABLE, skip OCR entirely
    if quality_result.quality_recommendation == "UNUSABLE":
        stages.append("OCR_SKIPPED_UNUSABLE_IMAGE")
        return PipelineResult(
            inspection_id=inspection_id,
            status=InspectionStatus.MANUAL_REVIEW,
            overall_assessment="MANUAL_REVIEW",
            quality_result=quality_result,
            ocr_result=None,
            declarations=[],
            rule_results=[],
            evidence=[],
            confidence=None,
            processing_time_seconds=round(time.perf_counter() - t_start, 3),
            pipeline_stages=stages,
            error="Image quality is insufficient for automated analysis. Manual inspection required.",
        )

    # ── 3. Fast Primary Preprocessing ─────────────────────────────────────────
    t0 = time.perf_counter()
    preprocessed = preprocess(img_np)
    t_prep = time.perf_counter() - t0
    stages.append("IMAGE_PREPROCESSED")

    # ── 4. Primary OCR Pass ───────────────────────────────────────────────────
    provider = get_ocr_provider()
    tokens_pass1: List[OCRToken] = []
    ocr_result: Optional[MultiPassOCRResult] = None

    t0 = time.perf_counter()
    try:
        tokens_pass1 = provider.extract_text(preprocessed.upscaled, pass_type="original")
        tokens_pass1 = _rescale_tokens(tokens_pass1, preprocessed.scale_factor)
        passes_used.append("original")
        stages.append("OCR_PASS_1_COMPLETE")
    except Exception as e:
        stages.append("OCR_PASS_1_FAILED")
        error = f"Primary OCR failed: {str(e)}"
    t_ocr1 = time.perf_counter() - t0

    # ── 5. Fast Declaration Extraction on Pass 1 ──────────────────────────────
    t0 = time.perf_counter()
    full_text_pass1 = "\n".join(t.text for t in tokens_pass1 if t.text.strip())
    declarations = extract_all_declarations(tokens_pass1, full_text_pass1)
    t_decl = time.perf_counter() - t0
    stages.append("DECLARATIONS_EXTRACTED")

    # ── 6. Conditional Fallback OCR Pass ──────────────────────────────────────
    # Only perform fallback OCR if critical declarations (MRP, Net Qty) are missing
    # or overall token confidence is low
    has_mrp = any(d.field == "MRP" and d.extracted_value for d in declarations)
    has_net_qty = any(d.field == "NET_QUANTITY" and d.extracted_value for d in declarations)
    found_count = sum(1 for d in declarations if d.extracted_value)
    mean_conf = (sum(t.confidence for t in tokens_pass1) / len(tokens_pass1)) if tokens_pass1 else 0.0

    skip_fallback = (has_mrp and has_net_qty and found_count >= 3 and mean_conf >= 0.40) or (found_count >= 5 and mean_conf >= 0.50)

    final_tokens = tokens_pass1

    if not skip_fallback and quality_result.quality_score >= 0.35:
        t0 = time.perf_counter()
        try:
            clahe_variant = preprocessed.enhanced_clahe
            tokens_pass2 = provider.extract_text(clahe_variant, pass_type="clahe")
            tokens_pass2 = _rescale_tokens(tokens_pass2, preprocessed.scale_factor)
            passes_used.append("clahe")

            # Deduplicate and fuse tokens from both passes
            final_tokens = _deduplicate_tokens(tokens_pass1 + tokens_pass2)
            full_text_fused = "\n".join(t.text for t in final_tokens if t.text.strip())

            # Re-extract declarations with enriched fused text
            t_re_decl = time.perf_counter()
            declarations = extract_all_declarations(final_tokens, full_text_fused)
            t_decl += (time.perf_counter() - t_re_decl)

            stages.append("OCR_FALLBACK_APPLIED")
        except Exception as e:
            stages.append("OCR_FALLBACK_FAILED")
        t_ocr2 = time.perf_counter() - t0
    else:
        stages.append("OCR_FALLBACK_SKIPPED_OPTIMIZED")

    final_full_text = "\n".join(t.text for t in final_tokens if t.text.strip())
    final_mean_conf = (
        round(sum(t.confidence for t in final_tokens) / len(final_tokens), 4)
        if final_tokens else 0.0
    )

    ocr_result = MultiPassOCRResult(
        tokens=final_tokens,
        full_text=final_full_text,
        mean_confidence=final_mean_conf,
        passes_used=passes_used,
        all_raw_candidates=final_tokens,
    )
    stages.append("OCR_COMPLETE")

    # ── 6.5. Font Size & Readability Analysis ──────────────────────────────────
    font_report: Optional[FontReadabilityReport] = None
    t_font = 0.0
    if settings.enable_font_analysis:
        t0 = time.perf_counter()
        try:
            font_report = analyze_font_and_readability(
                image_np=preprocessed.original,
                tokens=final_tokens,
                declarations=declarations,
            )
            stages.append("FONT_ANALYSIS_COMPLETE")
        except Exception as e:
            stages.append("FONT_ANALYSIS_FAILED")
        t_font = time.perf_counter() - t0

    # ── 6.6. Visual Element Detection (QR & Barcodes) ──────────────────────────
    visual_report: Optional[VisualElementsReport] = None
    t_visual = 0.0
    t_qr = 0.0
    t_bc = 0.0
    if settings.enable_visual_element_detection:
        t0 = time.perf_counter()
        try:
            visual_report = detect_visual_elements(
                image=preprocessed.original,
                inspection_id=inspection_id,
                save_dir=evidence_save_dir,
            )
            t_qr = visual_report.qr_detection_ms
            t_bc = visual_report.barcode_detection_ms
            stages.append("VISUAL_ELEMENTS_COMPLETE")
        except Exception as e:
            stages.append("VISUAL_ELEMENTS_FAILED")
        t_visual = (time.perf_counter() - t0) * 1000

    # ── 7. Rule Engine Evaluation ──────────────────────────────────────────────
    t0 = time.perf_counter()
    rule_results = evaluate_rules(
        extracted_declarations=declarations,
        category=product_category,
        quality_score=quality_result.quality_score,
        is_imported=is_imported,
    )
    t_rules = time.perf_counter() - t0
    stages.append("RULES_EVALUATED")

    # ── 8. Evidence Generation ─────────────────────────────────────────────────
    t0 = time.perf_counter()
    evidence_list: list[EvidenceSlice] = []
    if ocr_result:
        for decl in declarations:
            if decl.extracted_value and decl.raw_ocr_text:
                matching_rule = next(
                    (r for r in rule_results if r.declaration_field == decl.field), None
                )
                ev = generate_evidence(
                    tokens=ocr_result.tokens,
                    original_image=preprocessed.original,
                    declaration_field=decl.field,
                    matched_text=decl.raw_ocr_text,
                    rule_id=matching_rule.rule_id if matching_rule else None,
                    inspection_id=inspection_id,
                    save_dir=evidence_save_dir,
                )
                if ev:
                    evidence_list.append(ev)

    # Append visual element evidence slices
    if visual_report and visual_report.evidence_slices:
        evidence_list.extend(visual_report.evidence_slices)

    t_ev = time.perf_counter() - t0
    stages.append("EVIDENCE_GENERATED")

    # ── 9. Confidence Scoring ─────────────────────────────────────────────────
    confidence = compute_confidence(
        image_quality_score=quality_result.quality_score,
        ocr_mean_confidence=ocr_result.mean_confidence if ocr_result else 0.0,
        extracted_declarations=declarations,
        rule_results=rule_results,
    )
    stages.append("CONFIDENCE_COMPUTED")

    # ── 10. Overall Assessment ────────────────────────────────────────────────
    overall = compute_overall_status(rule_results)
    status_map = {
        "PASS": InspectionStatus.PASS,
        "MANUAL_REVIEW": InspectionStatus.MANUAL_REVIEW,
        "POTENTIAL_NON_COMPLIANCE": InspectionStatus.POTENTIAL_NON_COMPLIANCE,
    }
    final_status = status_map.get(overall, InspectionStatus.MANUAL_REVIEW)
    stages.append("ASSESSMENT_COMPLETE")

    t_total = time.perf_counter() - t_start

    # Structured timing breakdown
    timing_breakdown = {
        "image_read_ms": round(t_read * 1000, 1),
        "quality_ms": round(t_qual * 1000, 1),
        "preprocessing_ms": round(t_prep * 1000, 1),
        "ocr_primary_ms": round(t_ocr1 * 1000, 1),
        "ocr_fallback_ms": round(t_ocr2 * 1000, 1),
        "declaration_extraction_ms": round(t_decl * 1000, 1),
        "font_analysis_ms": round(t_font * 1000, 1),
        "qr_detection_ms": round(t_qr, 1),
        "barcode_detection_ms": round(t_bc, 1),
        "visual_element_detection_ms": round(t_visual, 1),
        "rule_engine_ms": round(t_rules * 1000, 1),
        "evidence_ms": round(t_ev * 1000, 1),
        "total_pipeline_ms": round(t_total * 1000, 1),
        "ocr_passes_executed": passes_used,
    }

    # Timing log for bottleneck identification
    print(
        f"[PIPELINE TIMING] id={inspection_id} total={t_total:.2f}s | "
        f"read={t_read*1000:.1f}ms | qual={t_qual*1000:.1f}ms | prep={t_prep*1000:.1f}ms | "
        f"ocr1={t_ocr1*1000:.1f}ms | ocr2={t_ocr2*1000:.1f}ms | decl={t_decl*1000:.1f}ms | "
        f"font={t_font*1000:.1f}ms | visual={t_visual:.1f}ms | rules={t_rules*1000:.1f}ms | passes={passes_used}"
    )

    return PipelineResult(
        inspection_id=inspection_id,
        status=final_status,
        overall_assessment=overall,
        quality_result=quality_result,
        ocr_result=ocr_result,
        declarations=declarations,
        rule_results=rule_results,
        evidence=evidence_list,
        confidence=confidence,
        font_analysis=font_report,
        visual_elements=visual_report,
        processing_time_seconds=round(t_total, 3),
        pipeline_stages=stages,
        timing_breakdown=timing_breakdown,
        error=error,
    )

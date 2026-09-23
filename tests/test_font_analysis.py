"""
MetroNetra — Unit & Integration Tests for Font Size and Readability Analysis
Tests:
- Visual font height calculation (pixels)
- Calibration honesty ("Not calibrated" when uncalibrated)
- Readability classification (READABLE, LOW READABILITY, REVIEW)
- Non-punitive compliance statuses (PASS, REVIEW)
- Backward compatibility for legacy inspection records
- API schemas and response contracts
- PDF report generation with font analysis
"""
import pytest
import numpy as np
import cv2
import os
from datetime import datetime
from sqlalchemy.orm import Session

from backend.services.ocr import OCRToken
from backend.services.declaration_extractor import ExtractedDeclaration
from backend.services.font_analysis import (
    analyze_font_and_readability,
    DeclarationFontAnalysis,
    FontReadabilityReport,
)
from backend.database.models import (
    Base, User, UserRole, Inspection, InspectionImage,
    Declaration, RuleResult, Evidence, FontAnalysisRecord,
    ProductCategory, InspectionStatus, DeclarationStatus
)
from backend.database.database import SessionLocal, engine, init_db
from backend.schemas.schemas import FontAnalysisItemSchema, AnalysisResponse, InspectionDetail
from backend.reports.report_generator import generate_pdf_report
from backend.config import settings


@pytest.fixture
def synthetic_image():
    """Create a synthetic 400x600 image with text-like high-contrast patterns."""
    img = np.ones((400, 600, 3), dtype=np.uint8) * 240
    # Draw dark text regions
    cv2.putText(img, "MRP Rs. 150.00", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
    cv2.putText(img, "NET QTY: 500 g", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
    cv2.putText(img, "MFD: 10/2026", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    return img


@pytest.fixture
def sample_declarations_and_tokens():
    """Create mock declarations and OCR tokens with bounding boxes."""
    tokens = [
        OCRToken(
            text="MRP Rs. 150.00",
            confidence=0.95,
            bbox=[60, 48, 90, 300],  # h = 30px
            polygon=[[48, 60], [300, 60], [300, 90], [48, 90]],
            ocr_pass="original"
        ),
        OCRToken(
            text="NET QTY: 500 g",
            confidence=0.92,
            bbox=[120, 48, 148, 280],  # h = 28px
            polygon=[[48, 120], [280, 120], [280, 148], [48, 148]],
            ocr_pass="original"
        ),
        OCRToken(
            text="MFD: 10/2026",
            confidence=0.55,  # Moderate confidence
            bbox=[180, 48, 206, 250],  # h = 26px
            polygon=[[48, 180], [250, 180], [250, 206], [48, 206]],
            ocr_pass="original"
        ),
    ]

    declarations = [
        ExtractedDeclaration(
            field="MRP",
            extracted_value="₹150.00",
            raw_ocr_text="MRP Rs. 150.00",
            normalized_value="150.00",
            extraction_confidence=0.95,
            source_ocr_token_index=0,
        ),
        ExtractedDeclaration(
            field="NET_QUANTITY",
            extracted_value="500 g",
            raw_ocr_text="NET QTY: 500 g",
            normalized_value="500 g",
            extraction_confidence=0.92,
            source_ocr_token_index=1,
        ),
        ExtractedDeclaration(
            field="MFG_DATE",
            extracted_value="10/2026",
            raw_ocr_text="MFD: 10/2026",
            normalized_value="2026-10-01",
            extraction_confidence=0.60,
            source_ocr_token_index=2,
        ),
        ExtractedDeclaration(
            field="CONSUMER_CARE",
            extracted_value=None,
            raw_ocr_text=None,
            normalized_value=None,
            extraction_confidence=0.0,
            source_ocr_token_index=None,
        ),
    ]

    return declarations, tokens


def test_font_analysis_uncalibrated_honesty(synthetic_image, sample_declarations_and_tokens):
    """
    Requirement 4: Do NOT claim exact mm measurement without physical calibration.
    Physical size must be explicitly reported as 'Not calibrated'.
    Visual text height must be reported in pixels.
    """
    declarations, tokens = sample_declarations_and_tokens
    report = analyze_font_and_readability(
        image_np=synthetic_image,
        tokens=tokens,
        declarations=declarations,
        pixels_per_mm=None,  # Uncalibrated
    )

    assert isinstance(report, FontReadabilityReport)
    assert report.is_calibrated is False
    assert report.calibration_status == "Not calibrated"

    mrp_item = next(it for it in report.items if it.field == "MRP")
    assert mrp_item.physical_size == "Not calibrated"
    assert mrp_item.text_height_px is not None
    assert mrp_item.text_height_px == 30  # 90 - 60
    assert mrp_item.readability == "READABLE"
    assert mrp_item.status == "PASS"
    assert mrp_item.confidence > 0.70


def test_font_analysis_calibrated_mode(synthetic_image, sample_declarations_and_tokens):
    """When calibrated reference scale is available, report estimated mm with disclaimer."""
    declarations, tokens = sample_declarations_and_tokens
    # Assume 10 pixels per mm
    report = analyze_font_and_readability(
        image_np=synthetic_image,
        tokens=tokens,
        declarations=declarations,
        pixels_per_mm=10.0,
    )

    assert report.is_calibrated is True
    assert "Calibrated" in report.calibration_status

    mrp_item = next(it for it in report.items if it.field == "MRP")
    assert "mm (calibrated)" in mrp_item.physical_size
    assert mrp_item.text_height_px == 30
    assert "3.0 mm" in mrp_item.physical_size


def test_font_analysis_undetected_declaration(synthetic_image, sample_declarations_and_tokens):
    """
    Requirement 6: Undetected or low-confidence items must route to REVIEW,
    not declare a legal violation.
    """
    declarations, tokens = sample_declarations_and_tokens
    report = analyze_font_and_readability(
        image_np=synthetic_image,
        tokens=tokens,
        declarations=declarations,
    )

    cc_item = next(it for it in report.items if it.field == "CONSUMER_CARE")
    assert cc_item.text_height_px is None
    assert cc_item.readability == "REVIEW"
    assert cc_item.status == "REVIEW"
    assert "not detected" in cc_item.notes.lower()


def test_font_analysis_readability_categories(synthetic_image):
    """
    Requirement 5: Readability must categorize into READABLE, LOW READABILITY, or REVIEW.
    """
    # Create tokens with varying quality
    tokens = [
        OCRToken(text="Sharp High Conf", confidence=0.98, bbox=[50, 50, 85, 200], ocr_pass="original"),
        OCRToken(text="Marginal Conf", confidence=0.38, bbox=[100, 50, 110, 150], ocr_pass="original"),
        OCRToken(text="Poor Conf", confidence=0.15, bbox=[150, 50, 156, 120], ocr_pass="original"),
    ]
    decls = [
        ExtractedDeclaration(field="MRP", extracted_value="Sharp", raw_ocr_text="Sharp High Conf", normalized_value="Sharp", extraction_confidence=0.98, source_ocr_token_index=0),
        ExtractedDeclaration(field="GENERIC_NAME", extracted_value="Marginal", raw_ocr_text="Marginal Conf", normalized_value="Marginal", extraction_confidence=0.38, source_ocr_token_index=1),
        ExtractedDeclaration(field="EXPIRY_DATE", extracted_value="Poor", raw_ocr_text="Poor Conf", normalized_value="Poor", extraction_confidence=0.15, source_ocr_token_index=2),
    ]

    report = analyze_font_and_readability(synthetic_image, tokens, decls)
    categories = {it.readability for it in report.items}
    # All categories must be from the defined set
    for cat in categories:
        assert cat in {"READABLE", "LOW READABILITY", "REVIEW"}


def test_schema_backward_compatibility():
    """
    Requirement 15: Schemas must be backward compatible and not fail when font_analysis is missing.
    """
    data = {
        "id": 1,
        "inspection_id": "LM-2026-999999",
        "product_name": "Test Product",
        "brand_name": "Test Brand",
        "product_category": ProductCategory.PACKAGED_FOOD,
        "status": InspectionStatus.PASS,
        "overall_confidence": 0.88,
        "is_demo": False,
        "demo_label": None,
        "notes": None,
        "created_at": datetime.utcnow(),
        "completed_at": datetime.utcnow(),
        "declarations": [],
        "rule_results": [],
        # font_analysis is intentionally omitted (simulating old inspection)
    }
    detail = InspectionDetail.model_validate(data)
    assert detail.font_analysis is None
    assert detail.inspection_id == "LM-2026-999999"


def test_database_persistence_and_backward_compatibility():
    """
    Test storing and retrieving FontAnalysisRecord in SQLite.
    Verify that an inspection without font records returns empty list without error.
    """
    init_db()
    db: Session = SessionLocal()
    try:
        # Create a test user
        user = db.query(User).filter(User.username == "font_test_user").first()
        if not user:
            user = User(
                username="font_test_user",
                full_name="Font Tester",
                role=UserRole.INSPECTOR,
                hashed_password="fake",
            )
            db.add(user)
            db.commit()
            db.refresh(user)

        # 1. Test legacy inspection (no font analysis records)
        legacy_insp = Inspection(
            inspection_id="LM-2026-LEGACY",
            inspector_id=user.id,
            product_name="Legacy Product",
            product_category=ProductCategory.PACKAGED_FOOD,
            status=InspectionStatus.PASS,
        )
        db.add(legacy_insp)
        db.commit()
        db.refresh(legacy_insp)

        assert legacy_insp.font_analysis_records == []

        # 2. Test new inspection with font analysis records
        new_insp = Inspection(
            inspection_id="LM-2026-NEWFONT",
            inspector_id=user.id,
            product_name="New Font Product",
            product_category=ProductCategory.PACKAGED_FOOD,
            status=InspectionStatus.PASS,
        )
        db.add(new_insp)
        db.commit()
        db.refresh(new_insp)

        fa_row = FontAnalysisRecord(
            inspection_id=new_insp.id,
            declaration_field="MRP",
            text_height_px=32,
            physical_size="Not calibrated",
            readability="READABLE",
            confidence=0.94,
            status="PASS",
            ocr_confidence=0.95,
            contrast_score=24.5,
            sharpness_score=65.2,
            bbox_json=[60, 48, 92, 300],
            notes="Clear legibility: estimated height 32px.",
        )
        db.add(fa_row)
        db.commit()

        db.refresh(new_insp)
        assert len(new_insp.font_analysis_records) == 1
        record = new_insp.font_analysis_records[0]
        assert record.declaration_field == "MRP"
        assert record.text_height_px == 32
        assert record.physical_size == "Not calibrated"
        assert record.readability == "READABLE"
        assert record.status == "PASS"

        # 3. Test PDF report generator with font records
        pdf_path = generate_pdf_report(new_insp, db, settings.reports_dir)
        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 1000

        # 4. Test PDF report generator with legacy inspection (no font records)
        pdf_legacy_path = generate_pdf_report(legacy_insp, db, settings.reports_dir)
        assert os.path.exists(pdf_legacy_path)
        assert os.path.getsize(pdf_legacy_path) > 1000

    finally:
        # Cleanup
        db.query(FontAnalysisRecord).filter(FontAnalysisRecord.declaration_field == "MRP").delete()
        db.query(Inspection).filter(Inspection.inspection_id.in_(["LM-2026-LEGACY", "LM-2026-NEWFONT"])).delete()
        db.commit()
        db.close()


def test_api_endpoints_font_analysis():
    """
    Test API endpoints /api/inspections/{id} and /api/inspections/{id}/font-analysis
    via FastAPI TestClient for both new and legacy inspections.
    """
    from fastapi.testclient import TestClient
    from backend.main import app
    from backend.auth.auth import create_access_token

    init_db()
    db: Session = SessionLocal()
    client = TestClient(app)
    token = create_access_token({"sub": "admin", "role": "ADMIN"})
    headers = {"Authorization": f"Bearer {token}"}

    try:
        user = db.query(User).filter(User.username == "admin").first()
        if not user:
            user = User(username="admin", full_name="Admin", role=UserRole.ADMIN, hashed_password="fake")
            db.add(user)
            db.commit()
            db.refresh(user)

        # Legacy inspection
        legacy = Inspection(
            inspection_id="LM-2026-API-LEGACY",
            inspector_id=user.id,
            product_name="Legacy API Product",
            product_category=ProductCategory.PACKAGED_FOOD,
            status=InspectionStatus.PASS,
        )
        # New inspection
        modern = Inspection(
            inspection_id="LM-2026-API-MODERN",
            inspector_id=user.id,
            product_name="Modern API Product",
            product_category=ProductCategory.PACKAGED_FOOD,
            status=InspectionStatus.PASS,
        )
        db.add(legacy)
        db.add(modern)
        db.commit()
        db.refresh(modern)

        # Add a font analysis record to modern
        fa = FontAnalysisRecord(
            inspection_id=modern.id,
            declaration_field="MRP",
            text_height_px=28,
            physical_size="Not calibrated",
            readability="READABLE",
            confidence=0.91,
            status="PASS",
            ocr_confidence=0.92,
            contrast_score=22.0,
            sharpness_score=55.0,
            notes="Clear text",
        )
        db.add(fa)
        db.commit()

        # 1. Test legacy GET /api/inspections/{id}
        r1 = client.get("/api/inspections/LM-2026-API-LEGACY", headers=headers)
        assert r1.status_code == 200
        data1 = r1.json()
        assert data1["font_analysis"] is None

        # 2. Test legacy GET /api/inspections/{id}/font-analysis
        r2 = client.get("/api/inspections/LM-2026-API-LEGACY/font-analysis", headers=headers)
        assert r2.status_code == 200
        data2 = r2.json()
        assert data2["available"] is False
        assert data2["message"] == "Not available for this inspection"

        # 3. Test modern GET /api/inspections/{id}
        r3 = client.get("/api/inspections/LM-2026-API-MODERN", headers=headers)
        assert r3.status_code == 200
        data3 = r3.json()
        assert data3["font_analysis"] is not None
        assert len(data3["font_analysis"]) == 1
        assert data3["font_analysis"][0]["declaration_field"] == "MRP"
        assert data3["font_analysis"][0]["text_height_px"] == 28
        assert data3["font_analysis"][0]["physical_size"] == "Not calibrated"
        assert data3["font_analysis"][0]["readability"] == "READABLE"

        # 4. Test modern GET /api/inspections/{id}/font-analysis
        r4 = client.get("/api/inspections/LM-2026-API-MODERN/font-analysis", headers=headers)
        assert r4.status_code == 200
        data4 = r4.json()
        assert data4["available"] is True
        assert len(data4["items"]) == 1
        assert data4["items"][0]["declaration_field"] == "MRP"
        assert data4["items"][0]["text_height_px"] == 28

    finally:
        db.query(FontAnalysisRecord).filter(FontAnalysisRecord.declaration_field == "MRP").delete()
        db.query(Inspection).filter(Inspection.inspection_id.in_(["LM-2026-API-LEGACY", "LM-2026-API-MODERN"])).delete()
        db.commit()
        db.close()


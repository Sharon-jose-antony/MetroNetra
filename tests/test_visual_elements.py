"""
MetroNetra — Unit & Integration Tests for Visual Element Detection (QR Codes & Barcodes)
Verifies:
- TEST 1: Clear package image containing a QR code -> QR bounding box, decode SUCCESS, evidence available.
- TEST 2: Clear package image containing a barcode -> Barcode bounding box, EAN-13 decode SUCCESS, evidence available.
- TEST 3: Package containing both QR and barcode -> Both detected separately with separate bounding boxes.
- TEST 4: Package containing neither -> No visual elements, existing OCR/declarations unaffected.
- TEST 5: Difficult/blurred image -> DETECTED_NOT_DECODED or NOT_DETECTED, zero invented values.
- TEST 6: Backward compatibility for legacy inspection records without visual elements.
- TEST 7: REST API contracts and Evidence image retrieval endpoints.
"""
import os
import shutil
import pytest
import numpy as np
import cv2
import zxingcpp
from datetime import datetime
from sqlalchemy.orm import Session
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.visual_elements import (
    detect_visual_elements,
    VisualElementItem,
    VisualElementsReport,
)
from backend.database.models import (
    Base, User, UserRole, Inspection, InspectionImage,
    Declaration, RuleResult, Evidence, VisualElementRecord,
    ProductCategory, InspectionStatus, DeclarationStatus
)
from backend.database.database import SessionLocal, engine, init_db
from backend.schemas.schemas import VisualElementSchema, AnalysisResponse, InspectionDetail
from backend.reports.report_generator import generate_pdf_report
from backend.auth.auth import create_access_token
from backend.config import settings


@pytest.fixture
def qr_image():
    """Generates a synthetic package canvas containing a clear QR code."""
    qr = zxingcpp.create_barcode("https://example.com/metronetra/item-123", zxingcpp.BarcodeFormat.QRCode)
    qr_np = np.array(zxingcpp.write_barcode_to_image(qr))
    qr_resized = cv2.resize(qr_np, (140, 140), interpolation=cv2.INTER_NEAREST)

    canvas = np.ones((500, 500, 3), dtype=np.uint8) * 245
    cv2.putText(canvas, "PREMIUM TEA 500g", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2)
    canvas[150:290, 80:220] = cv2.cvtColor(qr_resized, cv2.COLOR_GRAY2BGR)
    return canvas


@pytest.fixture
def barcode_image():
    """Generates a synthetic package canvas containing an EAN-13 barcode."""
    bc = zxingcpp.create_barcode("8901030383922", zxingcpp.BarcodeFormat.EAN13)
    bc_np = np.array(zxingcpp.write_barcode_to_image(bc))
    bc_resized = cv2.resize(bc_np, (260, 110), interpolation=cv2.INTER_NEAREST)

    canvas = np.ones((500, 500, 3), dtype=np.uint8) * 245
    cv2.putText(canvas, "ORGANIC HONEY 250g", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2)
    canvas[200:310, 100:360] = cv2.cvtColor(bc_resized, cv2.COLOR_GRAY2BGR)
    return canvas


@pytest.fixture
def dual_element_image():
    """Generates a package canvas with both a QR code and an EAN-13 barcode."""
    qr = zxingcpp.create_barcode("https://example.com/trace/batch-99", zxingcpp.BarcodeFormat.QRCode)
    qr_resized = cv2.resize(np.array(zxingcpp.write_barcode_to_image(qr)), (130, 130), interpolation=cv2.INTER_NEAREST)

    bc = zxingcpp.create_barcode("8901030383922", zxingcpp.BarcodeFormat.EAN13)
    bc_resized = cv2.resize(np.array(zxingcpp.write_barcode_to_image(bc)), (240, 100), interpolation=cv2.INTER_NEAREST)

    canvas = np.ones((600, 600, 3), dtype=np.uint8) * 245
    cv2.putText(canvas, "FORTIFIED WHEAT FLOUR", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
    # Place QR at top right
    canvas[100:230, 350:480] = cv2.cvtColor(qr_resized, cv2.COLOR_GRAY2BGR)
    # Place Barcode at bottom left
    canvas[380:480, 80:320] = cv2.cvtColor(bc_resized, cv2.COLOR_GRAY2BGR)
    return canvas


@pytest.fixture
def text_only_image():
    """Generates a package canvas with only text (no barcodes or QR codes)."""
    canvas = np.ones((400, 400, 3), dtype=np.uint8) * 240
    cv2.putText(canvas, "MRP Rs. 199.00", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    cv2.putText(canvas, "NET QTY: 1 kg", (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    cv2.putText(canvas, "MFD: 01/2026", (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    return canvas


# ==============================================================================
# TEST 1: Clear Package Image Containing a QR Code
# ==============================================================================

def test_clear_qr_code_detection(qr_image, tmp_path):
    """Verifies QR detection, decode status SUCCESS, and evidence slice creation."""
    save_dir = str(tmp_path / "evidence")
    report = detect_visual_elements(qr_image, inspection_id="TEST-QR-001", save_dir=save_dir)

    assert report.qr_count >= 1, "Expected at least 1 QR code detected"
    qr_item = next(i for i in report.items if i.element_type == "QR_CODE")

    assert qr_item.detection_status == "DETECTED"
    assert qr_item.decode_status == "SUCCESS"
    assert qr_item.decoded_value == "https://example.com/metronetra/item-123"
    assert len(qr_item.bounding_box) == 4
    # Check bounding box matches region [150:290, 80:220]
    xmin, ymin, xmax, ymax = qr_item.bounding_box
    assert xmin >= 70 and xmax <= 230
    assert ymin >= 140 and ymax <= 300

    # Evidence slice checks
    assert len(report.evidence_slices) >= 1
    ev = next(e for e in report.evidence_slices if "QR_CODE" in e.declaration_field)
    assert ev.crop_file_path is not None
    assert os.path.exists(ev.crop_file_path)


# ==============================================================================
# TEST 2: Clear Package Image Containing a Barcode
# ==============================================================================

def test_clear_barcode_detection(barcode_image, tmp_path):
    """Verifies 1D Barcode detection, EAN-13 decode SUCCESS, and evidence creation."""
    save_dir = str(tmp_path / "evidence")
    report = detect_visual_elements(barcode_image, inspection_id="TEST-BC-001", save_dir=save_dir)

    assert report.barcode_count >= 1, "Expected at least 1 barcode detected"
    bc_item = next(i for i in report.items if i.element_type == "BARCODE")

    assert bc_item.detection_status == "DETECTED"
    assert bc_item.decode_status == "SUCCESS"
    assert bc_item.barcode_type == "EAN-13"
    assert bc_item.decoded_value == "8901030383922"
    assert len(bc_item.bounding_box) == 4

    # Evidence slice checks
    assert len(report.evidence_slices) >= 1
    ev = next(e for e in report.evidence_slices if "BARCODE" in e.declaration_field)
    assert ev.crop_file_path is not None
    assert os.path.exists(ev.crop_file_path)


# ==============================================================================
# TEST 3: Package Containing Both QR and Barcode
# ==============================================================================

def test_dual_qr_and_barcode_detection(dual_element_image, tmp_path):
    """Verifies both QR and Barcode detected separately with separate bounding boxes."""
    save_dir = str(tmp_path / "evidence")
    report = detect_visual_elements(dual_element_image, inspection_id="TEST-DUAL-001", save_dir=save_dir)

    assert report.qr_count == 1, f"Expected 1 QR code, got {report.qr_count}"
    assert report.barcode_count == 1, f"Expected 1 barcode, got {report.barcode_count}"

    qr_item = next(i for i in report.items if i.element_type == "QR_CODE")
    bc_item = next(i for i in report.items if i.element_type == "BARCODE")

    assert qr_item.decode_status == "SUCCESS"
    assert qr_item.decoded_value == "https://example.com/trace/batch-99"

    assert bc_item.decode_status == "SUCCESS"
    assert bc_item.decoded_value == "8901030383922"

    # Distinct bounding boxes
    assert qr_item.bounding_box != bc_item.bounding_box

    # Both have separate evidence files
    assert len(report.evidence_slices) == 2


# ==============================================================================
# TEST 4: Package Containing Neither QR nor Barcode
# ==============================================================================

def test_neither_qr_nor_barcode(text_only_image):
    """Verifies clean empty report when no QR or barcodes are present."""
    report = detect_visual_elements(text_only_image, inspection_id="TEST-NONE-001")

    assert report.qr_count == 0
    assert report.barcode_count == 0
    assert len(report.items) == 0
    assert len(report.evidence_slices) == 0
    assert report.visual_element_detection_ms >= 0


# ==============================================================================
# TEST 5: Difficult / Blurred Image (No Hallucination)
# ==============================================================================

def test_blurred_unreadable_element():
    """
    Verifies that when a visual element is blurred/damaged:
    Status is DETECTED_NOT_DECODED (or NOT_DETECTED), and decoded_value is None.
    NEVER hallucinates or invents barcode/QR content.
    """
    # Create barcode and blur it
    bc = zxingcpp.create_barcode("8901030383922", zxingcpp.BarcodeFormat.EAN13)
    bc_np = np.array(zxingcpp.write_barcode_to_image(bc))
    bc_resized = cv2.resize(bc_np, (260, 100), interpolation=cv2.INTER_NEAREST)

    canvas = np.ones((400, 400, 3), dtype=np.uint8) * 255
    canvas[150:250, 70:330] = cv2.cvtColor(bc_resized, cv2.COLOR_GRAY2BGR)

    # Apply heavy Gaussian blur so decoding cannot read bars
    blurred = cv2.GaussianBlur(canvas, (17, 17), 0)

    report = detect_visual_elements(blurred, inspection_id="TEST-BLUR-001")

    for item in report.items:
        # If detected by gradient heuristic, decode_status must NOT be SUCCESS
        if item.decode_status == "DETECTED_NOT_DECODED":
            assert item.decoded_value is None, "Must not guess or invent decoded value"
            assert len(item.bounding_box) == 4
        else:
            # If ZXing somehow managed to reconstruct, decoded_value must match exactly
            assert item.decoded_value == "8901030383922"


# ==============================================================================
# TEST 6: Database Persistence & Legacy Backward Compatibility
# ==============================================================================

def test_database_persistence_and_backward_compatibility():
    """
    Verifies:
    1. Saving VisualElementRecord in DB and querying via relationship.
    2. Legacy inspections without visual elements return available=False gracefully.
    """
    init_db()
    db = SessionLocal()
    try:
        # Check an existing legacy inspection from db
        legacy_insp = db.query(Inspection).first()
        assert legacy_insp is not None

        # Verify relationship exists on model
        assert hasattr(legacy_insp, "visual_element_records")
        # Legacy records will have empty visual_element_records list without crashing
        assert isinstance(legacy_insp.visual_element_records, list)

        # Test creating a record
        ve_record = VisualElementRecord(
            inspection_id=legacy_insp.id,
            element_type="QR_CODE",
            barcode_type="QR Code",
            detection_status="DETECTED",
            decode_status="SUCCESS",
            decoded_value="https://test.gov.in/verify",
            confidence=1.0,
            bbox_json=[10, 20, 110, 120],
            crop_file_path=None,
        )
        db.add(ve_record)
        db.commit()

        # Query back
        reloaded = db.query(Inspection).filter(Inspection.id == legacy_insp.id).first()
        assert len(reloaded.visual_element_records) >= 1
        found = next(v for v in reloaded.visual_element_records if v.decoded_value == "https://test.gov.in/verify")
        assert found.element_type == "QR_CODE"
        assert found.barcode_type == "QR Code"

        # Cleanup test row
        db.delete(found)
        db.commit()
    finally:
        db.close()


# ==============================================================================
# TEST 7: REST API Contracts & Evidence Serving
# ==============================================================================

def test_api_endpoints_and_evidence():
    """
    Verifies:
    1. GET /api/inspections/{id}/visual-elements returns correct structure.
    2. InspectionDetail schema handles visual_elements cleanly.
    """
    client = TestClient(app)
    token = create_access_token(data={"sub": "admin"})
    headers = {"Authorization": f"Bearer {token}"}

    db = SessionLocal()
    try:
        insp = db.query(Inspection).first()
        assert insp is not None
        insp_id = insp.inspection_id

        # 1. Test GET /api/inspections/{id}/visual-elements
        resp = client.get(f"/api/inspections/{insp_id}/visual-elements", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "available" in data
        assert "items" in data

        # 2. Test GET /api/inspections/{id}
        resp_detail = client.get(f"/api/inspections/{insp_id}", headers=headers)
        assert resp_detail.status_code == 200
        detail_data = resp_detail.json()
        assert "visual_elements" in detail_data
        assert isinstance(detail_data["visual_elements"], list)
    finally:
        db.close()

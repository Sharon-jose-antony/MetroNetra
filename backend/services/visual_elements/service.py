"""
MetroNetra — Visual Element Detection Service (QR Codes & Barcodes)
Detects QR codes and 1D/2D barcodes on package imagery using high-performance
computer-vision routines and decoding engines.
Separates physical visual element detection from decoding to preserve bounding
boxes and evidence slices even when decoding fails.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from backend.config import settings
from backend.services.evidence import EvidenceSlice
from backend.services.preprocessing import crop_region

try:
    import zxingcpp
    HAS_ZXING = True
except ImportError:
    HAS_ZXING = False


@dataclass
class VisualElementItem:
    element_type: str                   # "QR_CODE" or "BARCODE"
    barcode_type: Optional[str] = None  # e.g. "EAN-13", "UPC-A", "CODE-128", "QR Code"
    bounding_box: List[int] = field(default_factory=list)  # [x1, y1, x2, y2]
    bbox_ymin: int = 0
    bbox_xmin: int = 0
    bbox_ymax: int = 0
    bbox_xmax: int = 0
    detection_status: str = "DETECTED"  # "DETECTED" or "NOT_DETECTED"
    decode_status: str = "SUCCESS"      # "SUCCESS", "DETECTED_NOT_DECODED"
    decoded_value: Optional[str] = None
    confidence: float = 1.0
    crop_file_path: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class VisualElementsReport:
    items: List[VisualElementItem] = field(default_factory=list)
    evidence_slices: List[EvidenceSlice] = field(default_factory=list)
    qr_count: int = 0
    barcode_count: int = 0
    qr_detection_ms: float = 0.0
    barcode_detection_ms: float = 0.0
    visual_element_detection_ms: float = 0.0


def _format_barcode_type(raw_format: str) -> str:
    """Format raw library format name into standard presentation text."""
    fmt = str(raw_format).replace("BarcodeFormat.", "").strip()
    mapping = {
        "EAN13": "EAN-13",
        "EAN8": "EAN-8",
        "UPCA": "UPC-A",
        "UPCE": "UPC-E",
        "Code128": "CODE-128",
        "Code39": "CODE-39",
        "Code93": "CODE-93",
        "ITF": "ITF-14",
        "Codabar": "Codabar",
        "DataMatrix": "Data Matrix",
        "QRCode": "QR Code",
        "MicroQRCode": "Micro QR Code",
        "RMQRCode": "rMQR Code",
        "PDF417": "PDF417",
        "Aztec": "Aztec",
    }
    return mapping.get(fmt, fmt)


def _detect_with_zxing(
    image: np.ndarray,
) -> Tuple[List[VisualElementItem], float, float]:
    """
    Run high-performance ZXing detection on the image.
    Converts image to BGR and Grayscale, sets return_errors=True, try_rotate=True,
    and runs multi-pass contrast checks to reliably capture both decodable and
    undecodable/damaged barcodes and QR codes.
    Returns (items, qr_time_ms, barcode_time_ms).
    """
    if not HAS_ZXING or image is None:
        return [], 0.0, 0.0

    t0 = time.perf_counter()
    h, w = image.shape[:2]
    items: List[VisualElementItem] = []

    # 1. Prepare multi-channel passes:
    # Preprocessing pipeline produces RGB numpy arrays. ZXing and OpenCV expect BGR or Grayscale.
    if len(image.shape) == 2:
        img_gray = image
        img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        passes = [img_bgr, img_gray]
    elif len(image.shape) == 3:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        passes = [img_bgr, img_gray, image]
    else:
        passes = [image]
        img_gray = image

    # Add CLAHE pass for low-contrast or glare-affected package labels
    if len(img_gray.shape) == 2:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        passes.append(clahe.apply(img_gray))

    raw_results = []
    seen_signatures = set()

    for p_img in passes:
        try:
            res = zxingcpp.read_barcodes(
                p_img,
                return_errors=True,
                try_rotate=True,
                try_downscale=True,
                try_invert=True,
            )
        except Exception:
            res = []

        for r in res:
            pos = r.position
            # Signature based on position to avoid redundant work across passes
            sig = (
                getattr(r.format, "name", str(r.format)),
                round(pos.top_left.x, -1),
                round(pos.top_left.y, -1),
                round(pos.bottom_right.x, -1),
                round(pos.bottom_right.y, -1),
            )
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                raw_results.append(r)

        # If any valid decoded barcode was found on this pass, we have sufficient results
        if any(getattr(r, "valid", False) and not getattr(r, "error", None) for r in raw_results):
            break

    for r in raw_results:
        raw_fmt = getattr(r.format, "name", str(r.format))
        is_qr = "QR" in raw_fmt.upper() or "AZTEC" in raw_fmt.upper()
        if is_qr:
            elem_type = "QR_CODE"
            bc_type = "QR Code"
        else:
            elem_type = "BARCODE"
            bc_type = _format_barcode_type(raw_fmt)

        pos = r.position
        pts_x = [pos.top_left.x, pos.top_right.x, pos.bottom_right.x, pos.bottom_left.x]
        pts_y = [pos.top_left.y, pos.top_right.y, pos.bottom_right.y, pos.bottom_left.y]

        xmin = max(0, int(min(pts_x)))
        ymin = max(0, int(min(pts_y)))
        xmax = min(w, int(max(pts_x)))
        ymax = min(h, int(max(pts_y)))

        # Ensure valid area
        if xmax <= xmin or ymax <= ymin:
            continue

        # In 1D barcodes, ZXing returns a scanline slice (often height < 20px).
        # Expand vertically so the bounding box encloses the full barcode bars.
        if elem_type == "BARCODE":
            bw = xmax - xmin
            bh = ymax - ymin
            if bh < 25 and bw >= 40:
                expand_y = int(bw * 0.25)
                ymin = max(0, ymin - expand_y)
                ymax = min(h, ymax + expand_y)

        # Separate Detection from Decoding
        has_error = getattr(r, "error", None) is not None
        is_valid = getattr(r, "valid", False) and (not has_error)
        text_val = r.text.strip() if r.text and len(r.text.strip()) > 0 else None

        if is_valid and text_val:
            decode_status = "SUCCESS"
            decoded_value = text_val
            conf = 1.0
            notes = f"Detected and decoded via ZXing engine ({bc_type})"
        else:
            decode_status = "DETECTED_NOT_DECODED"
            decoded_value = None
            conf = 0.55
            err_detail = f": {r.error}" if getattr(r, "error", None) else ""
            notes = f"Visually detected via ZXing engine ({bc_type}){err_detail}"

        # Filter out tiny noise false positives when undecoded
        if decode_status == "DETECTED_NOT_DECODED":
            if is_qr or "Matrix" in bc_type or "Aztec" in bc_type:
                if (xmax - xmin) < 24 or (ymax - ymin) < 24 or (xmax - xmin) * (ymax - ymin) < 500:
                    continue
            else:
                if max(xmax - xmin, ymax - ymin) < 40 or min(xmax - xmin, ymax - ymin) < 10:
                    continue

        items.append(
            VisualElementItem(
                element_type=elem_type,
                barcode_type=bc_type,
                bounding_box=[xmin, ymin, xmax, ymax],
                bbox_ymin=ymin,
                bbox_xmin=xmin,
                bbox_ymax=ymax,
                bbox_xmax=xmax,
                detection_status="DETECTED",
                decode_status=decode_status,
                decoded_value=decoded_value,
                confidence=conf,
                notes=notes,
            )
        )

    t_total = (time.perf_counter() - t0) * 1000
    qr_count = sum(1 for i in items if i.element_type == "QR_CODE")
    bc_count = sum(1 for i in items if i.element_type == "BARCODE")
    total_found = max(1, qr_count + bc_count)
    qr_ms = t_total * (qr_count / total_found if qr_count else 0.5)
    bc_ms = t_total * (bc_count / total_found if bc_count else 0.5)

    return items, round(qr_ms, 2), round(bc_ms, 2)


def _detect_opencv_qr(
    image: np.ndarray,
    existing_items: List[VisualElementItem],
) -> Tuple[List[VisualElementItem], float]:
    """Fallback QR detection using OpenCV QRCodeDetector."""
    t0 = time.perf_counter()
    h, w = image.shape[:2]
    new_items: List[VisualElementItem] = []

    if len(image.shape) == 3:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    try:
        qrd = cv2.QRCodeDetector()
        ret, pts = qrd.detect(img_bgr)
        if ret and pts is not None:
            pts_reshaped = pts.reshape(-1, 2)
            xmin = max(0, int(np.min(pts_reshaped[:, 0])))
            ymin = max(0, int(np.min(pts_reshaped[:, 1])))
            xmax = min(w, int(np.max(pts_reshaped[:, 0])))
            ymax = min(h, int(np.max(pts_reshaped[:, 1])))
            if xmax > xmin and ymax > ymin:
                # Check overlap with any existing detections (including barcodes)
                overlap = False
                for ex in existing_items:
                    ixmin = max(xmin, ex.bbox_xmin)
                    iymin = max(ymin, ex.bbox_ymin)
                    ixmax = min(xmax, ex.bbox_xmax)
                    iymax = min(ymax, ex.bbox_ymax)
                    if ixmax > ixmin and iymax > iymin:
                        overlap = True
                        break

                if not overlap:
                    # Attempt decode
                    decoded_text, _ = qrd.decode(img_bgr, pts)
                    text_val = decoded_text.strip() if decoded_text and len(decoded_text.strip()) > 0 else None
                    if text_val:
                        decode_status = "SUCCESS"
                        decoded_value = text_val
                        conf = 0.90
                    else:
                        is_valid_cand = True
                        bw = xmax - xmin
                        bh = ymax - ymin
                        aspect = bw / float(bh) if bh > 0 else 0
                        if not (0.70 <= aspect <= 1.45) or bw < 25 or bh < 25:
                            is_valid_cand = False

                        # Verify candidate ROI with ZXing (even with return_errors=True)
                        if is_valid_cand and HAS_ZXING:
                            roi_qr = img_bgr[ymin:ymax, xmin:xmax]
                            try:
                                roi_res = zxingcpp.read_barcodes(roi_qr, return_errors=True)
                                if not any("QR" in getattr(r.format, "name", str(r.format)).upper() for r in roi_res):
                                    is_valid_cand = False
                            except Exception:
                                is_valid_cand = False

                        if is_valid_cand:
                            decode_status = "DETECTED_NOT_DECODED"
                            decoded_value = None
                            conf = 0.50
                        else:
                            decode_status = None

                    if decode_status is not None:
                        new_items.append(
                        VisualElementItem(
                            element_type="QR_CODE",
                            barcode_type="QR Code",
                            bounding_box=[xmin, ymin, xmax, ymax],
                            bbox_ymin=ymin,
                            bbox_xmin=xmin,
                            bbox_ymax=ymax,
                            bbox_xmax=xmax,
                            detection_status="DETECTED",
                            decode_status=decode_status,
                            decoded_value=decoded_value,
                            confidence=conf,
                            notes="Detected via OpenCV QRCodeDetector",
                        )
                    )
    except Exception:
        pass

    t_ms = (time.perf_counter() - t0) * 1000
    return new_items, round(t_ms, 2)


def _detect_opencv_barcode(
    image: np.ndarray,
    existing_items: List[VisualElementItem],
) -> Tuple[List[VisualElementItem], float]:
    """Fallback barcode detection using OpenCV BarcodeDetector and morphological heuristics."""
    t0 = time.perf_counter()
    h, w = image.shape[:2]
    new_items: List[VisualElementItem] = []

    if len(image.shape) == 3:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image
        img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # 1. OpenCV BarcodeDetector
    try:
        bd = cv2.barcode.BarcodeDetector()
        ret, pts = bd.detect(img_bgr)
        if ret and pts is not None:
            pts_array = np.array(pts)
            for i in range(len(pts_array)):
                box = pts_array[i].reshape(-1, 2)
                xmin = max(0, int(np.min(box[:, 0])))
                ymin = max(0, int(np.min(box[:, 1])))
                xmax = min(w, int(np.max(box[:, 0])))
                ymax = min(h, int(np.max(box[:, 1])))
                if xmax <= xmin or ymax <= ymin:
                    continue

                # Expand thin scanlines
                bw = xmax - xmin
                bh = ymax - ymin
                if bh < 25 and bw >= 40:
                    expand_y = int(bw * 0.25)
                    ymin = max(0, ymin - expand_y)
                    ymax = min(h, ymax + expand_y)

                # Attempt decode
                try:
                    dec_ret, dec_info, dec_type = bd.decodeWithType(img_bgr, pts_array[i:i+1])
                    if dec_ret and dec_info and dec_info[0]:
                        decode_status = "SUCCESS"
                        decoded_value = dec_info[0].strip()
                        b_type = _format_barcode_type(dec_type[0]) if dec_type else "BARCODE"
                        conf = 0.90
                    else:
                        decode_status = "DETECTED_NOT_DECODED"
                        decoded_value = None
                        b_type = "1D Barcode"
                        conf = 0.50
                except Exception:
                    decode_status = "DETECTED_NOT_DECODED"
                    decoded_value = None
                    b_type = "1D Barcode"
                    conf = 0.50

                # Check overlap
                overlap = any(
                    ex.element_type == "BARCODE" and
                    max(xmin, ex.bbox_xmin) < min(xmax, ex.bbox_xmax) and
                    max(ymin, ex.bbox_ymin) < min(ymax, ex.bbox_ymax)
                    for ex in existing_items
                )
                if not overlap:
                    new_items.append(
                        VisualElementItem(
                            element_type="BARCODE",
                            barcode_type=b_type,
                            bounding_box=[xmin, ymin, xmax, ymax],
                            bbox_ymin=ymin,
                            bbox_xmin=xmin,
                            bbox_ymax=ymax,
                            bbox_xmax=xmax,
                            detection_status="DETECTED",
                            decode_status=decode_status,
                            decoded_value=decoded_value,
                            confidence=conf,
                            notes="Detected via OpenCV BarcodeDetector",
                        )
                    )
    except Exception:
        pass

    # 2. Morphological gradient heuristic for blurred/un-decodable 1D barcodes
    if not existing_items and not new_items:
        try:
            grad_x = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=-1)
            grad_y = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=-1)
            gradient = cv2.subtract(grad_x, grad_y)
            gradient = cv2.convertScaleAbs(gradient)

            blurred = cv2.blur(gradient, (9, 9))
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            closed = cv2.erode(closed, None, iterations=2)
            closed = cv2.dilate(closed, None, iterations=2)

            total_area = h * w
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                bx, by, bw, bh = cv2.boundingRect(c)
                aspect = bw / float(bh) if bh > 0 else 0
                area = cv2.contourArea(c)
                # Filter for typical 1D barcode aspect ratio and size
                if (
                    1500 < area < 0.30 * total_area
                    and 1.1 <= aspect <= 6.0
                    and 50 <= bw <= 0.70 * w
                    and 20 <= bh <= 0.50 * h
                ):
                    crop_candidate = img_gray[by:by+bh, bx:bx+bw]
                    _, thresh_c = cv2.threshold(crop_candidate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    transitions = []
                    for frac in [0.35, 0.50, 0.65]:
                        row_idx = int(bh * frac)
                        if 0 <= row_idx < bh:
                            transitions.append(np.count_nonzero(np.diff(thresh_c[row_idx, :] > 128)))

                    # Real barcodes have high frequency alternating lines
                    if transitions and np.mean(transitions) >= 12:
                        confirmed = False
                        decoded_val = None
                        status = "DETECTED_NOT_DECODED"
                        b_type = "1D Barcode"
                        if HAS_ZXING:
                            try:
                                roi_res = zxingcpp.read_barcodes(crop_candidate, return_errors=True)
                                if roi_res:
                                    confirmed = True
                                    r_roi = roi_res[0]
                                    if getattr(r_roi, "valid", False) and not getattr(r_roi, "error", None) and r_roi.text:
                                        status = "SUCCESS"
                                        decoded_val = r_roi.text.strip()
                                    b_type = _format_barcode_type(getattr(r_roi.format, "name", str(r_roi.format)))
                            except Exception:
                                pass
                        elif np.mean(transitions) >= 35 and min(transitions) >= 28:
                            confirmed = True

                        if confirmed:
                            new_items.append(
                                VisualElementItem(
                                    element_type="BARCODE",
                                    barcode_type=b_type,
                                    bounding_box=[bx, by, bx + bw, by + bh],
                                    bbox_ymin=by,
                                    bbox_xmin=bx,
                                    bbox_ymax=by + bh,
                                    bbox_xmax=bx + bw,
                                    detection_status="DETECTED",
                                    decode_status=status,
                                    decoded_value=decoded_val,
                                    confidence=0.85 if status == "SUCCESS" else 0.45,
                                    notes="Visual barcode stripe pattern localized",
                                )
                            )
                            break
        except Exception:
            pass

    t_ms = (time.perf_counter() - t0) * 1000
    return new_items, round(t_ms, 2)


def _deduplicate_items(items: List[VisualElementItem]) -> List[VisualElementItem]:
    """
    Deduplicates visual elements when multiple scanlines match the same barcode
    or one detection is contained within another (e.g. Micro-QR inside a standard QR).
    Merges bounding boxes of overlapping scanlines to encompass the full barcode area.
    """
    if len(items) <= 1:
        return items

    # Sort so SUCCESS decodes come first, followed by larger area items
    items.sort(
        key=lambda x: (
            1 if x.decode_status == "SUCCESS" else 0,
            (x.bbox_xmax - x.bbox_xmin) * (x.bbox_ymax - x.bbox_ymin),
        ),
        reverse=True,
    )

    kept: List[VisualElementItem] = []
    for item in items:
        area_item = (item.bbox_xmax - item.bbox_xmin) * (item.bbox_ymax - item.bbox_ymin)
        w_item = item.bbox_xmax - item.bbox_xmin
        is_duplicate = False

        for k in kept:
            if item.element_type == k.element_type:
                ixmin = max(item.bbox_xmin, k.bbox_xmin)
                iymin = max(item.bbox_ymin, k.bbox_ymin)
                ixmax = min(item.bbox_xmax, k.bbox_xmax)
                iymax = min(item.bbox_ymax, k.bbox_ymax)

                inter_w = max(0, ixmax - ixmin)
                inter_h = max(0, iymax - iymin)
                inter_area = inter_w * inter_h

                area_k = (k.bbox_xmax - k.bbox_xmin) * (k.bbox_ymax - k.bbox_ymin)
                min_area = min(area_item, area_k)

                # Check 2D intersection or 1D barcode scanline vertical alignment
                is_overlap = False
                if min_area > 0 and (inter_area / min_area) > 0.35:
                    is_overlap = True
                elif item.element_type == "BARCODE":
                    # Check if both items align horizontally across the same barcode bars
                    min_w = min(w_item, k.bbox_xmax - k.bbox_xmin)
                    if min_w > 0 and (inter_w / min_w) > 0.65:
                        vert_gap = max(0, max(item.bbox_ymin, k.bbox_ymin) - min(item.bbox_ymax, k.bbox_ymax))
                        if vert_gap < 0.60 * min_w:
                            is_overlap = True

                if is_overlap:
                    is_duplicate = True
                    # Expand the kept bounding box to fully envelope both scanline regions
                    k.bbox_xmin = min(k.bbox_xmin, item.bbox_xmin)
                    k.bbox_ymin = min(k.bbox_ymin, item.bbox_ymin)
                    k.bbox_xmax = max(k.bbox_xmax, item.bbox_xmax)
                    k.bbox_ymax = max(k.bbox_ymax, item.bbox_ymax)
                    k.bounding_box = [k.bbox_xmin, k.bbox_ymin, k.bbox_xmax, k.bbox_ymax]
                    break

        if not is_duplicate:
            kept.append(item)

    return kept


def detect_visual_elements(
    image: np.ndarray,
    inspection_id: str = "unknown",
    save_dir: Optional[str] = None,
) -> VisualElementsReport:
    """
    Main entry point for visual element detection.
    Scans the package image for QR codes and barcodes, computes bounding boxes,
    attempts decoding, and creates annotated evidence slices.
    """
    if not settings.enable_visual_element_detection or image is None:
        return VisualElementsReport()

    t_start = time.perf_counter()

    # Step 1: High-performance ZXing pass (decodes QR and standard 1D barcodes)
    items, qr_ms, bc_ms = _detect_with_zxing(image)

    # Step 2: OpenCV fallback for QR codes if none found
    has_qr = any(i.element_type == "QR_CODE" for i in items)
    if not has_qr:
        fb_qr_items, fb_qr_ms = _detect_opencv_qr(image, items)
        items.extend(fb_qr_items)
        qr_ms += fb_qr_ms

    # Step 3: OpenCV fallback for Barcodes if none found
    has_bc = any(i.element_type == "BARCODE" for i in items)
    if not has_bc:
        fb_bc_items, fb_bc_ms = _detect_opencv_barcode(image, items)
        items.extend(fb_bc_items)
        bc_ms += fb_bc_ms

    # Step 3.5: Deduplicate overlapping/contained sub-detections
    items = _deduplicate_items(items)

    # Step 4: Generate Evidence Crops
    evidence_slices: List[EvidenceSlice] = []
    qr_idx = 0
    bc_idx = 0

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for item in items:
        # Bounding box in [ymin, xmin, ymax, xmax] for crop_region
        bbox_slice = [item.bbox_ymin, item.bbox_xmin, item.bbox_ymax, item.bbox_xmax]
        crop = crop_region(image, bbox_slice, padding=16)

        # Visual bounding box highlight
        is_qr = item.element_type == "QR_CODE"
        color = (34, 197, 94) if is_qr else (59, 130, 246)  # Green for QR, Blue for Barcode
        field_name = (
            "QR_CODE" if is_qr and qr_idx == 0
            else f"QR_CODE_{qr_idx}" if is_qr
            else "BARCODE" if bc_idx == 0
            else f"BARCODE_{bc_idx}"
        )

        # Save annotated crop
        crop_path = None
        if save_dir and crop is not None:
            filename = f"{inspection_id}_{field_name}_evidence.jpg"
            crop_path = os.path.join(save_dir, filename)
            h_c, w_c = crop.shape[:2]
            annotated_crop = crop.copy()
            cv2.rectangle(annotated_crop, (2, 2), (w_c - 2, h_c - 2), color, 2)
            cv2.putText(
                annotated_crop,
                item.barcode_type or item.element_type,
                (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
            # Write annotated highlight image
            if len(annotated_crop.shape) == 3:
                write_img = cv2.cvtColor(annotated_crop, cv2.COLOR_RGB2BGR)
            else:
                write_img = annotated_crop
            cv2.imwrite(crop_path, write_img)
            item.crop_file_path = crop_path

        # Evidence slice representation for the system
        ev = EvidenceSlice(
            declaration_field=field_name,
            crop_image=crop,
            crop_file_path=crop_path,
            bbox=bbox_slice,
            ocr_text=item.decoded_value or f"[{item.element_type}: {item.decode_status}]",
            confidence=item.confidence,
            rule_id=None,
        )
        evidence_slices.append(ev)

        if is_qr:
            qr_idx += 1
        else:
            bc_idx += 1

    t_total = (time.perf_counter() - t_start) * 1000

    qr_count = sum(1 for i in items if i.element_type == "QR_CODE")
    barcode_count = sum(1 for i in items if i.element_type == "BARCODE")

    # Structured Debug Logging as requested
    bc_detected = "YES" if barcode_count > 0 else "NO"
    bc_decoded = "YES" if any(i.element_type == "BARCODE" and i.decode_status == "SUCCESS" for i in items) else "NO"

    print("[Visual Detection]")
    print(f"QR candidates: {qr_count}")
    print(f"Barcode candidates: {barcode_count}")
    print(f"Barcode detected: {bc_detected}")
    print(f"Barcode decoded: {bc_decoded}")
    if items:
        for i in items:
            print(f"Bounding box: {i.bounding_box} ({i.element_type}, status={i.decode_status}, decoded={i.decoded_value})")
    else:
        print("Bounding box: None")

    return VisualElementsReport(
        items=items,
        evidence_slices=evidence_slices,
        qr_count=qr_count,
        barcode_count=barcode_count,
        qr_detection_ms=round(qr_ms, 1),
        barcode_detection_ms=round(bc_ms, 1),
        visual_element_detection_ms=round(t_total, 1),
    )

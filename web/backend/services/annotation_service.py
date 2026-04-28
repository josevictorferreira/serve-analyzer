"""Local tennis-ball annotation session storage and YOLO export helpers."""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from typing import Any

import cv2

from ..paths import get_annotation_root, get_annotation_session_dir


CLASS_ID = 0
CLASS_NAME = "tennis_ball"
DEFAULT_BOX_SIZE_PX = 32
DEFAULT_MAX_FRAMES = 240
DEFAULT_CONFIDENCE = 0.20
RJTPP_REPO_ID = "RJTPP/tennis-ball-detection"
RJTPP_FILENAME = "best.pt"
REVIEWED_STATUSES = {"accepted", "corrected", "absent"}
SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

_RJTPP_MODEL: Any | None = None
_RJTPP_MODEL_ERROR: str | None = None


def create_annotation_session(
    video_path: str,
    source_filename: str,
    *,
    max_frames: int = DEFAULT_MAX_FRAMES,
    frame_step: int | None = None,
    prelabel: bool = True,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    """Create a local annotation session from a video file."""
    if max_frames < 1:
        raise ValueError("max_frames must be at least 1")
    if frame_step is not None and frame_step < 1:
        raise ValueError("frame_step must be at least 1")

    session_id = uuid.uuid4().hex[:12]
    session_dir = _safe_session_dir(session_id, create=True)
    frames_dir = os.path.join(session_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    ext = os.path.splitext(source_filename)[1].lower() or ".mp4"
    source_video = f"source{ext}"
    source_path = os.path.join(session_dir, source_video)

    try:
        shutil.copy2(video_path, source_path)
        session = _extract_session_frames(
            source_path,
            frames_dir,
            session_id=session_id,
            source_filename=source_filename,
            source_video=source_video,
            max_frames=max_frames,
            frame_step=frame_step,
            prelabel=prelabel,
            confidence=confidence,
        )
        _save_session(session)
        return get_annotation_session(session_id)
    except Exception:
        shutil.rmtree(session_dir, ignore_errors=True)
        raise


def list_annotation_sessions() -> list[dict[str, Any]]:
    """Return summaries for locally stored annotation sessions."""
    root = get_annotation_root()
    sessions: list[dict[str, Any]] = []
    for entry in os.listdir(root):
        if not SESSION_ID_RE.match(entry):
            continue
        manifest_path = os.path.join(root, entry, "session.json")
        if not os.path.isfile(manifest_path):
            continue
        try:
            session = _load_session(entry)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        sessions.append(_session_summary(session))
    sessions.sort(key=lambda item: item["updated_at"], reverse=True)
    return sessions


def get_annotation_session(session_id: str) -> dict[str, Any]:
    """Return a full annotation session manifest with current progress."""
    session = _load_session(session_id)
    session["progress"] = _progress_for(session)
    return session


def get_frame_image_path(session_id: str, frame_id: str) -> str:
    """Return the image path for a session frame."""
    session = _load_session(session_id)
    frame = _find_frame(session, frame_id)
    session_dir = os.path.realpath(_safe_session_dir(session_id))
    path = os.path.realpath(os.path.join(session_dir, frame["image_filename"]))
    if os.path.commonpath([session_dir, path]) != session_dir:
        raise ValueError("Invalid frame path")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Frame image not found: {frame_id}")
    return path


def review_frame(
    session_id: str,
    frame_id: str,
    action: str,
    bbox: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Apply a human review decision to a frame."""
    session = _load_session(session_id)
    frame = _find_frame(session, frame_id)
    action = action.lower()

    if action == "accept":
        prediction = frame.get("prediction")
        if not prediction or not prediction.get("bbox"):
            raise ValueError("Cannot accept a frame without a prediction")
        label_bbox = _clamp_bbox(prediction["bbox"], frame["width"], frame["height"])
        frame["label"] = _label_from_bbox(label_bbox, source="prediction")
        frame["status"] = "accepted"
    elif action == "correct":
        if bbox is None:
            raise ValueError("Corrected frames require a bbox")
        label_bbox = _clamp_bbox(bbox, frame["width"], frame["height"])
        frame["label"] = _label_from_bbox(label_bbox, source="manual")
        frame["status"] = "corrected"
    elif action == "absent":
        frame["label"] = None
        frame["status"] = "absent"
    elif action == "skip":
        frame["label"] = None
        frame["status"] = "skipped"
    else:
        raise ValueError(f"Unsupported review action: {action}")

    frame["reviewed_at"] = _now()
    session["updated_at"] = _now()
    session["progress"] = _progress_for(session)
    _save_session(session)
    return get_annotation_session(session_id)


def undo_frame_review(session_id: str, frame_id: str) -> dict[str, Any]:
    """Reset a reviewed frame back to pending."""
    session = _load_session(session_id)
    frame = _find_frame(session, frame_id)
    frame["status"] = "pending"
    frame["label"] = None
    frame.pop("reviewed_at", None)
    session["updated_at"] = _now()
    session["progress"] = _progress_for(session)
    _save_session(session)
    return get_annotation_session(session_id)


def export_yolo_dataset(session_id: str) -> dict[str, Any]:
    """Export reviewed labels as an Ultralytics YOLO dataset."""
    session = _load_session(session_id)
    reviewed_frames = [
        frame for frame in session["frames"] if frame.get("status") in REVIEWED_STATUSES
    ]
    if not reviewed_frames:
        raise ValueError("No reviewed frames are available to export")

    export_id = datetime.now(timezone.utc).strftime("yolo-%Y%m%d-%H%M%S")
    export_dir = os.path.join(_safe_session_dir(session_id), "exports", export_id)
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(export_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(export_dir, "labels", split), exist_ok=True)

    counts = {
        "train": {"images": 0, "positive": 0, "negative": 0},
        "val": {"images": 0, "positive": 0, "negative": 0},
        "test": {"images": 0, "positive": 0, "negative": 0},
    }

    session_dir = _safe_session_dir(session_id)
    for frame in reviewed_frames:
        split = frame.get("split") or "train"
        if split not in counts:
            split = "train"

        image_name = f"{frame['frame_id']}.jpg"
        label_name = f"{frame['frame_id']}.txt"
        src_image = os.path.join(session_dir, frame["image_filename"])
        dst_image = os.path.join(export_dir, "images", split, image_name)
        dst_label = os.path.join(export_dir, "labels", split, label_name)
        shutil.copy2(src_image, dst_image)

        label = frame.get("label")
        with open(dst_label, "w", encoding="utf-8") as label_file:
            if label:
                cx, cy, width, height = _bbox_to_yolo(
                    label["bbox"], frame["width"], frame["height"]
                )
                label_file.write(
                    f"{CLASS_ID} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}\n"
                )
                counts[split]["positive"] += 1
            else:
                counts[split]["negative"] += 1
        counts[split]["images"] += 1

    data_yaml_path = os.path.join(export_dir, "data.yaml")
    with open(data_yaml_path, "w", encoding="utf-8") as yaml_file:
        yaml_file.write(
            "path: "
            + export_dir
            + "\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: "
            + CLASS_NAME
            + "\n"
        )

    export_manifest = {
        "export_id": export_id,
        "session_id": session_id,
        "created_at": _now(),
        "dataset_dir": export_dir,
        "data_yaml": data_yaml_path,
        "counts": counts,
        "class_id": CLASS_ID,
        "class_name": CLASS_NAME,
        "split_policy": session["sampling"].get("split_policy"),
    }
    with open(os.path.join(export_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(export_manifest, f, indent=2)

    session.setdefault("exports", []).append(export_manifest)
    session["updated_at"] = _now()
    _save_session(session)
    return export_manifest


def evaluate_rjtpp_baseline(
    session_id: str, iou_threshold: float = 0.30
) -> dict[str, Any]:
    """Compare stored RJTPP predictions against reviewed human labels."""
    session = _load_session(session_id)
    reviewed_frames = [
        frame for frame in session["frames"] if frame.get("status") in REVIEWED_STATUSES
    ]
    if not reviewed_frames:
        raise ValueError("No reviewed frames are available for evaluation")

    tp = fp = fn = tn = visible_frames = 0
    for frame in reviewed_frames:
        label = frame.get("label")
        prediction = frame.get("prediction")
        predicted_bbox = prediction.get("bbox") if prediction else None
        if label:
            visible_frames += 1
            if predicted_bbox is None:
                fn += 1
                continue
            iou = _bbox_iou(label["bbox"], predicted_bbox)
            if iou >= iou_threshold:
                tp += 1
            else:
                fp += 1
                fn += 1
        elif predicted_bbox is None:
            tn += 1
        else:
            fp += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    result = {
        "session_id": session_id,
        "model": "RJTPP/tennis-ball-detection",
        "iou_threshold": float(iou_threshold),
        "reviewed_frames": len(reviewed_frames),
        "visible_frames": visible_frames,
        "detected_visible_frames": tp,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "evaluated_at": _now(),
    }
    session.setdefault("evaluations", []).append(result)
    session["updated_at"] = _now()
    _save_session(session)
    return result


def check_training_environment() -> dict[str, Any]:
    """Return local training dependency and device readiness information."""
    checks: dict[str, Any] = {}

    try:
        import torch

        checks["torch"] = {
            "available": True,
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "rocm_version": getattr(torch.version, "hip", None),
        }
    except Exception as exc:
        checks["torch"] = {"available": False, "error": str(exc)}

    try:
        import ultralytics

        checks["ultralytics"] = {
            "available": True,
            "version": getattr(ultralytics, "__version__", None),
        }
    except Exception as exc:
        checks["ultralytics"] = {"available": False, "error": str(exc)}

    try:
        import huggingface_hub

        checks["huggingface_hub"] = {
            "available": True,
            "version": getattr(huggingface_hub, "__version__", None),
        }
    except Exception as exc:
        checks["huggingface_hub"] = {"available": False, "error": str(exc)}

    torch_check = checks.get("torch", {})
    ultralytics_check = checks.get("ultralytics", {})
    if torch_check.get("cuda_available"):
        status = "rocm-ready" if torch_check.get("rocm_version") else "cuda-ready"
        device = "cuda"
    elif torch_check.get("available") and ultralytics_check.get("available"):
        status = "cpu-fallback"
        device = "cpu"
    else:
        status = "missing-dependencies"
        device = None

    return {
        "status": status,
        "recommended_device": device,
        "checks": checks,
        "notes": [
            "The current project shell may install CPU PyTorch wheels.",
            "ROCm readiness requires torch.cuda.is_available() with a ROCm build.",
        ],
    }


def _extract_session_frames(
    video_path: str,
    frames_dir: str,
    *,
    session_id: str,
    source_filename: str,
    source_video: str,
    max_frames: int,
    frame_step: int | None,
    prelabel: bool,
    confidence: float,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {source_filename}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if frame_count <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Video metadata is incomplete")

    frame_numbers = _sample_frame_numbers(frame_count, max_frames, frame_step)
    model = None
    prelabel_error = None
    if prelabel:
        model, prelabel_error = _get_rjtpp_model()

    frames: list[dict[str, Any]] = []
    prediction_error = None
    for index, frame_number in enumerate(frame_numbers):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = cap.read()
        if not ok:
            continue

        actual_height, actual_width = frame.shape[:2]
        frame_id = f"frame-{frame_number:06d}"
        image_name = f"frame_{frame_number:06d}.jpg"
        image_path = os.path.join(frames_dir, image_name)
        if not cv2.imwrite(image_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
            raise ValueError(f"Failed to write frame image: {image_name}")

        prediction = None
        if model is not None:
            try:
                prediction = _predict_rjtpp(model, frame, confidence)
            except Exception as exc:
                prediction_error = str(exc)
                model = None

        frames.append(
            {
                "frame_id": frame_id,
                "frame_number": int(frame_number),
                "time_sec": float(frame_number / fps) if fps > 0 else 0.0,
                "image_filename": os.path.join("frames", image_name),
                "width": int(actual_width),
                "height": int(actual_height),
                "split": _assign_split(index, len(frame_numbers)),
                "prediction": prediction,
                "label": None,
                "status": "pending",
            }
        )

    cap.release()
    if not frames:
        raise ValueError("No frames could be extracted from the video")

    now = _now()
    return {
        "id": session_id,
        "source_filename": source_filename,
        "source_video": source_video,
        "created_at": now,
        "updated_at": now,
        "classes": [{"id": CLASS_ID, "name": CLASS_NAME}],
        "video": {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration_sec": float(frame_count / fps) if fps > 0 else 0.0,
        },
        "sampling": {
            "max_frames": max_frames,
            "frame_step": frame_step,
            "sampled_frames": len(frames),
            "split_policy": "contiguous-frame-order-70-15-15",
            "negative_policy": "human-marked absent frames export as empty YOLO labels",
        },
        "prelabel": {
            "requested": prelabel,
            "model": RJTPP_REPO_ID if prelabel else None,
            "confidence": confidence,
            "error": prelabel_error or prediction_error,
        },
        "frames": frames,
        "progress": {},
        "exports": [],
        "evaluations": [],
    }


def _get_rjtpp_model() -> tuple[Any | None, str | None]:
    global _RJTPP_MODEL, _RJTPP_MODEL_ERROR
    if _RJTPP_MODEL is not None or _RJTPP_MODEL_ERROR is not None:
        return _RJTPP_MODEL, _RJTPP_MODEL_ERROR

    try:
        from ultralytics import YOLO

        model_path = os.environ.get("SERVE_ANALYZER_RJTPP_MODEL_PATH")
        if model_path:
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"RJTPP model file not found: {model_path}")
        elif os.environ.get("SERVE_ANALYZER_ALLOW_REMOTE_MODEL_DOWNLOAD") == "1":
            from huggingface_hub import hf_hub_download

            model_path = hf_hub_download(
                repo_id=RJTPP_REPO_ID,
                filename=RJTPP_FILENAME,
                revision=os.environ.get("SERVE_ANALYZER_RJTPP_REVISION", "main"),
            )
        else:
            raise RuntimeError(
                "RJTPP prelabeling requires SERVE_ANALYZER_RJTPP_MODEL_PATH or "
                "SERVE_ANALYZER_ALLOW_REMOTE_MODEL_DOWNLOAD=1"
            )
        _RJTPP_MODEL = YOLO(model_path)
    except Exception as exc:
        _RJTPP_MODEL_ERROR = str(exc)
    return _RJTPP_MODEL, _RJTPP_MODEL_ERROR


def _predict_rjtpp(model: Any, frame: Any, confidence: float) -> dict[str, Any] | None:
    height, width = frame.shape[:2]
    results = model.predict(source=frame, conf=confidence, verbose=False, device="cpu")
    if not results or results[0].boxes is None:
        return None

    boxes = results[0].boxes
    best: dict[str, Any] | None = None
    best_area = float("inf")
    for index in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[index].cpu().numpy()
        conf = float(boxes.conf[index].cpu().numpy()) if boxes.conf is not None else 0.0
        box_width = float(x2 - x1)
        box_height = float(y2 - y1)
        area = box_width * box_height
        if area <= 0 or area >= width * height * 0.01 or area >= best_area:
            continue
        best_area = area
        best = {
            "bbox": _clamp_bbox(
                {
                    "x": float(x1),
                    "y": float(y1),
                    "width": box_width,
                    "height": box_height,
                },
                width,
                height,
            ),
            "confidence": conf,
            "model": RJTPP_REPO_ID,
        }
    return best


def _sample_frame_numbers(
    frame_count: int, max_frames: int, frame_step: int | None
) -> list[int]:
    if frame_step is None:
        frame_step = max(1, frame_count // max_frames)
    frame_numbers = list(range(0, frame_count, frame_step))
    return frame_numbers[:max_frames]


def _assign_split(index: int, total: int) -> str:
    if total < 3:
        return "train"
    position = (index + 0.5) / total
    if position < 0.70:
        return "train"
    if position < 0.85:
        return "val"
    return "test"


def _progress_for(session: dict[str, Any]) -> dict[str, int]:
    counts = {
        "total": len(session.get("frames", [])),
        "pending": 0,
        "accepted": 0,
        "corrected": 0,
        "absent": 0,
        "skipped": 0,
        "reviewed": 0,
        "exportable": 0,
    }
    for frame in session.get("frames", []):
        status = frame.get("status", "pending")
        if status not in counts:
            status = "pending"
        counts[status] += 1
        if status != "pending":
            counts["reviewed"] += 1
        if status in REVIEWED_STATUSES:
            counts["exportable"] += 1
    return counts


def _session_summary(session: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": session["id"],
        "source_filename": session["source_filename"],
        "created_at": session["created_at"],
        "updated_at": session["updated_at"],
        "video": session["video"],
        "sampling": session["sampling"],
        "prelabel": session["prelabel"],
        "progress": _progress_for(session),
    }


def _load_session(session_id: str) -> dict[str, Any]:
    manifest_path = os.path.join(_safe_session_dir(session_id), "session.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Annotation session not found: {session_id}")
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def _save_session(session: dict[str, Any]) -> None:
    session["progress"] = _progress_for(session)
    session_dir = _safe_session_dir(session["id"], create=True)
    manifest_path = os.path.join(session_dir, "session.json")
    tmp_path = manifest_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(session, f, indent=2)
    os.replace(tmp_path, manifest_path)


def _safe_session_dir(session_id: str, *, create: bool = False) -> str:
    if not SESSION_ID_RE.match(session_id):
        raise ValueError("Invalid annotation session id")
    if create:
        return get_annotation_session_dir(session_id)
    return os.path.join(get_annotation_root(), session_id)


def _find_frame(session: dict[str, Any], frame_id: str) -> dict[str, Any]:
    for frame in session.get("frames", []):
        if frame.get("frame_id") == frame_id:
            return frame
    raise FileNotFoundError(f"Frame not found: {frame_id}")


def _label_from_bbox(bbox: dict[str, float], *, source: str) -> dict[str, Any]:
    return {
        "class_id": CLASS_ID,
        "class_name": CLASS_NAME,
        "bbox": bbox,
        "source": source,
    }


def _clamp_bbox(
    bbox: dict[str, float], image_width: int, image_height: int
) -> dict[str, float]:
    x = float(bbox.get("x", 0.0))
    y = float(bbox.get("y", 0.0))
    width = float(bbox.get("width", 0.0))
    height = float(bbox.get("height", 0.0))
    if width <= 0 or height <= 0:
        raise ValueError("Bounding box width and height must be positive")

    x = max(0.0, min(x, float(image_width - 1)))
    y = max(0.0, min(y, float(image_height - 1)))
    width = min(width, float(image_width) - x)
    height = min(height, float(image_height) - y)
    if width <= 0 or height <= 0:
        raise ValueError("Bounding box is outside the image")
    return {"x": x, "y": y, "width": width, "height": height}


def _bbox_to_yolo(
    bbox: dict[str, float], image_width: int, image_height: int
) -> tuple[float, float, float, float]:
    clamped = _clamp_bbox(bbox, image_width, image_height)
    cx = (clamped["x"] + clamped["width"] / 2.0) / image_width
    cy = (clamped["y"] + clamped["height"] / 2.0) / image_height
    width = clamped["width"] / image_width
    height = clamped["height"] / image_height
    return cx, cy, width, height


def _bbox_iou(first: dict[str, float], second: dict[str, float]) -> float:
    first_x2 = first["x"] + first["width"]
    first_y2 = first["y"] + first["height"]
    second_x2 = second["x"] + second["width"]
    second_y2 = second["y"] + second["height"]

    inter_x1 = max(first["x"], second["x"])
    inter_y1 = max(first["y"], second["y"])
    inter_x2 = min(first_x2, second_x2)
    inter_y2 = min(first_y2, second_y2)
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height
    first_area = first["width"] * first["height"]
    second_area = second["width"] * second["height"]
    union = first_area + second_area - inter_area
    return inter_area / union if union > 0 else 0.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

"""Tests for tennis-ball annotation session review and YOLO export."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from web.backend.services import annotation_service


class TestAnnotationService(unittest.TestCase):
    """Verify annotation review, export, and baseline evaluation contracts."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.env_patch = patch.dict(
            os.environ,
            {"SERVE_ANALYZER_ANNOTATION_DIR": self.temp_dir.name},
        )
        self.env_patch.start()

    def tearDown(self) -> None:
        self.env_patch.stop()
        self.temp_dir.cleanup()

    def test_review_actions_update_labels_and_progress(self) -> None:
        """Human review actions persist labels and recompute progress."""
        session_id = self._write_session()

        accepted = annotation_service.review_frame(
            session_id,
            "frame-000000",
            "accept",
        )
        first = accepted["frames"][0]
        self.assertEqual(first["status"], "accepted")
        self.assertEqual(first["label"]["source"], "prediction")

        corrected = annotation_service.review_frame(
            session_id,
            "frame-000001",
            "correct",
            {"x": 95, "y": 96, "width": 20, "height": 20},
        )
        second = corrected["frames"][1]
        self.assertEqual(second["status"], "corrected")
        self.assertEqual(second["label"]["source"], "manual")
        self.assertEqual(
            second["label"]["bbox"], {"x": 95.0, "y": 96.0, "width": 5.0, "height": 4.0}
        )
        self.assertEqual(corrected["progress"]["reviewed"], 2)
        self.assertEqual(corrected["progress"]["exportable"], 2)

        undone = annotation_service.undo_frame_review(session_id, "frame-000001")
        self.assertEqual(undone["frames"][1]["status"], "pending")
        self.assertIsNone(undone["frames"][1]["label"])
        self.assertEqual(undone["progress"]["reviewed"], 1)

    def test_export_writes_positive_and_empty_negative_labels(self) -> None:
        """YOLO export writes normalized positive labels and empty negatives."""
        session_id = self._write_session()
        annotation_service.review_frame(session_id, "frame-000000", "accept")
        annotation_service.review_frame(session_id, "frame-000001", "absent")

        export = annotation_service.export_yolo_dataset(session_id)

        self.assertTrue(os.path.isfile(export["data_yaml"]))
        self.assertEqual(
            export["counts"]["train"], {"images": 1, "positive": 1, "negative": 0}
        )
        self.assertEqual(
            export["counts"]["val"], {"images": 1, "positive": 0, "negative": 1}
        )

        dataset_dir = export["dataset_dir"]
        positive_label = os.path.join(
            dataset_dir, "labels", "train", "frame-000000.txt"
        )
        negative_label = os.path.join(dataset_dir, "labels", "val", "frame-000001.txt")
        with open(positive_label, encoding="utf-8") as f:
            self.assertEqual(f.read(), "0 0.200000 0.250000 0.200000 0.100000\n")
        with open(negative_label, encoding="utf-8") as f:
            self.assertEqual(f.read(), "")

    def test_baseline_evaluation_counts_reviewed_outcomes(self) -> None:
        """RJTPP baseline evaluation reports precision and recall from reviewed labels."""
        session_id = self._write_session()
        annotation_service.review_frame(session_id, "frame-000000", "accept")
        annotation_service.review_frame(session_id, "frame-000001", "absent")
        annotation_service.review_frame(
            session_id,
            "frame-000002",
            "correct",
            {"x": 40, "y": 40, "width": 10, "height": 10},
        )

        evaluation = annotation_service.evaluate_rjtpp_baseline(session_id)

        self.assertEqual(evaluation["reviewed_frames"], 3)
        self.assertEqual(evaluation["visible_frames"], 2)
        self.assertEqual(evaluation["true_positive"], 1)
        self.assertEqual(evaluation["false_positive"], 1)
        self.assertEqual(evaluation["false_negative"], 1)
        self.assertEqual(evaluation["true_negative"], 0)
        self.assertEqual(evaluation["precision"], 0.5)
        self.assertEqual(evaluation["recall"], 0.5)

    def _write_session(self) -> str:
        session_id = "test-session"
        session_dir = os.path.join(self.temp_dir.name, session_id)
        frames_dir = os.path.join(session_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        for frame_number in range(3):
            image_path = os.path.join(frames_dir, f"frame_{frame_number:06d}.jpg")
            with open(image_path, "wb") as f:
                f.write(b"fake jpeg")

        now = "2026-04-27T00:00:00+00:00"
        session = {
            "id": session_id,
            "source_filename": "serve.mp4",
            "source_video": "source.mp4",
            "created_at": now,
            "updated_at": now,
            "classes": [{"id": 0, "name": "tennis_ball"}],
            "video": {
                "width": 100,
                "height": 100,
                "fps": 60.0,
                "frame_count": 3,
                "duration_sec": 0.05,
            },
            "sampling": {
                "split_policy": "test",
                "negative_policy": "human-marked absent frames export as empty YOLO labels",
            },
            "prelabel": {
                "requested": True,
                "model": "RJTPP/tennis-ball-detection",
                "confidence": 0.2,
                "error": None,
            },
            "frames": [
                self._frame(
                    "frame-000000",
                    "frames/frame_000000.jpg",
                    "train",
                    {"x": 10, "y": 20, "width": 20, "height": 10},
                ),
                self._frame(
                    "frame-000001",
                    "frames/frame_000001.jpg",
                    "val",
                    {"x": 70, "y": 70, "width": 10, "height": 10},
                ),
                self._frame("frame-000002", "frames/frame_000002.jpg", "test", None),
            ],
            "progress": {},
            "exports": [],
            "evaluations": [],
        }
        with open(
            os.path.join(session_dir, "session.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(session, f)
        return session_id

    def _frame(
        self,
        frame_id: str,
        image_filename: str,
        split: str,
        prediction_bbox: dict[str, float] | None,
    ) -> dict[str, object]:
        prediction = None
        if prediction_bbox is not None:
            prediction = {
                "bbox": prediction_bbox,
                "confidence": 0.9,
                "model": "RJTPP/tennis-ball-detection",
            }
        return {
            "frame_id": frame_id,
            "frame_number": int(frame_id.rsplit("-", 1)[1]),
            "time_sec": 0.0,
            "image_filename": image_filename,
            "width": 100,
            "height": 100,
            "split": split,
            "prediction": prediction,
            "label": None,
            "status": "pending",
        }


if __name__ == "__main__":
    unittest.main()

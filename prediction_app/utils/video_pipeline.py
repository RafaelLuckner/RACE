"""End-to-end Random Forest video prediction pipeline."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

from .angle_utils import extract_angles_from_frame
from .constants import (
    ANGLE_COLUMNS,
    TRAINING_WINDOW_SIZE,
    DEFAULT_MIN_POSE_DETECTION_CONFIDENCE,
    DEFAULT_MIN_POSE_PRESENCE_CONFIDENCE,
    DEFAULT_OUTPUT_DIR,
)
from .feature_utils import build_frames_dataframe, create_temporal_features_window
from .model_utils import build_feature_columns
from .pose_utils import (
    PoseLandmarkerDetector,
    draw_landmarks_on_frame,
    draw_angles_on_frame,
    has_valid_landmarks,
)


def _safe_probability_column(class_name: str) -> str:
    return "prob_" + class_name.lower().replace(" ", "_")


class RandomForestVideoPredictor:
    """Processes a video, builds window dataset and exports annotated prediction video."""

    def __init__(
        self,
        model,
        scaler,
        class_name_to_id: Dict[str, int],
        class_id_to_name: Dict[int, str],
        pose_detector: PoseLandmarkerDetector,
        window_size: int,
        process_fps: int,
        angle_columns: List[str] | None = None,
        min_pose_detection_confidence: float = DEFAULT_MIN_POSE_DETECTION_CONFIDENCE,
        min_pose_presence_confidence: float = DEFAULT_MIN_POSE_PRESENCE_CONFIDENCE,
        max_seconds: int | None = None,
        progress_callback = None,
        visualization_options: List[str] | None = None,
    ) -> None:
        self.model = model
        self.scaler = scaler
        self.class_name_to_id = class_name_to_id
        self.class_id_to_name = class_id_to_name
        self.pose_detector = pose_detector
        self.window_size = int(window_size)
        self.process_fps = int(process_fps)
        self.angle_columns = angle_columns or list(ANGLE_COLUMNS)
        self.min_pose_detection_confidence = float(min_pose_detection_confidence)
        self.min_pose_presence_confidence = float(min_pose_presence_confidence)
        self.max_seconds = max_seconds
        self.progress_callback = progress_callback
        
        # Opções de visualização: ["classification", "angles", "landmarks"]
        self.visualization_options = visualization_options or ["classification", "angles", "landmarks"]

        self.model_classes = [int(c) for c in getattr(model, "classes_", sorted(class_id_to_name.keys()))]
        self.expected_feature_columns = build_feature_columns(self.window_size, self.angle_columns)

    def _extract_frame_records(
        self,
        video_path: Path,
    ) -> Tuple[pd.DataFrame, Dict[int, Dict], Dict[str, float]]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        interval = max(1, int(round(fps / max(self.process_fps, 1))))

        # Limitar frames se max_seconds foi definido
        max_frames = total_frames
        if self.max_seconds is not None:
            max_frames = min(total_frames, int(self.max_seconds * fps))

        frame_records: List[Dict] = []
        frame_lookup: Dict[int, Dict] = {}

        frame_idx = 0
        processed_count = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame_idx >= max_frames:
                break

            if frame_idx % interval != 0:
                frame_idx += 1
                continue

            # Chamar callback de progresso
            if self.progress_callback:
                self.progress_callback(processed_count, max_frames // interval, "Detectando pose")

            timestamp_s = frame_idx / max(fps, 1.0)
            timestamp_ms = int(timestamp_s * 1000)

            landmarks, visibility, presence = self.pose_detector.detect_for_video(frame, timestamp_ms)
            has_landmarks = has_valid_landmarks(
                landmarks,
                visibility,
                presence,
                min_detection_confidence=self.min_pose_detection_confidence,
                min_presence_confidence=self.min_pose_presence_confidence,
                min_points=8,
            )

            if landmarks:
                angles = extract_angles_from_frame(
                    landmarks,
                    visibility,
                    presence,
                    min_detection_confidence=self.min_pose_detection_confidence,
                    min_presence_confidence=self.min_pose_presence_confidence,
                )
            else:
                angles = {angle: np.nan for angle in self.angle_columns}

            frame_record = {
                "frame": frame_idx,
                "timestamp_s": float(timestamp_s),
                "has_landmarks": bool(has_landmarks),
                "landmarks": landmarks if has_landmarks else [],
                "visibility": visibility if has_landmarks else [],
                "presence": presence if has_landmarks else [],
                **angles,
            }
            frame_records.append(frame_record)
            frame_lookup[frame_idx] = frame_record

            frame_idx += 1
            processed_count += 1

        cap.release()

        frame_df = build_frames_dataframe(frame_records, self.angle_columns)
        video_info = {
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "process_fps": self.process_fps,
            "interval": interval,
        }
        return frame_df, frame_lookup, video_info

    def _predict_windows(
        self,
        frame_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X, metadata = create_temporal_features_window(
            frame_df,
            window_size=self.window_size,
            angle_columns=self.angle_columns,
            min_landmark_frames_in_window=1,
        )

        if X.empty or metadata.empty:
            raise ValueError(
                "No valid temporal windows were created. "
                "Try reducing process_fps threshold constraints or check landmark detection quality."
            )

        # Ensure all expected columns exist and in correct order
        for col in self.expected_feature_columns:
            if col not in X.columns:
                X[col] = 0.0
        
        # Reorder columns to match training order
        X = X[self.expected_feature_columns]
        
        # Fill any remaining NaN values with 0
        X = X.fillna(0.0)
        
        # DEBUG: Validate column count and names
        if X.shape[1] != len(self.expected_feature_columns):
            raise ValueError(
                f"Feature count mismatch: expected {len(self.expected_feature_columns)} "
                f"(window_size={self.window_size} × angles={len(self.angle_columns)}), "
                f"got {X.shape[1]}"
            )
        
        # Convert to numpy array for scaler (avoids pandas index/column name issues)
        X_array = X.values.astype(np.float64)
        X_scaled = self.scaler.transform(X_array)
        probabilities = self.model.predict_proba(X_scaled)
        max_indices = np.argmax(probabilities, axis=1)
        pred_ids = [int(self.model_classes[idx]) for idx in max_indices]

        predictions_df = metadata.copy()
        predictions_df["pred_label_id"] = pred_ids
        predictions_df["pred_label_name"] = [self.class_id_to_name.get(pid, str(pid)) for pid in pred_ids]
        predictions_df["confidence"] = probabilities.max(axis=1)

        for class_idx, class_id in enumerate(self.model_classes):
            class_name = self.class_id_to_name.get(class_id, str(class_id))
            prob_col = _safe_probability_column(class_name)
            predictions_df[prob_col] = probabilities[:, class_idx]

        final_dataset = pd.concat([predictions_df.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
        return final_dataset, predictions_df

    def _build_frame_prediction_map(self, predictions_df: pd.DataFrame) -> Dict[int, Dict]:
        frame_prediction_map: Dict[int, Dict] = {}

        for _, row in predictions_df.iterrows():
            end_frame = int(row["end_frame"])
            pred_id = int(row["pred_label_id"])
            pred_name = str(row["pred_label_name"])
            confidence = float(row["confidence"])

            probabilities = {}
            for class_id in self.model_classes:
                class_name = self.class_id_to_name.get(class_id, str(class_id))
                prob_col = _safe_probability_column(class_name)
                probabilities[class_name] = float(row.get(prob_col, 0.0))

            frame_prediction_map[end_frame] = {
                "pred_label_id": pred_id,
                "pred_label_name": pred_name,
                "confidence": confidence,
                "probabilities": probabilities,
            }

        return frame_prediction_map

    def _draw_prediction_overlay(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fps: float,
        current_prediction: Dict | None,
    ) -> np.ndarray:
        out = frame.copy()
        x0, y0 = 12, 12

        if current_prediction is None:
            cv2.rectangle(out, (x0, y0), (x0 + 460, y0 + 80), (0, 0, 0), -1)
            cv2.rectangle(out, (x0, y0), (x0 + 460, y0 + 80), (80, 80, 80), 2)
            cv2.putText(
                out,
                "Prediction: waiting for first complete window",
                (x0 + 10, y0 + 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        else:
            probs = current_prediction["probabilities"]
            panel_height = 85 + 24 * len(probs)
            cv2.rectangle(out, (x0, y0), (x0 + 460, y0 + panel_height), (0, 0, 0), -1)
            cv2.rectangle(out, (x0, y0), (x0 + 460, y0 + panel_height), (0, 200, 255), 2)

            title = (
                f"Prediction: {current_prediction['pred_label_name']} "
                f"({current_prediction['confidence']:.2%})"
            )
            cv2.putText(
                out,
                title,
                (x0 + 10, y0 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            for idx, (class_name, prob) in enumerate(probs.items()):
                text = f"P({class_name}) = {prob:.2%}"
                cv2.putText(
                    out,
                    text,
                    (x0 + 10, y0 + 60 + idx * 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )

        timestamp_s = frame_idx / max(fps, 1.0)
        cv2.putText(
            out,
            f"Frame: {frame_idx}",
            (out.shape[1] - 220, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            f"Time: {timestamp_s:.2f}s",
            (out.shape[1] - 220, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return out

    def _render_output_video(
        self,
        video_path: Path,
        output_video_path: Path,
        frame_lookup: Dict[int, Dict],
        frame_prediction_map: Dict[int, Dict],
        video_info: Dict[str, float],
    ) -> pd.DataFrame:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video for rendering: {video_path}")

        # Use same codec strategy as streamlit_app for WhatsApp compatibility
        # Try multiple codecs in order of preference until one works
        fourcc_options = [
            ('H264', cv2.VideoWriter_fourcc(*'H264')),   # H.264
            ('avc1', cv2.VideoWriter_fourcc(*'avc1')),   # MPEG-4 Part 10
            ('XVID', cv2.VideoWriter_fourcc(*'XVID')),   # MPEG-4 Part 2 (XVID)
            ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),   # MPEG-4 Part 2
            ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),   # Motion JPEG (fallback)
        ]
        
        out = None
        successful_codec = None
        for codec_name, fourcc in fourcc_options:
            out = cv2.VideoWriter(
                str(output_video_path),
                fourcc,
                float(video_info["fps"]),
                (int(video_info["width"]), int(video_info["height"])),
            )
            if out and out.isOpened():
                successful_codec = codec_name
                break
            if out:
                out.release()
        
        if not out or not out.isOpened():
            raise RuntimeError(
                f"Could not create VideoWriter with any codec. "
                f"Tried: {', '.join([c[0] for c in fourcc_options])}"
            )

        frame_rows: List[Dict] = []
        current_prediction = None
        last_landmarks = None
        last_visibility = None
        last_presence = None

        # Contar total de frames para progresso
        max_frame_limit = None
        if self.max_seconds is not None:
            max_frame_limit = int(self.max_seconds * float(video_info["fps"]))

        frame_idx = 0
        rendered_count = 0
        while True:
            ret, frame = cap.read()
            if not ret or (max_frame_limit and frame_idx >= max_frame_limit):
                break

            # Chamar callback de progresso
            if self.progress_callback:
                total_frames_to_render = max_frame_limit or video_info["total_frames"]
                self.progress_callback(frame_idx, total_frames_to_render, "Renderizando vídeo")

            if frame_idx in frame_prediction_map:
                current_prediction = frame_prediction_map[frame_idx]

            # Desenhar landmarks em TODOS os frames (usando última pose detectada)
            processed = frame_lookup.get(frame_idx)
            landmarks_drawn = False
            
            if processed and processed.get("has_landmarks"):
                # Atualizar última pose conhecida
                last_landmarks = processed["landmarks"]
                last_visibility = processed["visibility"]
                last_presence = processed["presence"]
                landmarks_drawn = True
            elif last_landmarks is not None:
                # Usar última pose detectada em frames intermediários
                landmarks_drawn = True
            
            # Desenhar ângulos com cores dinâmicas
            edge_colors = {}
            if "angles" in self.visualization_options and landmarks_drawn and last_landmarks is not None:
                frame, edge_colors = draw_angles_on_frame(
                    frame,
                    last_landmarks,
                    last_visibility,
                    last_presence,
                    min_detection_confidence=self.min_pose_detection_confidence,
                    min_presence_confidence=self.min_pose_presence_confidence,
                )
            
            # Desenhar landmarks (com cores das arestas se ângulos estiverem visíveis)
            if "landmarks" in self.visualization_options and landmarks_drawn and last_landmarks is not None:
                frame = draw_landmarks_on_frame(
                    frame,
                    last_landmarks,
                    last_visibility,
                    last_presence,
                    min_detection_confidence=self.min_pose_detection_confidence,
                    min_presence_confidence=self.min_pose_presence_confidence,
                    edge_colors=edge_colors,
                )

            frame = self._draw_prediction_overlay(
                frame,
                frame_idx,
                fps=float(video_info["fps"]),
                current_prediction=current_prediction,
            ) if "classification" in self.visualization_options else frame
            out.write(frame)

            row = {
                "frame": frame_idx,
                "timestamp_s": frame_idx / max(float(video_info["fps"]), 1.0),
                "landmarks_drawn": landmarks_drawn,
            }
            if current_prediction is not None:
                row["pred_label_id"] = current_prediction["pred_label_id"]
                row["pred_label_name"] = current_prediction["pred_label_name"]
                row["confidence"] = current_prediction["confidence"]
                for class_name, prob in current_prediction["probabilities"].items():
                    row[_safe_probability_column(class_name)] = prob
            frame_rows.append(row)

            frame_idx += 1

        cap.release()
        out.release()

        return pd.DataFrame(frame_rows)

    def _build_summary(self, predictions_df: pd.DataFrame, video_info: Dict[str, float]) -> Dict:
        if predictions_df.empty:
            return {
                "final_prediction": None,
                "total_windows": 0,
                "class_counts": {},
                "mean_probabilities": {},
                "video_info": video_info,
            }

        class_counts = Counter(predictions_df["pred_label_name"].tolist())

        mean_probabilities = {}
        for class_id in self.model_classes:
            class_name = self.class_id_to_name.get(class_id, str(class_id))
            prob_col = _safe_probability_column(class_name)
            if prob_col in predictions_df.columns:
                mean_probabilities[class_name] = float(predictions_df[prob_col].mean())

        final_prediction = max(class_counts.items(), key=lambda item: item[1])[0]

        return {
            "final_prediction": final_prediction,
            "total_windows": int(len(predictions_df)),
            "class_counts": dict(class_counts),
            "mean_probabilities": mean_probabilities,
            "video_info": video_info,
        }

    def process_video(self, video_path: str, output_dir: str | Path | None = None) -> Dict:
        input_path = Path(video_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Video not found: {input_path}")

        base_output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
        run_dir = base_output_dir / f"{input_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)

        frame_df, frame_lookup, video_info = self._extract_frame_records(input_path)
        frame_export_df = frame_df[["frame", "timestamp_s", "has_landmarks", *self.angle_columns]].copy()
        frame_dataset_path = run_dir / f"{input_path.stem}_frames_dataset.csv"
        frame_export_df.to_csv(frame_dataset_path, index=False)

        windows_dataset_df, predictions_df = self._predict_windows(frame_df)
        windows_dataset_path = run_dir / f"{input_path.stem}_windows_dataset.csv"
        windows_dataset_df.to_csv(windows_dataset_path, index=False)

        frame_prediction_map = self._build_frame_prediction_map(predictions_df)

        output_video_path = run_dir / f"{input_path.stem}_predicted.mp4"
        frame_predictions_df = self._render_output_video(
            input_path,
            output_video_path,
            frame_lookup,
            frame_prediction_map,
            video_info,
        )
        frame_predictions_path = run_dir / f"{input_path.stem}_frame_predictions.csv"
        frame_predictions_df.to_csv(frame_predictions_path, index=False)

        summary = self._build_summary(predictions_df, video_info)
        summary_path = run_dir / f"{input_path.stem}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as file_obj:
            json.dump(summary, file_obj, indent=2, ensure_ascii=False)

        return {
            "run_dir": run_dir,
            "output_video_path": output_video_path,
            "frame_dataset_path": frame_dataset_path,
            "windows_dataset_path": windows_dataset_path,
            "frame_predictions_path": frame_predictions_path,
            "summary_path": summary_path,
            "summary": summary,
            "video_info": video_info,
        }

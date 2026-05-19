"""End-to-end Random Forest video prediction pipeline."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

from .angle_utils import extract_angles_from_frame
from .rep_counter import RepetitionCounter, EXERCISE_REP_CONFIG
from .constants import (
    ANGLE_COLUMNS,
    VISIBILITY_COLUMNS,
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
        visibility_columns: List[str] | None = None,
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
        self.visibility_columns = visibility_columns or list(VISIBILITY_COLUMNS)
        self.min_pose_detection_confidence = float(min_pose_detection_confidence)
        self.min_pose_presence_confidence = float(min_pose_presence_confidence)
        self.max_seconds = max_seconds
        self.progress_callback = progress_callback
        
        # Opções de visualização: ["classification", "angles", "landmarks"]
        self.visualization_options = visualization_options or ["classification", "angles", "landmarks"]

        self.model_classes = [int(c) for c in getattr(model, "classes_", sorted(class_id_to_name.keys()))]
        # Expected features: window_size × n_angles = 15 × 8 = 120 (matching training notebook)
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
        last_timestamp_ms = -1
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
            # Garantir timestamp estritamente crescente (requisito do MediaPipe VIDEO mode)
            timestamp_ms = max(int(timestamp_s * 1000), last_timestamp_ms + 1)
            last_timestamp_ms = timestamp_ms

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

        frame_df = build_frames_dataframe(frame_records, self.angle_columns, self.visibility_columns)
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
            visibility_columns=self.visibility_columns,
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
                f"(window_size={self.window_size} × ({len(self.angle_columns)} angles + {len(self.visibility_columns)} visibilities)), "
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

    @staticmethod
    def _filled_rounded_rect(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                              color: tuple, alpha: float, radius: int = 8) -> None:
        """Draw a semi-transparent filled rounded rectangle in-place."""
        overlay = img.copy()
        r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
        cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in [(x1 + r, y1 + r), (x2 - r, y1 + r),
                       (x1 + r, y2 - r), (x2 - r, y2 - r)]:
            cv2.circle(overlay, (cx, cy), r, color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    def _draw_prediction_overlay(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fps: float,
        current_prediction: Dict | None,
        rep_counts: Dict[str, int] | None = None,
        current_exercise: str | None = None,
    ) -> np.ndarray:
        # Short display names for probability rows (must fit label column)
        _PLABELS = {
            "flexao":       "Flexao",
            "agachamento":  "Agach.",
            "rosca_biceps": "Rosca B.",
            "descanso":     "Descanso",
        }

        out = frame.copy()
        h, w = out.shape[:2]
        mg       = 10
        pad      = 10
        bg       = (15, 15, 15)
        alpha_bg = 0.80
        accent   = (0, 210, 255)
        row_h_p  = 22
        row_h_r  = 30
        font     = cv2.FONT_HERSHEY_SIMPLEX

        # ── Left panel: Prediction + probability bars ─────────────────────
        if current_prediction is not None:
            probs    = current_prediction["probabilities"]
            pred_pw  = 235
            label_w  = 68    # reserved for label text left of bar
            pct_w    = 28    # reserved for "xx%" right of bar
            bar_max_w = pred_pw - 2 * pad - label_w - pct_w
            title_h  = 32
            prob_h   = len(probs) * row_h_p
            ph       = pad + title_h + 6 + prob_h + pad

            px1, py1 = mg, mg
            px2, py2 = px1 + pred_pw, py1 + ph

            self._filled_rounded_rect(out, px1, py1, px2, py2, bg, alpha_bg)
            cv2.rectangle(out, (px1, py1), (px2, py1 + 3), accent, -1)

            ex_name = current_prediction["pred_label_name"].capitalize()
            conf    = f"  {current_prediction['confidence']:.0%}"
            cv2.putText(out, ex_name,
                (px1 + pad, py1 + pad + title_h - 6),
                font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            (tw, _), _ = cv2.getTextSize(ex_name, font, 0.65, 2)
            cv2.putText(out, conf,
                (px1 + pad + tw, py1 + pad + title_h - 6),
                font, 0.50, (160, 160, 160), 1, cv2.LINE_AA)

            sep_y = py1 + pad + title_h + 2
            cv2.line(out, (px1 + pad, sep_y), (px2 - pad, sep_y), (55, 55, 55), 1)

            for idx, (class_name, prob) in enumerate(probs.items()):
                row_y  = sep_y + 5 + idx * row_h_p
                is_act = (class_name == current_prediction["pred_label_name"])
                tc     = (255, 255, 255) if is_act else (130, 130, 130)
                disp   = _PLABELS.get(class_name, class_name.capitalize())
                cv2.putText(out, disp,
                    (px1 + pad, row_y + 14),
                    font, 0.40, tc, 1, cv2.LINE_AA)

                bar_x = px1 + pad + label_w
                bar_y = row_y + 5
                bar_h = 9
                cv2.rectangle(out, (bar_x, bar_y),
                    (bar_x + bar_max_w, bar_y + bar_h), (40, 40, 40), -1)
                filled_w = max(2, int(prob * bar_max_w))
                cv2.rectangle(out, (bar_x, bar_y),
                    (bar_x + filled_w, bar_y + bar_h),
                    accent if is_act else (75, 75, 75), -1)
                cv2.putText(out, f"{prob:.0%}",
                    (bar_x + bar_max_w + 4, row_y + 14),
                    font, 0.38, tc, 1, cv2.LINE_AA)

        # ── Right panel: Rep counter ──────────────────────────────────────
        tracked = {ex: cnt for ex, cnt in (rep_counts or {}).items()
                   if ex in EXERCISE_REP_CONFIG}
        if tracked:
            rep_pw   = 195
            header_h = 28
            rh = pad + header_h + 4 + len(tracked) * row_h_r + pad
            rx1 = w - rep_pw - mg
            ry1 = mg
            rx2 = w - mg
            ry2 = ry1 + rh

            self._filled_rounded_rect(out, rx1, ry1, rx2, ry2, bg, alpha_bg)
            cv2.rectangle(out, (rx1, ry1), (rx2, ry1 + 3), accent, -1)

            cv2.putText(out, "REPS",
                (rx1 + pad, ry1 + pad + header_h - 7),
                font, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

            sep_y2 = ry1 + pad + header_h
            cv2.line(out, (rx1 + pad, sep_y2), (rx2 - pad, sep_y2), (55, 55, 55), 1)

            for i, (ex, cnt) in enumerate(tracked.items()):
                _, color = self._EX_DISPLAY.get(ex, (ex, (180, 180, 180)))
                label, _ = self._EX_DISPLAY.get(
                    ex, (ex.replace("_", " ").title(), None))
                row_y  = sep_y2 + 5 + i * row_h_r
                is_cur = (ex == current_exercise)
                mid_y  = row_y + row_h_r // 2 - 1

                if is_cur:
                    hl = out.copy()
                    cv2.rectangle(hl,
                        (rx1 + 3, row_y + 1), (rx2 - 3, row_y + row_h_r - 2),
                        (40, 40, 40), -1)
                    cv2.addWeighted(hl, 0.85, out, 0.15, 0, out)

                cv2.circle(out, (rx1 + pad + 5, mid_y), 5, color, -1, cv2.LINE_AA)

                lbl_color = (255, 255, 255) if is_cur else (160, 160, 160)
                lbl_thick = 2 if is_cur else 1
                cv2.putText(out, label,
                    (rx1 + pad + 16, mid_y + 5),
                    font, 0.46, lbl_color, lbl_thick, cv2.LINE_AA)

                count_str = str(cnt)
                (tw, _), _ = cv2.getTextSize(count_str, font, 0.62, 2)
                cnt_color = accent if (is_cur and cnt > 0) else (
                    (220, 220, 220) if cnt > 0 else (70, 70, 70))
                cv2.putText(out, count_str,
                    (rx2 - tw - pad, mid_y + 5),
                    font, 0.62, cnt_color, 2, cv2.LINE_AA)

        return out

    # ------------------------------------------------------------------ #
    #  Rep-count overlay (kept for reference — no longer called)          #
    # ------------------------------------------------------------------ #

    # Exercise display config: (label, BGR color)
    _EX_DISPLAY = {
        "flexao":       ("Flexão",      (  0, 230,  80)),
        "agachamento":  ("Agachamento", ( 60, 180, 255)),
        "rosca_biceps": ("Rosca Bíceps", (0, 200, 255)),
    }

    def _draw_rep_count_overlay(
        self,
        frame: np.ndarray,
        rep_counts: Dict[str, int],
        current_exercise: str | None,
    ) -> np.ndarray:
        """
        Draw a compact rep-counter panel at the bottom-right of the frame.
        Only exercises that are being tracked (EXERCISE_REP_CONFIG) are shown.
        The currently predicted exercise row is highlighted.
        """
        tracked = {ex: cnt for ex, cnt in rep_counts.items()
                   if ex in EXERCISE_REP_CONFIG}
        if not tracked:
            return frame

        h, w = frame.shape[:2]
        n_rows = len(tracked)

        # Scale font/padding proportionally to frame width
        base_scale = max(0.45, min(0.70, w / 1280.0))
        row_h   = int(34 * base_scale + 10)
        pad_x   = 12
        pad_y   = 10
        title_h = int(30 * base_scale + 6)
        panel_w = int(220 * base_scale + 20)
        panel_h = title_h + n_rows * row_h + pad_y

        # Panel anchor: bottom-right with margin
        margin = 12
        px1 = w - panel_w - margin
        py1 = h - panel_h - margin
        px2 = w - margin
        py2 = h - margin

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (px1, py1), (px2, py2), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Border
        cv2.rectangle(frame, (px1, py1), (px2, py2), (100, 100, 100), 1)

        # Title
        cv2.putText(
            frame, "Reps",
            (px1 + pad_x, py1 + title_h - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            base_scale * 0.72,
            (255, 255, 255),
            2, cv2.LINE_AA,
        )

        # Separator line under title
        cv2.line(frame, (px1 + 4, py1 + title_h), (px2 - 4, py1 + title_h),
                 (80, 80, 80), 1)

        # One row per tracked exercise
        for i, (ex, cnt) in enumerate(tracked.items()):
            label, color = self._EX_DISPLAY.get(
                ex, (ex.replace("_", " ").title(), (200, 200, 200))
            )
            is_current = (ex == current_exercise)
            row_y = py1 + title_h + i * row_h

            # Highlight bar for current exercise
            if is_current:
                cv2.rectangle(
                    frame,
                    (px1 + 2, row_y + 2),
                    (px2 - 2, row_y + row_h - 2),
                    (40, 40, 40), -1,
                )

            # Colored dot
            dot_r = max(5, int(6 * base_scale))
            dot_x = px1 + pad_x + dot_r
            dot_y = row_y + row_h // 2
            cv2.circle(frame, (dot_x, dot_y), dot_r, color, -1, cv2.LINE_AA)

            # Exercise label
            label_color = (255, 255, 255)
            cv2.putText(
                frame, label,
                (dot_x + dot_r + 6, dot_y + int(base_scale * 10) + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                base_scale * 0.58,
                label_color,
                2, cv2.LINE_AA,
            )

            # Count (right-aligned, bold look via thickness=2)
            count_str = str(cnt)
            (tw, _), _ = cv2.getTextSize(
                count_str, cv2.FONT_HERSHEY_SIMPLEX,
                base_scale * 0.80, 2,
            )
            count_color = (255, 255, 255) if cnt > 0 else (160, 160, 160)
            cv2.putText(
                frame, count_str,
                (px2 - tw - pad_x, dot_y + int(base_scale * 10) + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                base_scale * 0.80,
                count_color,
                2, cv2.LINE_AA,
            )

        return frame

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

        # Live repetition counter (state machine — same logic as notebook 5)
        live_counter = RepetitionCounter()

        # Contar total de frames para progresso
        max_frame_limit = None
        if self.max_seconds is not None:
            max_frame_limit = min(
                int(self.max_seconds * float(video_info["fps"])),
                int(video_info["total_frames"]),
            )

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

            # Update live rep counter on processed frames
            current_exercise = (
                current_prediction["pred_label_name"] if current_prediction else None
            )
            processed = frame_lookup.get(frame_idx)
            if processed and current_exercise and current_exercise in EXERCISE_REP_CONFIG:
                joint = EXERCISE_REP_CONFIG[current_exercise]["joint"]
                r_col  = f"right_{joint}"
                l_col  = f"left_{joint}"
                rw_col = f"right_{joint}_visibility_weight"
                lw_col = f"left_{joint}_visibility_weight"
                r  = float(processed.get(r_col,  float("nan")))
                l  = float(processed.get(l_col,  float("nan")))
                wr_raw = processed.get(rw_col, 1.0)
                wl_raw = processed.get(lw_col, 1.0)
                wr = 0.0 if (wr_raw is None or np.isnan(float(wr_raw)) or np.isnan(r)) else float(np.clip(wr_raw, 0, 1))
                wl = 0.0 if (wl_raw is None or np.isnan(float(wl_raw)) or np.isnan(l)) else float(np.clip(wl_raw, 0, 1))
                r  = 0.0 if np.isnan(r) else r
                l  = 0.0 if np.isnan(l) else l
                total_w = wr + wl
                if total_w > 0:
                    bilateral = (r * wr + l * wl) / total_w
                    live_counter.update(current_exercise, bilateral)
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
                rep_counts=live_counter.counts,
                current_exercise=current_exercise,
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

        return pd.DataFrame(frame_rows), live_counter.counts

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
            "rep_counts": {},   # filled in process_video after counting
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
        frame_predictions_df, live_rep_counts = self._render_output_video(
            input_path,
            output_video_path,
            frame_lookup,
            frame_prediction_map,
            video_info,
        )
        frame_predictions_path = run_dir / f"{input_path.stem}_frame_predictions.csv"
        frame_predictions_df.to_csv(frame_predictions_path, index=False)

        # Calcular tempo por exercício a partir dos frames
        time_per_exercise: Dict[str, float] = {}
        if "pred_label_name" in frame_predictions_df.columns and "timestamp_s" in frame_predictions_df.columns:
            fps = float(video_info["fps"])
            frame_duration = 1.0 / max(fps, 1.0)
            labeled = frame_predictions_df.dropna(subset=["pred_label_name"])
            for ex, group in labeled.groupby("pred_label_name"):
                time_per_exercise[str(ex)] = round(len(group) * frame_duration, 1)

        # Re-encodar para H.264 para compatibilidade com browsers
        try:
            h264_path = output_video_path.with_name(output_video_path.stem + "_web.mp4")
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(output_video_path),
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-movflags", "+faststart",
                    "-an",
                    str(h264_path),
                ],
                check=True,
                capture_output=True,
            )
            output_video_path.unlink(missing_ok=True)
            output_video_path = h264_path
        except Exception:
            pass  # Mantém o vídeo original se ffmpeg falhar

        summary = self._build_summary(predictions_df, video_info)
        summary["rep_counts"] = live_rep_counts
        summary["time_per_exercise"] = time_per_exercise
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
            "rep_counts": live_rep_counts,
            "time_per_exercise": time_per_exercise,
            "video_info": video_info,
        }

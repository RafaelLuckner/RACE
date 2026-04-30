import numpy as np
import pandas as pd

def calculate_angle(p1, vertex, p2):
    if p1 is None or vertex is None or p2 is None:
        return None

    p1     = np.array(p1[:2],     dtype=np.float32)  # apenas x, y
    vertex = np.array(vertex[:2], dtype=np.float32)
    p2     = np.array(p2[:2],     dtype=np.float32)

    v1 = p1 - vertex
    v2 = p2 - vertex

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    dot_product = np.dot(v1, v2)
    cos_angle = dot_product / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def extract_angles_from_landmarks(df):
    """
    Extrai ângulos articulares dos dados de landmarks.
    IMPORTANTE: Processa cada exercício SEPARADAMENTE para evitar perda de dados.
    Inclui pesos de visibilidade para cada ângulo calculado.
    """
    landmarks = {
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_wrist': 15,
        'right_wrist': 16,
        'left_hip': 23,
        'right_hip': 24,
        'left_knee': 25,
        'right_knee': 26,
        'left_ankle': 27,
        'right_ankle': 28,
    }

    frames_data = []
    
    # ⭐ IMPORTANTE: Processar cada exercício separadamente
    for exercise in df['exercise'].unique():
        exercise_df = df[df['exercise'] == exercise]
        
        for frame_id in exercise_df['frame'].unique():
            frame_df = exercise_df[exercise_df['frame'] == frame_id]

            landmarks_coords = {}
            landmarks_visibility = {}
            
            for _, row in frame_df.iterrows():
                landmark_idx = int(row['landmark_idx'])
                for name, lm_idx in landmarks.items():
                    if landmark_idx == lm_idx:
                        landmarks_coords[name] = [row['x'], row['y']]
                        landmarks_visibility[name] = row['visibility']

            if len(landmarks_coords) < 8:
                continue

            angles = {
                'frame': frame_id,
                'timestamp_s': frame_df.iloc[0]['timestamp_s'],
                'exercise': frame_df.iloc[0]['exercise'],
            }

            # Cotovelo Direito: flexão/extensão (ombro → cotovelo → pulso)
            if all(k in landmarks_coords for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
                angles['right_cotovelo'] = calculate_angle(
                    landmarks_coords['right_shoulder'],
                    landmarks_coords['right_elbow'],
                    landmarks_coords['right_wrist'],
                )
                angles['right_cotovelo_visibility_weight'] = np.mean([
                    landmarks_visibility['right_shoulder'],
                    landmarks_visibility['right_elbow'],
                    landmarks_visibility['right_wrist']
                ])

            # Cotovelo Esquerdo: flexão/extensão (ombro → cotovelo → pulso)
            if all(k in landmarks_coords for k in ['left_shoulder', 'left_elbow', 'left_wrist']):
                angles['left_cotovelo'] = calculate_angle(
                    landmarks_coords['left_shoulder'],
                    landmarks_coords['left_elbow'],
                    landmarks_coords['left_wrist'],
                )
                angles['left_cotovelo_visibility_weight'] = np.mean([
                    landmarks_visibility['left_shoulder'],
                    landmarks_visibility['left_elbow'],
                    landmarks_visibility['left_wrist']
                ])

            # Ombro Direito: abução/adução (cotovelo → ombro → quadril)
            if all(k in landmarks_coords for k in ['right_elbow', 'right_shoulder', 'right_hip']):
                angles['right_ombro'] = calculate_angle(
                    landmarks_coords['right_elbow'],
                    landmarks_coords['right_shoulder'],
                    landmarks_coords['right_hip'],
                )
                angles['right_ombro_visibility_weight'] = np.mean([
                    landmarks_visibility['right_elbow'],
                    landmarks_visibility['right_shoulder'],
                    landmarks_visibility['right_hip']
                ])

            # Ombro Esquerdo: abução/adução (cotovelo → ombro → quadril)
            if all(k in landmarks_coords for k in ['left_elbow', 'left_shoulder', 'left_hip']):
                angles['left_ombro'] = calculate_angle(
                    landmarks_coords['left_elbow'],
                    landmarks_coords['left_shoulder'],
                    landmarks_coords['left_hip'],
                )
                angles['left_ombro_visibility_weight'] = np.mean([
                    landmarks_visibility['left_elbow'],
                    landmarks_visibility['left_shoulder'],
                    landmarks_visibility['left_hip']
                ])

            # Joelho Direito: flexão/extensão (quadril → joelho → tornozelo)
            if all(k in landmarks_coords for k in ['right_hip', 'right_knee', 'right_ankle']):
                angles['right_joelho'] = calculate_angle(
                    landmarks_coords['right_hip'],
                    landmarks_coords['right_knee'],
                    landmarks_coords['right_ankle'],
                )
                angles['right_joelho_visibility_weight'] = np.mean([
                    landmarks_visibility['right_hip'],
                    landmarks_visibility['right_knee'],
                    landmarks_visibility['right_ankle']
                ])

            # Joelho Esquerdo: flexão/extensão (quadril → joelho → tornozelo)
            if all(k in landmarks_coords for k in ['left_hip', 'left_knee', 'left_ankle']):
                angles['left_joelho'] = calculate_angle(
                    landmarks_coords['left_hip'],
                    landmarks_coords['left_knee'],
                    landmarks_coords['left_ankle'],
                )
                angles['left_joelho_visibility_weight'] = np.mean([
                    landmarks_visibility['left_hip'],
                    landmarks_visibility['left_knee'],
                    landmarks_visibility['left_ankle']
                ])

            # Quadril Direito: flexão/extensão (joelho → quadril → ombro)
            if all(k in landmarks_coords for k in ['right_knee', 'right_hip', 'right_shoulder']):
                angles['right_quadril'] = calculate_angle(
                    landmarks_coords['right_knee'],
                    landmarks_coords['right_hip'],
                    landmarks_coords['right_shoulder'],
                )
                angles['right_quadril_visibility_weight'] = np.mean([
                    landmarks_visibility['right_knee'],
                    landmarks_visibility['right_hip'],
                    landmarks_visibility['right_shoulder']
                ])

            # Quadril Esquerdo: flexão/extensão (joelho → quadril → ombro)
            if all(k in landmarks_coords for k in ['left_knee', 'left_hip', 'left_shoulder']):
                angles['left_quadril'] = calculate_angle(
                    landmarks_coords['left_knee'],
                    landmarks_coords['left_hip'],
                    landmarks_coords['left_shoulder'],
                )
                angles['left_quadril_visibility_weight'] = np.mean([
                    landmarks_visibility['left_knee'],
                    landmarks_visibility['left_hip'],
                    landmarks_visibility['left_shoulder']
                ])

            frames_data.append(angles)

    return pd.DataFrame(frames_data)

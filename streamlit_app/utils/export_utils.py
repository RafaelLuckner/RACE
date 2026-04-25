"""Utilitários para exportação de dados"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
from config import BODY_LANDMARKS


def export_landmarks_to_csv(frames_data, video_info, output_path, exercise=None):
    """
    Exporta dados dos landmarks para CSV
    
    Args:
        frames_data: lista com dados de cada frame
        video_info: informações do vídeo
        output_path: caminho do arquivo CSV
        exercise: nome do exercício para adicionar em todas as linhas
    """
    data = []
    model = video_info.get('model', 'unknown')
    
    for frame_data in frames_data:
        # Usar ALL landmarks, não apenas filtered_landmarks
        landmarks = frame_data.get('landmarks', [])
        visibility = frame_data.get('visibility', [])
        presence = frame_data.get('presence', [])
        
        for idx, landmark in enumerate(landmarks):
            # Verificar se passou pelos filtros
            vis = visibility[idx] if idx < len(visibility) else 0
            pres = presence[idx] if idx < len(presence) else 0
            min_det_conf = video_info.get('min_pose_detection_confidence', 0.2)
            min_pres_conf = video_info.get('min_pose_presence_confidence', 0.2)
            passed_filter = (vis >= min_det_conf and pres >= min_pres_conf)
            
            # Obter nome do landmark
            landmark_name = BODY_LANDMARKS.get(idx, f"landmark_{idx}")
            
            row = {
                'frame': frame_data['frame_idx'],
                'processed_frame': frame_data.get('processed_frame_idx', frame_data['frame_idx']),
                'timestamp_s': frame_data['timestamp'],
                'model': model,
                'processing_time_ms': frame_data.get('processing_time', 0) * 1000,
                'landmark_idx': idx,
                'landmark_name': landmark_name,
                'x': landmark['x'],
                'y': landmark['y'],
                'z': landmark['z'],
                'visibility': vis,
                'presence': pres,
                'passed_filter': passed_filter
            }
            if exercise is not None:
                row['exercise'] = exercise
            data.append(row)
    
    df = pd.DataFrame(data)
    df = df.sort_values(['frame', 'landmark_idx']).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    return df


def export_landmarks_to_json(frames_data, video_info, output_path):
    """
    Exporta dados dos landmarks para JSON
    
    Args:
        frames_data: lista com dados de cada frame
        video_info: informações do vídeo
        output_path: caminho do arquivo JSON
    """
    output_data = {
        'video_info': {
            'total_frames': video_info['total_frames'],
            'fps': video_info['original_fps'],
            'width': video_info['width'],
            'height': video_info['height'],
            'fps_process': video_info['fps_process'],
            'model': video_info.get('model', 'unknown'),
            'min_pose_detection_confidence': video_info.get('min_pose_detection_confidence', 0.2),
            'min_pose_presence_confidence': video_info.get('min_pose_presence_confidence', 0.2)
        },
        'frames': []
    }
    
    for frame_data in frames_data:
        frame_obj = {
            'frame_idx': frame_data['frame_idx'],
            'processed_frame_idx': frame_data['processed_frame_idx'],
            'timestamp': frame_data['timestamp'],
            'processing_time_ms': frame_data.get('processing_time', 0) * 1000,
            'landmarks': []
        }
        
        for landmark in frame_data['filtered_landmarks']:
            frame_obj['landmarks'].append({
                'index': landmark['index'],
                'name': landmark['name'],
                'x': landmark['landmark']['x'],
                'y': landmark['landmark']['y'],
                'z': landmark['landmark']['z'],
                'visibility': landmark['visibility'],
                'presence': landmark['presence']
            })
        
        output_data['frames'].append(frame_obj)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)


def export_summary_json(frames_data, video_info, output_path):
    """
    Exporta resumo estatístico dos landmarks para JSON
    
    Args:
        frames_data: lista com dados de cada frame
        video_info: informações do vídeo
        output_path: caminho do arquivo JSON
    """
    from config import BODY_LANDMARKS
    
    # Agregar dados por landmark
    landmark_stats = {}
    
    for frame_data in frames_data:
        for landmark in frame_data['filtered_landmarks']:
            idx = landmark['index']
            name = landmark['name']
            
            if idx not in landmark_stats:
                landmark_stats[idx] = {
                    'name': name,
                    'detections': 0,
                    'visibility_values': [],
                    'presence_values': [],
                    'x_values': [],
                    'y_values': [],
                    'z_values': []
                }
            
            landmark_stats[idx]['detections'] += 1
            landmark_stats[idx]['visibility_values'].append(landmark['visibility'])
            landmark_stats[idx]['presence_values'].append(landmark['presence'])
            landmark_stats[idx]['x_values'].append(landmark['landmark']['x'])
            landmark_stats[idx]['y_values'].append(landmark['landmark']['y'])
            landmark_stats[idx]['z_values'].append(landmark['landmark']['z'])
    
    # Calcular estatísticas
    output_data = {
        'video_info': {
            'total_frames': video_info['total_frames'],
            'fps': video_info['original_fps'],
            'processed_frames': len(frames_data)
        },
        'landmark_statistics': {}
    }
    
    for idx, stats in landmark_stats.items():
        output_data['landmark_statistics'][str(idx)] = {
            'name': stats['name'],
            'detections': stats['detections'],
            'detection_rate': stats['detections'] / len(frames_data),
            'visibility': {
                'mean': np.mean(stats['visibility_values']),
                'min': np.min(stats['visibility_values']),
                'max': np.max(stats['visibility_values']),
                'std': np.std(stats['visibility_values'])
            },
            'presence': {
                'mean': np.mean(stats['presence_values']),
                'min': np.min(stats['presence_values']),
                'max': np.max(stats['presence_values']),
                'std': np.std(stats['presence_values'])
            },
            'position': {
                'x': {
                    'mean': np.mean(stats['x_values']),
                    'min': np.min(stats['x_values']),
                    'max': np.max(stats['x_values'])
                },
                'y': {
                    'mean': np.mean(stats['y_values']),
                    'min': np.min(stats['y_values']),
                    'max': np.max(stats['y_values'])
                },
                'z': {
                    'mean': np.mean(stats['z_values']),
                    'min': np.min(stats['z_values']),
                    'max': np.max(stats['z_values'])
                }
            }
        }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)


def create_landmarks_table(frames_data):
    """Cria tabela pandas com dados dos landmarks"""
    data = []
    
    for frame_data in frames_data:
        for landmark in frame_data['filtered_landmarks']:
            data.append({
                'Frame': frame_data['frame_idx'],
                'Timestamp (s)': f"{frame_data['timestamp']:.2f}",
                'Landmark': landmark['name'],
                'X': f"{landmark['landmark']['x']:.4f}",
                'Y': f"{landmark['landmark']['y']:.4f}",
                'Z': f"{landmark['landmark']['z']:.4f}",
                'Visibility': f"{landmark['visibility']:.4f}",
                'Presence': f"{landmark['presence']:.4f}"
            })
    
    return pd.DataFrame(data)

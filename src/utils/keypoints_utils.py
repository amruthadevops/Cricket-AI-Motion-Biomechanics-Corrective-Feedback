import numpy as np

def safe_get_keypoints(keypoints, frame_idx=None, point_idx=None):
    """
    Safely get keypoints from a nested structure.
    
    Args:
        keypoints: Keypoints data structure (list, dict, or numpy array)
        frame_idx: Frame index if keypoints is a list of frames
        point_idx: Point index if keypoints is a single frame
        
    Returns:
        keypoints: Extracted keypoints or None if not found
    """
    if keypoints is None:
        return None
    
    try:
        # Handle different data structures
        if isinstance(keypoints, dict):
            if 'keypoints' in keypoints:
                return np.array(keypoints['keypoints'])
            elif 'pose' in keypoints:
                return np.array(keypoints['pose'])
            else:
                return np.array(list(keypoints.values()))
        elif isinstance(keypoints, list):
            if frame_idx is not None and frame_idx < len(keypoints):
                frame_data = keypoints[frame_idx]
                if isinstance(frame_data, (list, np.ndarray)) and point_idx is not None:
                    if isinstance(frame_data, list) and point_idx < len(frame_data):
                        return frame_data[point_idx]
                    elif isinstance(frame_data, np.ndarray) and point_idx * 3 < len(frame_data):
                        return frame_data[point_idx * 3: point_idx * 3 + 3]
                else:
                    return np.array(frame_data)
            else:
                return np.array(keypoints)
        elif isinstance(keypoints, np.ndarray):
            return keypoints
        else:
            return None
    except Exception as e:
        print(f"Error getting keypoints: {e}")
        return None

def normalize_keypoints(keypoints):
    """
    Normalize keypoints to a consistent numpy array format.
    
    Args:
        keypoints: Keypoints data in various formats
        
    Returns:
        keypoints: Normalized keypoints as numpy array or None
    """
    if keypoints is None:
        return None
    
    try:
        # Handle different data structures
        if isinstance(keypoints, dict):
            if 'keypoints' in keypoints:
                return np.array(keypoints['keypoints'])
            elif 'pose' in keypoints:
                return np.array(keypoints['pose'])
            else:
                return np.array(list(keypoints.values()))
        elif isinstance(keypoints, list):
            return np.array(keypoints)
        elif isinstance(keypoints, np.ndarray):
            return keypoints
        else:
            return None
    except Exception as e:
        print(f"Error normalizing keypoints: {e}")
        return None

def validate_keypoints(keypoints):
    """
    Validate keypoints structure and dimensions.
    
    Args:
        keypoints: Keypoints data to validate
        
    Returns:
        bool: True if keypoints are valid, False otherwise
    """
    if keypoints is None:
        return False
    
    try:
        # Check if it's a numpy array with at least 2 dimensions
        if isinstance(keypoints, np.ndarray):
            return len(keypoints.shape) >= 2 and keypoints.shape[1] >= 3
        
        # Check if it's a dictionary with valid values
        if isinstance(keypoints, dict):
            return len(keypoints) > 0
        
        # Check if it's a list with valid values
        if isinstance(keypoints, list):
            return len(keypoints) > 0
        
        return False
    except Exception as e:
        print(f"Error validating keypoints: {e}")
        return False

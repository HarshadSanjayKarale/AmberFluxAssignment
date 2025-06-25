import cv2
import os
from typing import List

def extract_frames_from_video(video_path: str, output_dir: str, interval: int = 1) -> List[str]:
    """
    Extract frames from video at specified intervals
    
    Args:
        video_path (str): Path to the input video file
        output_dir (str): Directory to save extracted frames
        interval (int): Extract one frame every N seconds (default: 1)
    
    Returns:
        List[str]: List of paths to extracted frame images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"📹 Video Info:")
    print(f"   FPS: {fps}")
    print(f"   Total Frames: {total_frames}")
    print(f"   Duration: {duration:.2f} seconds")
    
    frame_paths = []
    frame_interval = int(fps * interval)  # Extract frame every N seconds
    frame_count = 0
    extracted_count = 0
    
    # Get base filename without extension for naming frames
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame at specified intervals
            if frame_count % frame_interval == 0:
                # Generate frame filename
                frame_filename = f"{video_basename}_frame_{extracted_count:04d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                
                # Save frame
                success = cv2.imwrite(frame_path, frame)
                if success:
                    frame_paths.append(frame_path)
                    extracted_count += 1
                    print(f"🖼️  Extracted frame {extracted_count}: {frame_filename}")
                else:
                    print(f"⚠️  Failed to save frame: {frame_filename}")
            
            frame_count += 1
    
    finally:
        cap.release()
    
    print(f"✅ Extraction complete: {len(frame_paths)} frames saved")
    return frame_paths

def get_video_info(video_path: str) -> dict:
    """
    Get basic information about a video file
    
    Args:
        video_path (str): Path to the video file
    
    Returns:
        dict: Dictionary containing video information
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_seconds": duration,
        "resolution": f"{width}x{height}"
    }

def validate_video_file(video_path: str) -> bool:
    """
    Validate if a file is a valid video file
    
    Args:
        video_path (str): Path to the video file
    
    Returns:
        bool: True if valid video file, False otherwise
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        
        # Try to read first frame
        ret, frame = cap.read()
        cap.release()
        
        return ret and frame is not None
    
    except Exception:
        return False
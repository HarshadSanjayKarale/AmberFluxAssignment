import cv2
import numpy as np
from typing import List

def compute_feature_vector(image_path: str, method: str = "color_histogram") -> List[float]:
    """
    Compute feature vector for an image using specified method
    
    Args:
        image_path (str): Path to the image file
        method (str): Method to use for feature extraction
                     Options: "color_histogram", "gray_histogram", "combined"
    
    Returns:
        List[float]: Normalized feature vector
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Select feature extraction method
    if method == "color_histogram":
        return compute_color_histogram(image)
    elif method == "gray_histogram":
        return compute_gray_histogram(image)
    elif method == "combined":
        return compute_combined_features(image)
    else:
        raise ValueError(f"Unknown method: {method}")

def compute_color_histogram(image: np.ndarray, bins: int = 32) -> List[float]:
    """
    Compute normalized color histogram for BGR image
    
    Args:
        image (np.ndarray): Input image in BGR format
        bins (int): Number of bins for each color channel
    
    Returns:
        List[float]: Flattened and normalized histogram
    """
    # Compute histogram for each channel
    hist_b = cv2.calcHist([image], [0], None, [bins], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [bins], [0, 256])
    hist_r = cv2.calcHist([image], [2], None, [bins], [0, 256])
    
    # Concatenate histograms
    hist = np.concatenate([hist_b, hist_g, hist_r])
    
    # Normalize histogram
    hist = hist.flatten()
    hist = hist / (np.sum(hist) + 1e-8)  # Add small epsilon to avoid division by zero
    
    return hist.tolist()

def compute_gray_histogram(image: np.ndarray, bins: int = 64) -> List[float]:
    """
    Compute normalized grayscale histogram
    
    Args:
        image (np.ndarray): Input image in BGR format
        bins (int): Number of bins for histogram
    
    Returns:
        List[float]: Normalized histogram
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute histogram
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])
    
    # Normalize histogram
    hist = hist.flatten()
    hist = hist / (np.sum(hist) + 1e-8)
    
    return hist.tolist()

def compute_combined_features(image: np.ndarray) -> List[float]:
    """
    Compute combined feature vector using multiple methods
    
    Args:
        image (np.ndarray): Input image in BGR format
    
    Returns:
        List[float]: Combined and normalized feature vector
    """
    features = []
    
    # Color histogram (reduced bins for efficiency)
    color_hist = compute_color_histogram(image, bins=16)  # 16*3 = 48 features
    features.extend(color_hist)
    
    # Grayscale histogram
    gray_hist = compute_gray_histogram(image, bins=32)  # 32 features
    features.extend(gray_hist)
    
    # Basic color moments (mean, std for each channel)
    color_moments = compute_color_moments(image)  # 6 features
    features.extend(color_moments)
    
    # Texture features using LBP
    texture_features = compute_texture_features(image)  # 10 features
    features.extend(texture_features)
    
    # Normalize the entire feature vector
    features = np.array(features)
    features = features / (np.linalg.norm(features) + 1e-8)
    
    return features.tolist()

def compute_color_moments(image: np.ndarray) -> List[float]:
    """
    Compute color moments (mean and standard deviation) for each channel
    
    Args:
        image (np.ndarray): Input image in BGR format
    
    Returns:
        List[float]: Color moments [mean_b, mean_g, mean_r, std_b, std_g, std_r]
    """
    moments = []
    
    # Compute mean and std for each channel
    for channel in range(image.shape[2]):
        channel_data = image[:, :, channel].flatten()
        moments.append(np.mean(channel_data))
        moments.append(np.std(channel_data))
    
    return moments

def compute_texture_features(image: np.ndarray) -> List[float]:
    """
    Compute basic texture features using Local Binary Pattern (LBP)
    
    Args:
        image (np.ndarray): Input image in BGR format
    
    Returns:
        List[float]: Texture features
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Simple LBP implementation
    lbp = compute_lbp(gray)
    
    # Compute histogram of LBP
    hist = cv2.calcHist([lbp], [0], None, [10], [0, 256])
    hist = hist.flatten()
    hist = hist / (np.sum(hist) + 1e-8)
    
    return hist.tolist()

def compute_lbp(image: np.ndarray, radius: int = 1, neighbors: int = 8) -> np.ndarray:
    """
    Simple Local Binary Pattern implementation
    
    Args:
        image (np.ndarray): Grayscale image
        radius (int): Radius of circle
        neighbors (int): Number of neighbors
    
    Returns:
        np.ndarray: LBP image
    """
    rows, cols = image.shape
    lbp = np.zeros((rows, cols), dtype=np.uint8)
    
    for i in range(radius, rows - radius):
        for j in range(radius, cols - radius):
            center = image[i, j]
            pattern = 0
            
            # Simple 3x3 neighborhood
            for di in [-1, -1, -1, 0, 0, 1, 1, 1]:
                for dj in [-1, 0, 1, -1, 1, -1, 0, 1]:
                    if i + di < rows and j + dj < cols:
                        if image[i + di, j + dj] >= center:
                            pattern += 1
            
            lbp[i, j] = pattern
    
    return lbp

def compute_edge_features(image: np.ndarray) -> List[float]:
    """
    Compute edge-based features using Canny edge detection
    
    Args:
        image (np.ndarray): Input image in BGR format
    
    Returns:
        List[float]: Edge features
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Compute edge statistics
    edge_pixels = np.sum(edges > 0)
    total_pixels = edges.shape[0] * edges.shape[1]
    edge_density = edge_pixels / total_pixels
    
    # Horizontal and vertical edge components
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_mean = np.mean(edge_magnitude)
    edge_std = np.std(edge_magnitude)
    
    return [edge_density, edge_mean, edge_std]

def compare_feature_vectors(vec1: List[float], vec2: List[float]) -> float:
    """
    Compute similarity between two feature vectors using cosine similarity
    
    Args:
        vec1 (List[float]): First feature vector
        vec2 (List[float]): Second feature vector
    
    Returns:
        float: Cosine similarity score (0-1, higher is more similar)
    """
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    
    # Compute cosine similarity
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    
    similarity = dot_product / (norm_v1 * norm_v2)
    return max(0, similarity)  # Ensure non-negative
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import os
import shutil
from typing import List, Dict, Any
import uvicorn

from video_utils import extract_frames_from_video
from vector_utils import compute_feature_vector
from qdrant_utils import QdrantManager

app = FastAPI(title="Video Processing & Vector Similarity Search", version="1.0.0")

# Initialize Qdrant manager
qdrant_manager = QdrantManager()

# Ensure frames directory exists
FRAMES_DIR = "frames"
UPLOAD_DIR = "uploads"
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.on_event("startup")
async def startup_event():
    """Initialize Qdrant collection on startup"""
    try:
        await qdrant_manager.initialize_collection()
        print("✅ Qdrant collection initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Qdrant: {e}")
        raise e

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Video Processing & Vector Similarity Search API",
        "endpoints": {
            "upload_video": "/upload-video/",
            "query_by_frame": "/query/frame/{frame_index}",
            "query_by_image": "/query/image/",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        collection_info = await qdrant_manager.get_collection_info()
        return {
            "status": "healthy",
            "qdrant_status": "connected",
            "collection_points": collection_info.points_count if collection_info else 0
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "qdrant_status": "disconnected",
            "error": str(e)
        }

@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload video endpoint that:
    1. Accepts video file
    2. Extracts frames every 1 second
    3. Computes feature vectors
    4. Stores vectors in Qdrant
    """
    try:
        # Validate file type
        if not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        # Save uploaded video
        video_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"📹 Video saved: {video_path}")
        
        # Extract frames
        frame_paths = extract_frames_from_video(video_path, FRAMES_DIR)
        print(f"🖼️  Extracted {len(frame_paths)} frames")
        
        # Process each frame and store in Qdrant
        processed_frames = []
        for i, frame_path in enumerate(frame_paths):
            try:
                # Compute feature vector
                feature_vector = compute_feature_vector(frame_path)
                
                # Store in Qdrant
                point_id = await qdrant_manager.add_vector(
                    vector=feature_vector,
                    payload={
                        "frame_path": frame_path,
                        "frame_index": i,
                        "video_filename": file.filename
                    }
                )
                
                processed_frames.append({
                    "frame_index": i,
                    "frame_path": frame_path,
                    "point_id": point_id
                })
                
            except Exception as e:
                print(f"⚠️  Error processing frame {i}: {e}")
                continue
        
        # Clean up uploaded video
        os.remove(video_path)
        
        return {
            "message": "Video processed successfully",
            "video_filename": file.filename,
            "total_frames": len(frame_paths),
            "processed_frames": len(processed_frames),
            "frames": processed_frames[:10]  # Return first 10 for brevity
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@app.get("/query/frame/{frame_index}")
async def query_by_frame_index(frame_index: int, top_k: int = 5):
    """
    Query similar frames by frame index
    """
    try:
        # Find the frame file
        frame_files = [f for f in os.listdir(FRAMES_DIR) if f.endswith(('.jpg', '.png'))]
        frame_files.sort()
        
        if frame_index >= len(frame_files):
            raise HTTPException(status_code=404, detail="Frame index not found")
        
        frame_path = os.path.join(FRAMES_DIR, frame_files[frame_index])
        
        # Compute feature vector for query frame
        query_vector = compute_feature_vector(frame_path)
        
        # Search similar vectors
        results = await qdrant_manager.search_similar(query_vector, top_k)
        
        return {
            "query_frame": {
                "index": frame_index,
                "path": frame_path
            },
            "similar_frames": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying frame: {str(e)}")

@app.post("/query/image/")
async def query_by_image(file: UploadFile = File(...), top_k: int = Form(5)):
    """
    Query similar frames by uploading an image
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Save uploaded image temporarily
        temp_image_path = os.path.join(UPLOAD_DIR, f"query_{file.filename}")
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Compute feature vector for query image
        query_vector = compute_feature_vector(temp_image_path)
        
        # Search similar vectors
        results = await qdrant_manager.search_similar(query_vector, top_k)
        
        # Clean up temporary file
        os.remove(temp_image_path)
        
        return {
            "query_image": file.filename,
            "similar_frames": results
        }
        
    except Exception as e:
        # Clean up on error
        temp_image_path = os.path.join(UPLOAD_DIR, f"query_{file.filename}")
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        raise HTTPException(status_code=500, detail=f"Error querying image: {str(e)}")

@app.get("/frames/")
async def list_frames():
    """List all extracted frames"""
    try:
        frame_files = [f for f in os.listdir(FRAMES_DIR) if f.endswith(('.jpg', '.png'))]
        frame_files.sort()
        
        return {
            "total_frames": len(frame_files),
            "frames": frame_files
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing frames: {str(e)}")

@app.delete("/frames/")
async def clear_frames():
    """Clear all frames and reset Qdrant collection"""
    try:
        # Clear frames directory
        for filename in os.listdir(FRAMES_DIR):
            file_path = os.path.join(FRAMES_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        # Recreate Qdrant collection
        await qdrant_manager.initialize_collection()
        
        return {"message": "All frames cleared and collection reset"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing frames: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
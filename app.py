from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from greeter import GreeterAgent
import cv2
import base64
from typing import List
import asyncio
import json
import time
import logging
from collections import deque
from datetime import datetime
from pathlib import Path
import shutil
import sqlite3
from fastapi import HTTPException
from dotenv import load_dotenv
import os
import numpy as np
import uuid
from deepface import DeepFace
from contextlib import asynccontextmanager
import tensorflow as tf

# Load environment variables
load_dotenv()

# Configuration
config = {
    "rtsp_stream": os.getenv("RTSP_STREAM"),
    "host": os.getenv("HOST", "0.0.0.0"),
    "port": int(os.getenv("PORT", 8000))
}

# Set up logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Add at the top with other globals
active_connections = 0

# More detailed GPU detection logging
logger.info("Checking TensorFlow GPU availability...")
logger.info(f"CUDA visible devices: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
logger.info(f"TF_FORCE_GPU_ALLOW_GROWTH: {os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH', 'Not Set')}")

# Configure TensorFlow to use GPU
physical_devices = tf.config.list_physical_devices()
logger.info("All physical devices:")
for device in physical_devices:
    logger.info(f"Found device: {device.device_type} - {device.name}")

gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    try:
        # Enable GPU memory growth
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Enabled memory growth for GPU: {gpu}")

        # Make GPU visible to TensorFlow
        tf.config.set_visible_devices(gpu_devices, 'GPU')
        logical_devices = tf.config.list_logical_devices('GPU')
        logger.info(f"TensorFlow configured with GPU devices: {logical_devices}")

        # Test GPU availability
        with tf.device('/GPU:0'):
            logger.info("Testing GPU computation...")
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
            logger.info(f"GPU test computation result shape: {c.shape}")
            logger.info("GPU test successful")
    except RuntimeError as e:
        logger.error(f"GPU configuration failed: {e}")
else:
    logger.warning("No GPU devices available for TensorFlow")
    logger.info("Available devices:")
    logger.info(tf.config.list_physical_devices())

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the background processing
    await greeter_agent.start_background_processing()

    yield

    # Cleanup on shutdown
    await greeter_agent.cleanup()

app = FastAPI(lifespan=lifespan)

# Mount static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/unknown_faces", StaticFiles(directory="unknown_faces"), name="unknown_faces")
app.mount("/known_faces", StaticFiles(directory="known_faces"), name="known_faces")
templates = Jinja2Templates(directory="templates")

# Log the config
logger.debug(f"Creating GreeterAgent with config: {config}")

# Create the agent
greeter_agent = GreeterAgent(config)

# Verify agent creation
if greeter_agent.cap is None or not greeter_agent.cap.isOpened():
    logger.error("Failed to initialize GreeterAgent camera")
else:
    logger.info("GreeterAgent camera initialized successfully")

# Initialize the log storage
log_history = deque(maxlen=100)  # Stores last 100 logs

def log_handler(record):
    """Custom log handler to capture logs for the web interface"""
    log_entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "level": record.levelname,
        "message": record.getMessage()
    }
    log_history.append(log_entry)

# Create and add the custom handler
handler = logging.Handler()
handler.handle = log_handler  # Override handle method with our custom handler
logging.getLogger('__main__').addHandler(handler)
logging.getLogger('greeter').addHandler(handler)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global active_connections
    client_id = id(websocket)
    await websocket.accept()
    active_connections += 1
    last_frame_time = 0
    frame_interval = 1/30.0  # 30 FPS target

    logger.info(f"New client connected (ID: {client_id}). Active connections: {active_connections}")

    try:
        while True:
            current_time = time.time()
            if current_time - last_frame_time < frame_interval:
                await asyncio.sleep(0.001)  # Small sleep to prevent CPU spinning
                continue

            if websocket.client_state.value == 3:  # WebSocket.DISCONNECTED
                break

            # Get the latest processed frame
            with greeter_agent.frame_lock:
                latest_frame = getattr(greeter_agent, 'latest_processed_frame', None)

            if latest_frame:
                await websocket.send_text(json.dumps(latest_frame))
                last_frame_time = current_time

            await asyncio.sleep(0.001)  # Prevent tight loop

    except WebSocketDisconnect:
        logger.info(f"Client disconnected (ID: {client_id})")
    except Exception as e:
        if "close message has been sent" not in str(e):
            logger.error(f"WebSocket error: {e}")
    finally:
        active_connections -= 1
        logger.info(f"Client cleanup (ID: {client_id}). Active connections: {active_connections}")
        try:
            await websocket.close()
        except:
            pass

@app.get("/api/camera/status")
async def get_camera_status():
    if greeter_agent.cap and greeter_agent.cap.isOpened():
        return {"status": "connected"}
    return {"status": "disconnected"}

@app.post("/api/camera/reconnect")
async def reconnect_camera():
    greeter_agent.initialize_camera()
    return {"status": "reconnection attempted"}

@app.get("/api/logs")
async def get_logs():
    """Endpoint to retrieve recent logs"""
    return {"logs": list(log_history)}

@app.get("/api/unknown-faces")
async def get_unknown_faces():
    unknown_faces_dir = Path("unknown_faces")
    faces = [f.name for f in unknown_faces_dir.glob("*.jpg")]
    return faces

@app.post("/api/save-known-faces")
async def save_known_faces(data: dict):
    try:
        name = data['name']
        files = data['files']

        unknown_dir = Path("unknown_faces")
        known_dir = Path("known_faces")

        for file in files:
            unknown_path = unknown_dir / file
            if unknown_path.exists():
                # Create new filename with provided name
                new_filename = f"{name}_face_{unknown_path.stem.split('_')[-1]}.jpg"
                known_path = known_dir / new_filename

                # Move file to known_faces
                shutil.move(str(unknown_path), str(known_path))

        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/unknown-faces/delete-all")
async def delete_all_unknown_faces():
    try:
        unknown_faces_dir = Path("unknown_faces")
        for file in unknown_faces_dir.glob("*.jpg"):
            file.unlink()  # Delete the file
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/known-faces")
async def get_known_faces():
    try:
        known_faces_dir = Path("known_faces")
        faces = [f.name for f in known_faces_dir.glob("*.jpg")]
        return {"success": True, "faces": faces}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/known-faces/delete-person")
async def delete_person(data: dict):
    try:
        person_name = data["person"]
        known_faces_dir = Path("known_faces")
        # Delete all files that start with the person's name
        for file in known_faces_dir.glob(f"{person_name}_face*.jpg"):
            file.unlink()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


ACTION_CONFIG_FILE = Path("actions.config")

@app.get("/api/actions/config")
async def get_action_config():
    try:
        if ACTION_CONFIG_FILE.exists():
            config = json.loads(ACTION_CONFIG_FILE.read_text())
            return {"success": True, "config": config}
        return {"success": True, "config": {}}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/actions/config")
async def save_action_config(config: dict):
    try:
        # Validate URLs in a more permissive way
        for key, value in config.items():
            if isinstance(value, dict) and 'url' in value:  # Check if it's a dict with url
                url = value['url']
                if url and not (url.startswith('http://') or url.startswith('https://')):
                    return {"success": False, "error": f"Invalid URL format for {key}. URL must start with http:// or https://"}

        # Save configuration
        ACTION_CONFIG_FILE.write_text(json.dumps(config, indent=2))
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/database")
async def get_database():
    try:
        conn = sqlite3.connect('greeter.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT timestamp, person_name, category, sentiment, confidence, action, image_path
            FROM interactions
            ORDER BY timestamp DESC
        ''')
        rows = cursor.fetchall()
        return [
            {
                "timestamp": row[0],
                "person_name": row[1],
                "category": row[2],
                "sentiment": row[3],
                "confidence": row[4],
                "action": row[5],
                "image_path": row[6]
            }
            for row in rows
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-face")
async def upload_face(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file upload: {file.filename}")

        if tf.config.list_physical_devices('GPU'):
            logger.info("DeepFace using GPU for face extraction")
        elif tf.config.list_physical_devices('MPS'):
            logger.info("DeepFace using Apple Metal (MPS) for face extraction")
        else:
            logger.info("DeepFace using CPU for face extraction")

        # Read the uploaded file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            logger.error("Failed to decode image")
            return {"success": False, "error": "Invalid image file"}

        # Use DeepFace to extract faces
        try:
            extracted_faces = DeepFace.extract_faces(
                img_path=img,
                detector_backend='opencv',
                enforce_detection=False,
                align=True
            )

            if not extracted_faces:
                return {"success": False, "error": "No faces detected in image"}

            # Save each detected face
            unknown_faces_dir = Path("unknown_faces")
            unknown_faces_dir.mkdir(exist_ok=True)
            saved_files = []

            for face_data in extracted_faces:
                try:
                    face_img = face_data['face']

                    # Ensure proper format
                    if face_img.max() <= 1.0:
                        face_img = (face_img * 255).astype(np.uint8)

                    # Convert RGB to BGR if needed
                    if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                        face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)

                    # Save the face
                    filename = f"uploaded_face_{uuid.uuid4()}.jpg"
                    face_path = unknown_faces_dir / filename
                    success = cv2.imwrite(str(face_path), face_img)

                    if success:
                        saved_files.append(filename)
                        logger.info(f"Saved face to {face_path}")
                    else:
                        logger.error(f"Failed to save face to {face_path}")

                except Exception as e:
                    logger.error(f"Error processing extracted face: {str(e)}")
                    continue

            if not saved_files:
                return {"success": False, "error": "Failed to save any faces"}

            return {
                "success": True,
                "message": f"Successfully saved {len(saved_files)} faces",
                "files": saved_files
            }

        except Exception as e:
            logger.error(f"Face extraction failed: {str(e)}")
            return {"success": False, "error": "Face detection failed"}

    except Exception as e:
        logger.error(f"Face upload error: {str(e)}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host=config["host"], port=config["port"], log_level="info")

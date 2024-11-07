import torch
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import logging
import sqlite3
from datetime import datetime
from transformers import pipeline
import time
import os
import threading
from deepface import DeepFace
import traceback
from queue import Queue, Full
import aiohttp
import json
from threading import Thread
import tensorflow as tf
import json
import aiohttp
import traceback
import asyncio
from urllib.parse import urlencode
import base64

# Configure webhook debug logging while keeping other loggers at INFO
webhook_logger = logging.getLogger('greeter')
webhook_logger.setLevel(logging.INFO)

# Add a debug handler if needed
debug_handler = logging.StreamHandler()
debug_handler.setLevel(logging.DEBUG)
webhook_logger.addHandler(debug_handler)

class GreeterAgent:
    def __init__(self, config):
        """Initialize the GreeterAgent"""
        self.logger = logging.getLogger("greeter")
        self.logger.info("Initializing GreeterAgent")

        # Set up logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.DEBUG)

        # Silence YOLO's logger
        logging.getLogger('ultralytics').setLevel(logging.WARNING)

        self.logger.info("Initializing GreeterAgent")
        self.config = config
        self.cap = None
        self.frame_interval = 1.0 / 30
        self.last_frame_time = 0

        # Log config contents
        self.logger.debug(f"Received config: {self.config}")

        # Initialize AI models first
        try:
            self.logger.info("Loading YOLO model...")
            self.detector = YOLO('yolov8n.pt', verbose=False)

            # Check for available hardware acceleration
            if torch.backends.mps.is_available():
                self.device = 'mps'  # Use Metal Performance Shaders on Mac
                self.logger.info("YOLO using Apple Metal (MPS)")
            elif torch.cuda.is_available():
                self.device = 'cuda'  # Use CUDA if available
                self.logger.info("YOLO using CUDA GPU")
            else:
                self.device = 'cpu'  # Fallback to CPU
                self.logger.info("YOLO using CPU")

            self.detector.to(self.device)
            self.logger.info(f"AI model loaded successfully on {self.device}")

            # Define classes of interest (COCO dataset)
            self.target_classes = {
                0: 'person',
                16: 'dog',
                17: 'cat',
                62: 'chair',  # Common false positive for packages
                63: 'couch',  # Common false positive for packages
                64: 'potted plant',
                67: 'dining table',
                73: 'book',  # Can help with package detection
                84: 'book bag',
                85: 'suitcase',
                86: 'handbag',
                87: 'tie',
            }

            # Add detection cooldown to prevent overload
            self.last_detection_time = 0
            self.detection_interval = 0.2  # Reduce to 5 FPS for AI processing (was 0.066)

            # Process frames
            self.frame_skip = 2  # Skip frames to maintain performance
            self.frame_count = 0

        except Exception as e:
            self.logger.error(f"Failed to load AI model: {e}")
            self.detector = None

        # Initialize camera
        success = self.initialize_camera()
        if not success:
            self.logger.error("Failed to initialize any camera source")

        self.max_reconnect_attempts = 5
        self.reconnect_delay = 2  # seconds between reconnection attempts
        self.last_reconnect_attempt = 0

        # Add FPS tracking
        self.fps = 0
        self.fps_frames = 0
        self.fps_start = time.time()
        self.fps_update_interval = 1.0  # Update FPS every second

        # Adjust detection smoothing parameters
        self.last_detections = []  # Store recent detections
        self.detection_history = 10  # Reduce history length (was 30)
        self.confidence_threshold = 0.40  # Lower threshold slightly
        self.iou_threshold = 0.4  # Increase IOU threshold (was 0.3)
        self.smoothing_weight = 0.8  # Increase smoothing weight (was 0.6)

        # Add buffer management
        self.max_frame_delay = 1.0  # Increase allowed delay (was 0.5)
        self.frame_skip = 1  # Process every frame initially
        self.frame_buffer_size = 3  # Was 1

        # Adjust detection parameters for better performance
        self.detection_interval = 0.1  # Reduce AI processing frequency
        self.confidence_threshold = 0.75  # Slightly lower confidence threshold

        # Add performance monitoring
        self.processing_times = []
        self.max_processing_times = 30  # Track last 30 processing times

        # Add frame sync parameters
        self.max_latency = 1.0  # Maximum acceptable latency in seconds
        self.sync_check_interval = 1.0  # How often to check sync
        self.last_sync_check = time.time()  # Initialize sync check time
        self.frame_timestamp = time.time()  # Initialize frame timestamp

        # Add frame management
        self.latest_frame = None
        self.latest_processed_frame = None
        self.frame_lock = threading.Lock()
        self.running = True
        self.last_detection_time = 0
        self.detection_interval = 0.033  # Run detection at ~30fps
        self.frame_thread = threading.Thread(target=self._frame_grabber, daemon=True)
        self.frame_thread.start()

        # Initialize detection state tracking
        self.last_detection_state = {}
        self.last_detections = []
        self.detection_history = 10  # Number of frames to keep track of
        self.iou_threshold = 0.3
        self.smoothing_weight = 0.3

        # Add face analysis settings
        self.face_analysis_enabled = True
        self.last_face_result = None
        self.last_face_time = 0
        self.face_cache_duration = 5.0  # Increased to 5 seconds
        self.last_face_save_time = 0
        self.face_save_cooldown = 10.0
        self.last_emotion_state = None
        self.last_sentiment_time = 0
        self.sentiment_cooldown = 2.0
        self.sentiment_threshold = 0.80

        # Add face directories
        self.unknown_faces_dir = Path("unknown_faces")
        self.known_faces_dir = Path("known_faces")
        self.unknown_faces_dir.mkdir(exist_ok=True)
        self.known_faces_dir.mkdir(exist_ok=True)

        # Enhanced face tracking
        self.face_cache_duration = 5.0  # Base duration
        self.face_max_duration = 30.0   # Maximum time to show name without re-verification
        self.last_face_result = None
        self.face_tracking_enabled = True
        self.consecutive_matches = 0     # Track consecutive matches for the same person

        # Add face tracking for multiple faces
        self.face_tracks = {}  # Dictionary to store face tracks
        self.face_tracks_lock = threading.Lock()
        self.track_id_counter = 0
        self.track_timeout = 5.0  # Seconds before a track is considered expired
        self.max_tracking_distance = 100  # Pixels - max distance for track association

        # Initialize face analysis queue with logging
        self.logger.info("Initializing face analysis queue...")
        self.face_analysis_queue = Queue(maxsize=10)

        # Start face analysis thread with logging
        self.logger.info("Starting face analysis thread...")
        self.face_analysis_thread = Thread(
            target=self._face_analysis_worker,
            name="FaceAnalysisWorker",
            daemon=True
        )
        self.face_analysis_thread.start()
        self.logger.info("Face analysis thread started")

        self.action_config = {}
        #self.load_action_config()

        self.emotion_update_queue = Queue(maxsize=100)  # New queue for emotion updates
        self.last_emotion_updates = {}  # Cache of last known emotions

        # Add aiohttp session as instance variable
        self.session = None

        # Add webhook throttling
        self.last_webhook_time = 0
        self.webhook_cooldown = 8.0  # 15 seconds between webhooks

        # Add queues for async processing
        self.webhook_queue = Queue(maxsize=100)
        self.detection_queue = Queue(maxsize=100)

        # Start worker threads
        self.webhook_thread = Thread(target=self._webhook_worker, daemon=True)
        self.detection_thread = Thread(target=self._detection_worker, daemon=True)
        self.webhook_thread.start()
        self.detection_thread.start()

        self.logger.info("Worker threads started")

        # Add background processing task
        self.processing_task = None
        self.should_process = True

        # Add frame freeze detection parameters
        self.last_frame_content = None
        self.last_frame_change_time = time.time()
        self.max_freeze_duration = 3.0  # Maximum time (seconds) before considering frame frozen
        self.frame_similarity_threshold = 1.00  # Threshold for considering frames identical

        # Add queues for async processing
        self.webhook_queue = Queue(maxsize=100)
        self.detection_queue = Queue(maxsize=100)

        # Start worker threads
        self.webhook_thread = Thread(target=self._webhook_worker, daemon=True)
        self.detection_thread = Thread(target=self._detection_worker, daemon=True)
        self.webhook_thread.start()
        self.detection_thread.start()

    async def initialize_aiohttp_session(self):
        """Initialize aiohttp session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def cleanup(self):
        """Cleanup resources including background task"""
        self.should_process = False
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        await super().cleanup()  # Call existing cleanup

    async def send_webhook_notification(self, notification_type: str, data: dict):
        """Send webhook notification based on action configuration"""
        try:
            self.logger.debug(f"[WEBHOOK] TRACE: Called with type '{notification_type}' and data: {data}")
            current_time = time.time()

            # Check webhook cooldown
            if current_time - self.last_webhook_time < self.webhook_cooldown:
                self.logger.debug("[WEBHOOK] Skipped due to cooldown")
                return

            self.last_webhook_time = current_time

            # Load action configuration
            if not Path('actions.config').exists():
                self.logger.warning("[WEBHOOK] No actions.config file found")
                return

            with open('actions.config', 'r') as f:
                config = json.loads(f.read())
                self.logger.debug(f"[WEBHOOK] Loaded config: {config}")

            if notification_type not in config:
                self.logger.warning(f"[WEBHOOK] No configuration found for {notification_type}")
                self.logger.warning(f"[WEBHOOK] Available types: {list(config.keys())}")
                return

            webhook_config = config[notification_type]
            self.logger.debug(f"[WEBHOOK] Using config: {webhook_config}")

            # Get URL and method
            method = webhook_config.get('method', 'POST')
            url = webhook_config.get('url', '').strip()

            if not url:
                self.logger.warning("[WEBHOOK] No URL configured")
                return

            # Ensure session is initialized
            if self.session is None:
                self.logger.info("[WEBHOOK] Initializing new aiohttp session")
                self.session = aiohttp.ClientSession()

            # Parse and process body template
            if webhook_config.get('body'):
                try:
                    # Parse the body template string into a dict
                    body_template = json.loads(webhook_config['body'])
                    self.logger.debug(f"[WEBHOOK] Body template: {body_template}")

                    # Replace variables in the text field
                    if 'text' in body_template:
                        text = body_template['text']
                        for var_name, var_value in data.items():
                            placeholder = f"${{{var_name}}}"
                            if placeholder in text:
                                text = text.replace(placeholder, str(var_value))
                        body_template['text'] = text

                    # Convert to form data
                    form_data = {
                        'text': body_template.get('text', ''),
                        'color': body_template.get('color', '#00FF00'),
                        'repeat': str(body_template.get('repeat', 0))  # Convert to string
                    }

                    self.logger.info(f"[WEBHOOK] Attempting to send to {url} with form data: {form_data}")

                    # Create and run the task
                    async def send_request():
                        try:
                            async with self.session.post(
                                url=url,
                                data=form_data,
                                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                                timeout=aiohttp.ClientTimeout(total=5)
                            ) as response:
                                response_text = await response.text()
                                self.logger.info(f"[WEBHOOK] Response status: {response.status}")
                                self.logger.info(f"[WEBHOOK] Response text: {response_text}")
                                if response.status >= 400:
                                    self.logger.error(f"[WEBHOOK] Failed: {response.status} - {response_text}")
                                else:
                                    self.logger.info(f"[WEBHOOK] Success: {response_text}")
                        except Exception as e:
                            self.logger.error(f"[WEBHOOK] Request failed: {str(e)}")
                            self.logger.error(traceback.format_exc())

                    # Run the task
                    self.logger.debug("[WEBHOOK] Creating task")
                    await send_request()
                    self.logger.debug("[WEBHOOK] Task completed")

                except json.JSONDecodeError as e:
                    self.logger.error(f"[WEBHOOK] Invalid JSON in body template: {str(e)}")
                except Exception as e:
                    self.logger.error(f"[WEBHOOK] Processing failed: {str(e)}")
                    self.logger.error(traceback.format_exc())

        except Exception as e:
            self.logger.error(f"[WEBHOOK] Error: {str(e)}")
            self.logger.error(traceback.format_exc())

    def start_face_analysis_thread(self):
        """Start the face analysis worker thread"""
        # Create and start the worker thread
        self.face_analysis_queue = Queue(maxsize=10)
        self.face_analysis_thread = Thread(
            target=self._face_analysis_worker,
            name="FaceAnalysisWorker",
            daemon=True
        )

        try:
            self.face_analysis_thread.start()
            self.logger.info("Face analysis thread started")
        except Exception as e:
            self.logger.error(f"Failed to start face analysis thread: {e}")

    def _frame_grabber(self):
        """Background thread to continuously grab frames with auto-reconnect and freeze detection"""
        consecutive_errors = 0
        max_errors = 5
        error_reset_time = 10
        last_error_time = time.time()

        while self.running:
            try:
                if not self.cap or not self.cap.isOpened():
                    self.logger.warning("Camera connection lost, attempting reconnection...")
                    if self.attempt_reconnection():
                        consecutive_errors = 0
                        continue
                    time.sleep(1)
                    continue

                # Grab the next frame
                ret = self.cap.grab()

                if not ret:
                    consecutive_errors += 1
                    last_error_time = time.time()
                    self.logger.warning(f"Frame grab failed (errors: {consecutive_errors}/{max_errors})")

                    if consecutive_errors >= max_errors:
                        self.logger.error("Too many consecutive errors, forcing reconnection...")
                        self.cap.release()
                        self.cap = None
                        continue

                    time.sleep(0.1)
                    continue

                # Retrieve and check for frozen frame
                ret, frame = self.cap.retrieve()
                if ret:
                    frame = np.asarray(frame, dtype=np.uint8)

                    # Check for frozen frame
                    if self._is_frame_frozen(frame):
                        self.logger.warning("Detected frozen frame, forcing RTSP reconnection...")
                        self.cap.release()
                        self.cap = None
                        continue

                    with self.frame_lock:
                        self.latest_frame = frame
                        self.frame_timestamp = time.time()
                    consecutive_errors = 0
                else:
                    consecutive_errors += 1
                    last_error_time = time.time()

            except Exception as e:
                self.logger.error(f"Frame grabber error: {str(e)}")
                consecutive_errors += 1
                last_error_time = time.time()

                if consecutive_errors >= max_errors:
                    self.logger.error("Too many consecutive errors, forcing reconnection...")
                    if self.cap:
                        self.cap.release()
                    self.cap = None
                    time.sleep(1)

            time.sleep(0.001)

    def _is_frame_frozen(self, current_frame):
        """Check if the frame is frozen by comparing with previous frame"""
        try:
            current_time = time.time()

            # Initialize if this is the first frame
            if self.last_frame_content is None:
                self.last_frame_content = current_frame.copy()
                self.last_frame_change_time = current_time
                return False

            # Calculate frame similarity
            try:
                # Downscale frames for faster comparison
                small_current = cv2.resize(current_frame, (32, 32))
                small_last = cv2.resize(self.last_frame_content, (32, 32))

                # Calculate mean squared error
                mse = np.mean((small_current - small_last) ** 2)
                similarity = 1 - (mse / 255**2)  # Normalize to 0-1 range

                # Check if frames are too similar for too long
                if similarity > self.frame_similarity_threshold:
                    if current_time - self.last_frame_change_time > self.max_freeze_duration:
                        self.logger.warning(f"Frame frozen for {current_time - self.last_frame_change_time:.1f} seconds")
                        return True
                else:
                    # Frame has changed, update reference
                    self.last_frame_content = current_frame.copy()
                    self.last_frame_change_time = current_time

                return False

            except Exception as e:
                self.logger.error(f"Frame comparison error: {str(e)}")
                return False

        except Exception as e:
            self.logger.error(f"Frame freeze detection error: {str(e)}")
            return False

    def initialize_camera(self):
        """Initialize camera with better error handling and logging"""
        self.logger.info("Starting camera initialization")

        if self.cap:
            self.logger.info("Releasing existing camera")
            self.cap.release()

        # First try RTSP
        if self.try_rtsp_stream():
            # Verify connection by reading a test frame
            ret, _ = self.cap.read()
            if ret:
                self.logger.info("Camera initialization successful")
                return True
            else:
                self.logger.error("Camera initialized but failed to read test frame")
                return False

        # If RTSP fails, try fallback sources
        return self.fallback_to_test_source()

    def try_rtsp_stream(self):
        """Attempt to connect to RTSP stream with better error handling"""
        try:
            if not self.config.get("rtsp_stream"):
                self.logger.error("No RTSP stream URL in config")
                return False

            # Release existing capture if any
            if self.cap is not None:
                self.cap.release()
                self.cap = None

            stream_url = self.config["rtsp_stream"]

            # Enhanced FFMPEG options for better stability and auto-reconnect
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
                'rtsp_transport;tcp'  # Use TCP for RTSP
                '|rtsp_flags;prefer_tcp'  # Prefer TCP over UDP
                '|stimeout;5000000'  # Socket timeout in microseconds (5 seconds)
                '|fflags;nobuffer'  # Reduce buffering
                '|flags;low_delay'  # Minimize latency
                '|reorder_queue_size;0'  # Disable reordering
                '|max_delay;500000'  # Maximum demux-decode delay (500ms)
            )

            self.logger.info(f"Connecting to RTSP stream: {stream_url}")
            self.cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)

            if not self.cap.isOpened():
                self.logger.error("Failed to open RTSP stream")
                return False

            # Minimal buffering
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Reset error counter on successful connection
            self.consecutive_errors = 0
            return True

        except Exception as e:
            self.logger.exception(f"RTSP initialization failed: {str(e)}")
            return False

    def fallback_to_test_source(self):
        """Try fallback sources with better error handling"""
        self.logger.info("Attempting fallback sources")

        try:
            self.logger.info("Trying webcam (device 0)")
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                # Optimize camera settings for low latency
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

                # Set additional OpenCV backend properties for lower latency
                self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # Skip color conversion if possible

                ret, frame = self.cap.read()
                if ret:
                    self.logger.info("Successfully initialized webcam at 640x480")
                    return True
                else:
                    self.logger.error("Could not read frame from webcam")
            else:
                self.logger.error("Could not open webcam")
        except Exception as e:
            self.logger.exception(f"Webcam initialization failed: {str(e)}")

        self.logger.error("All camera initialization attempts failed")
        return False

    # Get the latest frame with all processing and prepare for web display
    # This is the main function that is called to get the latest frame
    # It will return the latest frame with all processing and prepare for web display
    # It will also return the frame as a numpy array
    def get_frame(self):
        """Get the latest frame with all processing and prepare for web display"""
        try:
            current_time = time.time()
            webhook_data = None  # Initialize webhook_data at the start

            # Update FPS calculation
            self.fps_frames += 1
            if current_time - self.fps_start >= self.fps_update_interval:
                self.fps = self.fps_frames / (current_time - self.fps_start)
                self.fps_frames = 0
                self.fps_start = current_time

            # Get frame from grabber thread
            with self.frame_lock:
                if self.latest_frame is None:
                    self.logger.warning("No frame available")
                    return False, None
                frame = self.latest_frame.copy()  # Make copy to avoid conflicts

            # Queue detection instead of running it directly
            should_run_detection = False
            if not hasattr(self, 'last_detection_time') or \
               current_time - self.last_detection_time >= 0.5:  # 500ms = 2 FPS
                should_run_detection = True
                self.last_detection_time = current_time

                try:
                    self.detection_queue.put_nowait({
                        'frame': frame.copy(),
                        'timestamp': current_time
                    })
                except Full:
                    self.logger.debug("Detection queue full, skipping frame")

            # Use last known results if available
            results = None
            with self.frame_lock:
                if hasattr(self, 'last_results'):
                    results = self.last_results

            if results is None:
                return True, frame  # Return unprocessed frame if no results

            # Process detections with smoothing (existing code)
            for box in results.boxes:
                try:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()[:4]
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])

                    # Initialize track_id before using it
                    track_id = None

                    # Process person detections by class_id
                    if confidence >= self.confidence_threshold and class_id == 0:
                        try:
                            track_id = str(self._get_track_id((x1, y1, x2, y2)))
                        except Exception as e:
                            self.logger.error(f"Tracking failed: {str(e)}")
                            continue  # Skip this detection if tracking fails

                        # Extract face region
                        try:
                            face_region = frame[int(y1):int(y2), int(x1):int(x2)].copy()
                        except Exception as e:
                            self.logger.error(f"Face extraction failed: {str(e)}")
                            continue

                        # Queue analysis if we have a valid face region
                        if face_region.size > 0:
                            should_analyze = True
                            if track_id in self.face_tracks:
                                last_analysis = self.face_tracks[track_id].get('last_analysis_time', 0)
                                if current_time - last_analysis < 2.0:
                                    should_analyze = False

                            if should_analyze:
                                try:
                                    self.face_analysis_queue.put_nowait({
                                        'faces': [{
                                            'face': face_region,
                                            'bbox': (x1, y1, x2, y2)
                                        }],
                                        'track_id': track_id,
                                        'timestamp': current_time
                                    })
                                    if track_id in self.face_tracks:
                                        self.face_tracks[track_id]['last_analysis_time'] = current_time
                                except Full:
                                    self.logger.debug("Face analysis queue is full, skipping frame")

                    # Draw bounding box with smoothing
                    try:
                        if not hasattr(self, 'box_positions'):
                            self.box_positions = {}

                        # Only draw if we have a valid track_id
                        if track_id:
                            # Smooth box positions
                            if track_id not in self.box_positions:
                                self.box_positions[track_id] = {
                                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                    'last_update': current_time
                                }
                            else:
                                # Only update positions every 300ms
                                if current_time - self.box_positions[track_id]['last_update'] >= 0.3:
                                    # Smooth movement
                                    alpha = 0.3
                                    pos = self.box_positions[track_id]
                                    pos['x1'] = pos['x1'] * (1-alpha) + x1 * alpha
                                    pos['y1'] = pos['y1'] * (1-alpha) + y1 * alpha
                                    pos['x2'] = pos['x2'] * (1-alpha) + x2 * alpha
                                    pos['y2'] = pos['y2'] * (1-alpha) + y2 * alpha
                                    pos['last_update'] = current_time

                            # Draw using smoothed positions
                            smooth_pos = self.box_positions[track_id]
                            cv2.rectangle(frame,
                                      (int(smooth_pos['x1']), int(smooth_pos['y1'])),
                                      (int(smooth_pos['x2']), int(smooth_pos['y2'])),
                                      (0, 255, 0),
                                      2)

                            # Add label with identity and emotion
                            label = f"Person ({confidence:.2f})"
                            if track_id in self.face_tracks:
                                track = self.face_tracks[track_id]
                                label_parts = []

                                # Add identity if available
                                identity = track.get('identity')
                                if identity and identity != 'unknown':
                                    label_parts.append(f"Name: {identity}")

                                # Add emotion if available
                                emotion = track.get('emotion')
                                emotion_conf = track.get('emotion_confidence')
                                if emotion and emotion_conf and emotion_conf > 40:
                                    label_parts.append(f"Emotion: {emotion} ({emotion_conf:.1f}%)")

                                if label_parts:
                                    label += " | " + " | ".join(label_parts)

                            cv2.putText(frame,
                                      label,
                                      (int(smooth_pos['x1']), int(smooth_pos['y1']) - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      0.6,
                                      (0, 255, 0),
                                      2)

                    except Exception as e:
                        self.logger.error(f"Drawing error: {str(e)}")
                        continue

                except Exception as e:
                    self.logger.error(f"Box processing error: {str(e)}")
                    continue

            return True, frame

        except Exception as e:
            self.logger.error(f"Frame processing error: {str(e)}")
            if 'frame' in locals():
                return False, frame
            return False, None

    def _drop_frames_keep_detections(self):
        """Drop frames while maintaining detection boxes"""
        frames_dropped = 0
        start_time = time.time()
        last_frame = None

        self.logger.info("Starting frame drop sequence...")

        # Drop frames until we catch up
        while frames_dropped < 300:  # Reduced limit for faster response
            if not self.cap.grab():
                self.logger.error("Failed to grab frame during dropping")
                break

            frames_dropped += 1

            # Check every few frames
            if frames_dropped % 3 == 0:  # Check more frequently
                ret, frame = self.cap.retrieve()
                if not ret:
                    continue

                last_frame = frame
                current_time = time.time()

                # If we've dropped enough frames, return
                if frames_dropped > 30:  # Ensure we drop at least some frames
                    self.logger.info(f"Dropped {frames_dropped} frames")
                    return True, last_frame

        self.logger.warning(f"Frame dropping limit reached after {frames_dropped} frames")
        if last_frame is not None:
            return True, last_frame
        return False, None

    def _clear_buffer(self):
        """Improved buffer clearing"""
        if not self.cap or not self.cap.isOpened():
            return

        max_frames_to_clear = 5  # Limit how many frames we'll clear
        frames_cleared = 0

        while frames_cleared < max_frames_to_clear:
            ret = self.cap.grab()
            if not ret:
                break
            frames_cleared += 1

            # Stop if we're caught up
            if time.time() - self.last_frame_time < self.max_frame_delay:
                break

        if frames_cleared > 0:
            self.logger.debug(f"Cleared {frames_cleared} frames from buffer")

    def attempt_reconnection(self):
        """Attempt to reconnect to the camera with backoff"""
        current_time = time.time()

        # Enforce delay between reconnection attempts
        if current_time - self.last_reconnect_attempt < self.reconnect_delay:
            return False

        self.last_reconnect_attempt = current_time
        self.logger.info("Attempting camera reconnection...")

        # Release existing connection if any
        if self.cap:
            self.cap.release()
            self.cap = None

        # Try to reinitialize
        for attempt in range(self.max_reconnect_attempts):
            self.logger.info(f"Reconnection attempt {attempt + 1}/{self.max_reconnect_attempts}")

            if self.initialize_camera():
                self.logger.info("Successfully reconnected to camera")
                return True

            # Wait before next attempt (exponential backoff)
            time.sleep(min(self.reconnect_delay * (2 ** attempt), 30))

        self.logger.error("Failed to reconnect after maximum attempts")
        return False

    def setup_logging(self):
        """Setup logging with DEBUG level"""
        logging.basicConfig(
            level=logging.DEBUG,  # Changed from INFO to DEBUG
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("greeter")
        self.logger.setLevel(logging.DEBUG)  # Explicitly set logger level to DEBUG

    def setup_database(self):
        self.conn = sqlite3.connect('greeter.db')
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                person_name TEXT,
                category TEXT,
                sentiment TEXT,
                action TEXT,
                image_path TEXT
            )
        ''')
        self.conn.commit()

    def _get_track_id(self, bbox):
        """Get or create track ID for detection"""
        current_time = time.time()

        # Clean up old tracks
        self.face_tracks = {
            track_id: track for track_id, track in self.face_tracks.items()
            if current_time - track['last_seen'] < self.track_timeout
        }

        # Try to match with existing track
        for track_id, track in self.face_tracks.items():
            if self._calculate_iou(bbox, track['bbox']) > self.iou_threshold:
                # Update existing track
                track['bbox'] = bbox
                track['last_seen'] = current_time
                return track_id

        # Create new track
        new_track_id = str(int(time.time() * 1000))
        self.face_tracks[new_track_id] = {
            'bbox': bbox,
            'last_seen': current_time,
            'last_processed': 0,  # Add this field
            'name': None,
            'emotion': None,
            'emotion_confidence': 0,
            'consecutive_matches': 0
        }
        return new_track_id

    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        # Convert boxes to [x1, y1, x2, y2] format
        box1 = [float(x) for x in box1]
        box2 = [float(x) for x in box2]

        # Calculate intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # Calculate intersection area
        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection

        # Return IoU
        return intersection / union if union > 0 else 0

    def _update_face_tracks(self, face_locations, current_time):
        """Update face tracking with new detections"""
        # Remove expired tracks
        self.face_tracks = {
            track_id: track for track_id, track in self.face_tracks.items()
            if current_time - track['last_seen'] < self.track_timeout
        }

        # Match new detections to existing tracks
        unmatched_detections = []
        matched_track_ids = set()

        for face_loc in face_locations:
            bbox = face_loc['bbox']
            center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

            # Find closest track
            best_track_id = None
            min_distance = float('inf')

            for track_id, track in self.face_tracks.items():
                if track_id in matched_track_ids:
                    continue

                track_center = track['center']
                distance = ((center[0] - track_center[0]) ** 2 +
                          (center[1] - track_center[1]) ** 2) ** 0.5

                if distance < min_distance and distance < self.max_tracking_distance:
                    min_distance = distance
                    best_track_id = track_id

            if best_track_id is not None:
                # Update existing track
                self.face_tracks[best_track_id].update({
                    'bbox': bbox,
                    'center': center,
                    'last_seen': current_time
                })
                matched_track_ids.add(best_track_id)
            else:
                # Create new track
                unmatched_detections.append({
                    'bbox': bbox,
                    'center': center,
                    'face_img': face_loc['face_img']
                })

        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            track_id = self.track_id_counter
            self.track_id_counter += 1
            self.face_tracks[track_id] = {
                'bbox': detection['bbox'],
                'center': detection['center'],
                'last_seen': current_time,
                'name': None,
                'confidence': 0,
                'consecutive_matches': 0
            }

    def handle_pet(self, region, pet_type, confidence):
        """Handle pet detections"""
        pass

    def handle_object(self, region, object_type, confidence):
        """Handle other object detections"""
        pass

    def get_status(self):
        """Get current status of the GreeterAgent"""
        return {
            "fps": round(self.fps, 1),
            "device": self.device,
            "camera_active": bool(self.cap and self.cap.isOpened()),
            "detection_interval": self.detection_interval,
            "confidence_threshold": self.confidence_threshold,
            "last_detection_time": self.last_detection_time,
            "current_time": time.time()
        }

    def __del__(self):
        """Cleanup when object is destroyed"""
        self.running = False
        if hasattr(self, 'frame_thread'):
            self.frame_thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        # Convert boxes to [x1, y1, x2, y2] format
        box1 = [float(x) for x in box1]
        box2 = [float(x) for x in box2]

        # Calculate intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # Calculate intersection area
        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection

        # Return IoU
        return intersection / union if union > 0 else 0

    def update_detection_history(self, current_boxes):
        """Update detection history with new detections"""
        current_time = time.time()

        # Group current detections by class
        current_state = {}
        for new_box in current_boxes:
            if new_box['class_id'] in self.target_classes:
                class_name = self.target_classes[new_box['class_id']]
                if class_name not in current_state:
                    current_state[class_name] = 0
                current_state[class_name] += 1

        # Only log if the state has changed
        if current_state != self.last_detection_state:
            summary = []
            for class_name, count in current_state.items():
                summary.append(f"{count} {class_name}(s)")

            if summary:
                self.logger.info("Scene changed - Detected: " + ", ".join(summary))
                self.last_detection_state = current_state.copy()

        # Age existing detections
        for det in self.last_detections:
            det['age'] += 1

        # Keep detections that haven't aged out
        self.last_detections = [det for det in self.last_detections if det['age'] < self.detection_history]

        # Add new detections with improved position averaging
        for new_box in current_boxes:
            matched = False
            best_iou = 0
            best_match = None

            # Find best matching existing detection
            for existing in self.last_detections:
                if existing['class_id'] == new_box['class_id']:
                    iou = self.calculate_iou(existing['box'], new_box['box'])
                    if iou > best_iou and iou > self.iou_threshold:
                        best_iou = iou
                        best_match = existing

            if best_match is not None:
                # Update existing detection
                weight = max(0.3, self.smoothing_weight * (1 - best_match['age'] / self.detection_history))
                best_match['box'] = [
                    weight * new_box['box'][i] + (1 - weight) * best_match['box'][i]
                    for i in range(4)
                ]
                best_match['confidence'] = (
                    weight * new_box['confidence'] +
                    (1 - weight) * best_match['confidence']
                )
                best_match['age'] = 0
                matched = True

            if not matched:
                # Add new detection with a small initial age
                new_box['age'] = 2
                self.last_detections.append(new_box)

    def draw_smooth_detections(self, frame, detections):
        """Draw detections with labels and emotion data"""
        try:
            # Debug what we're receiving
            self.logger.debug(f"Drawing {len(detections)} detections")
            self.logger.debug(f"Frame type: {type(frame)}, shape: {frame.shape}")

            # Ensure we're working with a numpy array
            if not isinstance(frame, np.ndarray):
                self.logger.error("Invalid frame type in draw_smooth_detections")
                return None

            display_frame = frame.copy()

            for det in detections:
                x1, y1, x2, y2 = map(int, det[:4])
                confidence = det[4]
                class_id = int(det[5])

                # Draw box
                color = (0, 255, 0) if class_id == 0 else (255, 0, 0)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

                # Add label
                if class_id == 0 and len(det) > 6:  # Person with track_id
                    track_id = det[6]
                    label = f"Person ({confidence:.2f})"

                    # Add emotion if available
                    if track_id in self.face_tracks:
                        track = self.face_tracks[track_id]
                        emotion = track.get('emotion')
                        emotion_conf = track.get('emotion_confidence')
                        if emotion and emotion_conf:
                            label += f" | {emotion}: {emotion_conf:.1f}%"
                else:
                    label = f"{self.target_classes.get(class_id, 'unknown')} ({confidence:.2f})"

                # Draw label background
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(display_frame,
                            (x1, y1 - text_height - 10),
                            (x1 + text_width, y1),
                            (0, 0, 0),
                            -1)
                cv2.putText(display_frame, label,
                          (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.6,
                          color,
                          2)

            # Debug output frame
            self.logger.debug(f"Output frame type: {type(display_frame)}, shape: {display_frame.shape}")
            return display_frame

        except Exception as e:
            self.logger.error(f"Error in draw_smooth_detections: {str(e)}")
            return frame  # Return original frame on error

    # This is a worker thread that processes faces in the queue
    def _face_analysis_worker(self):
        """Background worker for face analysis"""
        from deepface import DeepFace
        thread_logger = logging.getLogger("greeter.face_worker")
        thread_logger.info("Face analysis worker started")

        # Create an event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        os.makedirs("known_faces", exist_ok=True)
        os.makedirs("unknown_faces", exist_ok=True)
        # Initialize save times tracking
        last_save_times = {}
        SAVE_INTERVAL = 10.0  # Save faces every 10 seconds

        try:
            db_conn = sqlite3.connect('greeter.db')
            cursor = db_conn.cursor()
            # Ensure table exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    person_name TEXT,
                    category TEXT,
                    sentiment TEXT,
                    confidence REAL,
                    action TEXT,
                    image_path TEXT
                )
            ''')
            db_conn.commit()

            while True:
                #make sure we start with an unknown identity
                identity = "unknown"

                try:
                    face_data = self.face_analysis_queue.get()
                    faces = face_data.get('faces', [])
                    track_id = face_data.get('track_id')
                    current_time = time.time()

                    if not faces:
                        thread_logger.debug("No faces in data, skipping")
                        continue

                    # Log face data details
                    thread_logger.debug(f"Received face data for track {track_id}")
                    thread_logger.debug(f"Number of faces: {len(faces)}")
                    for i, face in enumerate(faces):
                        face_img = face['face']
                        thread_logger.debug(f"Face {i} shape: {face_img.shape}")

                    # Process the first face (assuming one face per person)
                    face = faces[0]
                    face_img = face['face']

                    # Extract face with DeepFace for better face region detection
                    try:
                        # Ensure face_img is uint8 before processing
                        face_img = np.asarray(face_img, dtype=np.uint8)

                        extracted_faces = DeepFace.extract_faces(
                            img_path=face_img,
                            detector_backend='retinaface',  # Changed from opencv to retinaface
                            enforce_detection=True,  # Set to True to ensure face detection
                            align=True
                        )

                        if extracted_faces and len(extracted_faces) > 0:
                            # Get the first detected face and ensure proper format
                            extracted_face = extracted_faces[0]['face']

                            # Normalize pixel values to 0-255 range if needed
                            if extracted_face.max() <= 1.0:
                                extracted_face = (extracted_face * 255).astype(np.uint8)

                            # Convert RGB to BGR (or BGR to RGB if needed)
                            face_img = cv2.cvtColor(extracted_face, cv2.COLOR_RGB2BGR)

                            # Double-check final format
                            face_img = np.asarray(face_img, dtype=np.uint8)

                            thread_logger.info(f"Face image stats - shape: {face_img.shape}, dtype: {face_img.dtype}, range: [{face_img.min()}, {face_img.max()}]")
                        else:
                            thread_logger.warning("No face detected by DeepFace extractor")
                            #let eject from this loop and try again, we're not going to do anything with this frame
                            continue
                    except Exception as e:
                        thread_logger.warning(f"Face extraction failed, using original: {str(e)}")
                        continue

                    # Facial recognition test
                    #face recognition
                    # Face recognition against known faces database
                    # Skip face recognition if known_faces directory is empty
                    if not os.listdir("known_faces"):
                        thread_logger.info("No known faces in database, skipping recognition")
                    else:
                        try:
                            # Clear any existing tensorflow session
                            tf.keras.backend.clear_session()

                            # Save current face for debugging
                            # Create ram directory if it doesn't exist
                            os.makedirs("ram", exist_ok=True)
                            current_face_path = "ram/currentface.jpg"
                            cv2.imwrite(current_face_path, face_img)

                            dfs = DeepFace.find(
                                img_path=current_face_path,
                                db_path="known_faces",
                                detector_backend='opencv',
                                model_name='Facenet',
                                enforce_detection=False,
                                silent=True,
                                align=True,
                                distance_metric='euclidean_l2'
                            )

                            if len(dfs) > 0 and not dfs[0].empty:
                                match = dfs[0].iloc[0]
                                distance = match['distance']

                                # Adjust threshold for Facenet euclidean distance
                                # Lower distance = better match
                                RECOGNITION_THRESHOLD = 0.8  # Increase this value to be more lenient

                                if distance <= RECOGNITION_THRESHOLD:
                                    identity = os.path.basename(match['identity']).split('_face_')[0]
                                    thread_logger.info(f"Match found with distance {distance:.3f}")

                                else:
                                    identity = "unknown"
                                    thread_logger.info(f"Match rejected due to high distance {distance:.3f}")

                                with self.face_tracks_lock:
                                    if track_id in self.face_tracks:
                                        self.face_tracks[track_id].update({
                                            'identity': identity,
                                            'recognition_distance': distance,
                                            'last_recognition': current_time
                                        })
                                        thread_logger.info(f"Recognized {identity} for track {track_id} (distance: {distance:.3f})")
                            else:
                                thread_logger.debug(f"No matching faces found in database for track {track_id}")

                        except Exception as e:
                            thread_logger.warning(f"Face recognition failed: {str(e)}\nTraceback: {traceback.format_exc()}")
                            continue

                    # Check if we should save this face
                    should_save = False
                    if identity == "unknown" and \
                       ((track_id not in last_save_times or \
                       (current_time - last_save_times.get(track_id, 0)) >= SAVE_INTERVAL)):
                        should_save = True
                        last_save_times[track_id] = current_time

                    # Save face if needed
                    if should_save:
                        try:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"unknown_face_{timestamp}.jpg"

                            filepath = os.path.join("unknown_faces", filename)

                            cv2.imwrite(filepath, face_img)
                            thread_logger.info(f"Saved unknown face: {filepath}")
                        except Exception as e:
                            thread_logger.error(f"Error saving face: {str(e)}")

                    # Do emotion analysis on the extracted face
                    try:
                        result = DeepFace.analyze(
                            img_path=face_img,
                            actions=['emotion'],
                            enforce_detection=False,
                            detector_backend='opencv',
                            silent=True
                        )

                        if isinstance(result, list):
                            result = result[0]

                        emotion_scores = result.get('emotion', {})
                        dominant_emotion = result.get('dominant_emotion')
                        confidence = emotion_scores.get(dominant_emotion, 0)

                        thread_logger.info(f"Raw emotion scores: {emotion_scores}")

                        # Update face tracks
                        with self.face_tracks_lock:
                            if track_id in self.face_tracks:
                                self.face_tracks[track_id].update({
                                    'emotion': dominant_emotion,
                                    'emotion_confidence': confidence,
                                    'last_seen': current_time,
                                    'emotion_scores': emotion_scores
                                })
                                thread_logger.info(f"Updated track {track_id} with emotion {dominant_emotion} ({confidence:.1f}%)")

                    except Exception as e:
                        thread_logger.error(f"Emotion analysis failed: {str(e)}")

                    # After face recognition, when we have a match
                    if len(dfs) > 0 and not dfs[0].empty:
                        match = dfs[0].iloc[0]
                        distance = match['distance']

                        if distance <= RECOGNITION_THRESHOLD:
                            identity = os.path.basename(match['identity']).split('_face_')[0]
                            thread_logger.info(f"Match found with distance {distance:.3f}")

                            # Get emotion data if available
                            emotion = None
                            emotion_confidence = 0
                            if 'emotion' in locals():
                                emotion = dominant_emotion
                                emotion_confidence = confidence

                            # Get the matched image path from DeepFace results
                            matched_image_path = match['identity']  # This is the full path to the matched face

                            # Create timestamp
                            current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                            try:
                                # Database logging
                                cursor.execute('''
                                    INSERT INTO interactions
                                    (person_name, category, sentiment, confidence, action, image_path)
                                    VALUES (?, ?, ?, ?, ?, ?)
                                ''', (
                                    identity,
                                    'recognition',
                                    emotion,
                                    emotion_confidence,
                                    f'Recognized with distance {distance:.3f}',
                                    matched_image_path
                                ))
                                db_conn.commit()
                                thread_logger.info(f"Logged interaction for {identity}")

                                # Send webhook notification using the event loop
                                webhook_data = {
                                    'timestamp': current_timestamp,
                                    'image': matched_image_path,
                                    'person': identity,
                                    'confidence': float(distance),
                                    'sentiment': emotion or 'unknown',
                                    'category': 'known'  # Make sure this is 'known'
                                }
                                loop.run_until_complete(self.send_webhook_notification('known', webhook_data))

                            except Exception as e:
                                thread_logger.error(f"Database/webhook logging failed: {str(e)}")
                    else:
                        # For unknown faces
                        try:
                            # Save unknown face image with timestamp
                            current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            image_filename = f"unknown_faces/unknown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                            cv2.imwrite(image_filename, face_img)

                            # Get emotion data if available
                            emotion = None
                            emotion_confidence = 0
                            if 'emotion' in locals():
                                emotion = dominant_emotion
                                emotion_confidence = confidence

                            cursor.execute('''
                                INSERT INTO interactions
                                (person_name, category, sentiment, confidence, action, image_path)
                                VALUES (?, ?, ?, ?, ?, ?)
                            ''', (
                                'unknown',
                                'detection',
                                emotion,
                                emotion_confidence,
                                f'Unknown face detected with emotion {emotion} ({emotion_confidence:.1f}%)',
                                image_filename
                            ))
                            db_conn.commit()
                            thread_logger.info(f"Logged unknown face interaction with emotion {emotion}")

                            # Send webhook notification using the event loop
                            webhook_data = {
                                'timestamp': current_timestamp,
                                'image': image_filename,
                                'confidence': float(emotion_confidence) if emotion_confidence else 0.0,
                                'sentiment': emotion or 'unknown',
                                'category': 'unknown'
                            }
                            loop.run_until_complete(self.send_webhook_notification('unknown', webhook_data))

                        except Exception as e:
                            thread_logger.error(f"Database/webhook logging failed for unknown face: {str(e)}")

                except Exception as e:
                    thread_logger.error(f"Face processing error: {str(e)}\n{traceback.format_exc()}")
                    continue

        except Exception as e:
            thread_logger.error(f"Worker thread error: {str(e)}")
        finally:
            # Clean up the event loop
            loop.close()


    def get_recent_interactions(self, limit=10):
        """Get recent interactions from database"""
        try:
            conn = sqlite3.connect('greeter.db')
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp, person_name, category, sentiment, confidence, action
                FROM interactions
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            return cursor.fetchall()
        except Exception as e:
            self.logger.error(f"Failed to get recent interactions: {str(e)}")
            return []

    def get_person_history(self, person_name, limit=10):
        """Get interaction history for specific person"""
        try:
            conn = sqlite3.connect('greeter.db')
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp, category, sentiment, confidence, action
                FROM interactions
                WHERE person_name = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (person_name, limit))
            return cursor.fetchall()
        except Exception as e:
            self.logger.error(f"Failed to get person history: {str(e)}")
            return []

    async def start_background_processing(self):
        """Start continuous background processing"""
        if not self.processing_task:
            self.processing_task = asyncio.create_task(self._continuous_processing())
            self.logger.info("Started background processing task")

    async def _continuous_processing(self):
        """Continuous processing loop that runs regardless of websocket connections"""
        self.logger.info("Starting continuous frame processing")
        while self.should_process:
            try:
                success, frame = self.get_frame()
                if success and frame is not None:
                    # Store the latest processed frame for websocket clients
                    with self.frame_lock:
                        _, buffer = cv2.imencode('.jpg', frame)
                        self.latest_processed_frame = {
                            "type": "frame",
                            "data": base64.b64encode(buffer).decode('utf-8'),
                            "status": self.get_status(),
                            "timestamp": time.time()
                        }

                await asyncio.sleep(0.033)  # ~30 FPS

            except Exception as e:
                self.logger.error(f"Background processing error: {e}")
                await asyncio.sleep(1)  # Wait before retrying

    def _webhook_worker(self):
        """Background worker for webhook processing"""
        thread_logger = logging.getLogger("greeter.webhook_worker")
        thread_logger.info("Webhook worker started")

        # Create an event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while True:
            try:
                webhook_data = self.webhook_queue.get()
                thread_logger.debug(f"Processing webhook: {webhook_data}")

                # Run the webhook notification in the event loop
                loop.run_until_complete(
                    self.send_webhook_notification(
                        webhook_data['type'],
                        webhook_data['data']
                    )
                )

            except Exception as e:
                thread_logger.error(f"Webhook worker error: {str(e)}")
                thread_logger.error(traceback.format_exc())
            finally:
                # Ensure we mark the task as done even if it fails
                self.webhook_queue.task_done()

            # Small sleep to prevent tight loop
            time.sleep(0.001)

    def _detection_worker(self):
        """Background thread for processing detections"""
        while True:
            try:
                frame_data = self.detection_queue.get()
                frame = frame_data['frame']

                # Run detection in separate thread
                try:
                    results = self.detector(frame)[0]
                    with self.frame_lock:
                        self.last_results = results
                except Exception as e:
                    self.logger.error(f"Detection failed: {str(e)}")

            except Exception as e:
                self.logger.error(f"Detection worker error: {str(e)}")
            finally:
                self.detection_queue.task_done()
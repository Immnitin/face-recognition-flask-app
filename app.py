# # app.py - Fixed version with proper error handling
# from flask import Flask, render_template, request, jsonify
# from flask_cors import CORS 
# import cv2
# import os
# import numpy as np
# import pandas as pd
# from datetime import datetime
# import base64
# from deepface import DeepFace
# import mediapipe as mp
# import traceback  # Added for debugging

# # --- CONFIGURATION ---
# SAVE_RAW = True
# DATA_DIR = "data"
# EMB_DIR = "embeddings"
# MODEL_NAME = "Facenet"
# BLUR_THRESH = 35.0
# EXPOSURE_MIN, EXPOSURE_MAX = 40, 220
# MIN_INTER_OCULAR_PX = 45

# # --- INITIALIZATION ---
# app = Flask(__name__)
# CORS(app)

# os.makedirs(DATA_DIR, exist_ok=True)
# os.makedirs(EMB_DIR, exist_ok=True)

# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(
#     static_image_mode=True, 
#     max_num_faces=1, 
#     refine_landmarks=True,
#     min_detection_confidence=0.6
# )

# # --- HELPER & ANALYSIS FUNCTIONS ---
# LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
# RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
# UPPER_LIP_IDX = [13]
# LOWER_LIP_IDX = [14]
# MOUTH_RIGHT_CORNER_IDX = 291
# MOUTH_LEFT_CORNER_IDX = 61
# NOSE_TIP_IDX = 1

# def landmark_to_xy(landmark, frame_w, frame_h):
#     return int(landmark.x * frame_w), int(landmark.y * frame_h)

# def euclidian_distance(p1, p2):
#     return np.hypot(p1[0] - p2[0], p1[1] - p2[1])


# def safe_landmark_access(landmarks, indices, frame_w, frame_h):
#     """Safely access landmarks with bounds checking"""
#     try:
#         if len(landmarks) <= max(indices):
#             return None
#         return [landmark_to_xy(landmarks[i], frame_w, frame_h) for i in indices]
#     except (IndexError, AttributeError):
#         return None

# def eye_aspect_ratio(landmarks, eye_idx, frame_w, frame_h):
#     try:
#         pts = safe_landmark_access(landmarks, eye_idx, frame_w, frame_h)
#         if pts is None or len(pts) < 6:
#             return 0.25  # Default "open" value
#         (x1,y1),(x2,y2),(x3,y3),(x4,y4),(x5,y5),(x6,y6) = pts
#         ear = (euclidian_distance((x2,y2),(x6,y6)) + euclidian_distance((x3,y3),(x5,y5))) / \
#               (2.0 * max(1e-6, euclidian_distance((x1,y1),(x4,y4))))
#         return max(0.0, min(1.0, ear))  # Clamp between 0 and 1
#     except Exception:
#         return 0.25

# def mouth_open_ratio(landmarks, frame_w, frame_h):
#     try:
#         if len(landmarks) <= max(UPPER_LIP_IDX + LOWER_LIP_IDX + [MOUTH_LEFT_CORNER_IDX, MOUTH_RIGHT_CORNER_IDX]):
#             return 0.0
#         top_lip = landmark_to_xy(landmarks[13], frame_w, frame_h)
#         bottom_lip = landmark_to_xy(landmarks[14], frame_w, frame_h)
#         mouth_left = landmark_to_xy(landmarks[MOUTH_LEFT_CORNER_IDX], frame_w, frame_h)
#         mouth_right = landmark_to_xy(landmarks[MOUTH_RIGHT_CORNER_IDX], frame_w, frame_h)
#         vertical_dist = euclidian_distance(top_lip, bottom_lip)
#         horizontal_dist = euclidian_distance(mouth_left, mouth_right)
#         return vertical_dist / max(1e-6, horizontal_dist)
#     except Exception:
#         return 0.0

# def head_yaw_normalized(landmarks, frame_w, frame_h, interocular_dist):
#     try:
#         if len(landmarks) <= max([NOSE_TIP_IDX] + LEFT_EYE_IDX + RIGHT_EYE_IDX):
#             return 0.0
#         nose = landmark_to_xy(landmarks[NOSE_TIP_IDX], frame_w, frame_h)
#         left_pts = safe_landmark_access(landmarks, LEFT_EYE_IDX, frame_w, frame_h)
#         right_pts = safe_landmark_access(landmarks, RIGHT_EYE_IDX, frame_w, frame_h)
#         if left_pts is None or right_pts is None:
#             return 0.0
#         left_eye_center = np.mean(left_pts, axis=0)
#         right_eye_center = np.mean(right_pts, axis=0)
#         eye_center = (left_eye_center + right_eye_center) / 2.0
#         return (nose[0] - eye_center[0]) / max(1e-6, interocular_dist)
#     except Exception:
#         return 0.0

# def quality_checks(face_crop):
#     if face_crop is None or face_crop.size == 0: 
#         return False, "empty_crop"
#     try:
#         gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
#         if cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESH: 
#             return False, "blurry"
#         if not (EXPOSURE_MIN <= gray.mean() <= EXPOSURE_MAX): 
#             return False, "bad_exposure"
#         return True, "ok"
#     except Exception: 
#         return False, "qc_error"

# def compute_embedding(face_crop):
#     try:
#         rep = DeepFace.represent(face_crop, model_name=MODEL_NAME, enforce_detection=False)
#         if isinstance(rep, list) and len(rep) > 0:
#             emb = np.array(rep[0]["embedding"], dtype=np.float32)
#             return emb / np.linalg.norm(emb)
#         return None
#     except Exception: 
#         return None

# def save_embedding_record(user_id, embedding, meta):
#     try:
#         csv_path = os.path.join(EMB_DIR, "embeddings.csv")
#         record = meta.copy()
#         record["embedding"] = embedding.tolist()
#         df = pd.DataFrame([record])
#         df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)
#         return True
#     except Exception: 
#         return False

# # --- FLASK ROUTES ---

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/validate_frame', methods=['POST'])
# def validate_frame():
#     try:
#         # Parse request data
#         data = request.get_json()
#         if not data or 'image' not in data or 'prompt' not in data:
#             return jsonify({"status": "CONTINUE", "feedback": "INVALID_REQUEST"}), 400

#         # Decode image
#         try:
#             image_data = base64.b64decode(data['image'].split(',')[1])
#             nparr = np.frombuffer(image_data, np.uint8)
#             frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         except Exception as e:
#             print(f"Image decode error: {e}")
#             return jsonify({"status": "CONTINUE", "feedback": "IMAGE_DECODE_ERROR"})
        
#         if frame is None:
#             return jsonify({"status": "CONTINUE", "feedback": "INVALID_FRAME"})

#         img_h, img_w, _ = frame.shape
#         if img_h == 0 or img_w == 0:
#             return jsonify({"status": "CONTINUE", "feedback": "INVALID_FRAME_SIZE"})

#         # Process with MediaPipe
#         try:
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = face_mesh.process(rgb_frame)
#         except Exception as e:
#             print(f"MediaPipe processing error: {e}")
#             return jsonify({"status": "CONTINUE", "feedback": "PROCESSING_ERROR"})

#         if not results.multi_face_landmarks:
#             return jsonify({"status": "CONTINUE", "feedback": "NO_FACE_DETECTED"})

#         landmarks = results.multi_face_landmarks[0].landmark
        
#         # Check if we have enough landmarks
#         if len(landmarks) < 468:  # MediaPipe face mesh should have 468 landmarks
#             return jsonify({"status": "CONTINUE", "feedback": "INCOMPLETE_LANDMARKS"})

#         # Calculate interocular distance safely
#         try:
#             left_pts = safe_landmark_access(landmarks, LEFT_EYE_IDX, img_w, img_h)
#             right_pts = safe_landmark_access(landmarks, RIGHT_EYE_IDX, img_w, img_h)
#             if left_pts is None or right_pts is None:
#                 return jsonify({"status": "CONTINUE", "feedback": "LANDMARK_ACCESS_ERROR"})
            
#             left_center = np.mean(left_pts, axis=0)
#             right_center = np.mean(right_pts, axis=0)
#             interocular = euclidian_distance(left_center, right_center)
#         except Exception as e:
#             print(f"Interocular calculation error: {e}")
#             return jsonify({"status": "CONTINUE", "feedback": "CALCULATION_ERROR"})
        
#         if interocular < MIN_INTER_OCULAR_PX:
#             return jsonify({"status": "CONTINUE", "feedback": "MOVE_CLOSER"})

#         # Quality checks
#         qc_pass, qc_reason = quality_checks(frame)
#         if not qc_pass:
#             return jsonify({"status": "CONTINUE", "feedback": qc_reason.upper()})

#         # Calculate features safely
#         prompt_tag = data['prompt']
#         ear = eye_aspect_ratio(landmarks, LEFT_EYE_IDX, img_w, img_h) * 0.5 + \
#               eye_aspect_ratio(landmarks, RIGHT_EYE_IDX, img_w, img_h) * 0.5
#         yaw = head_yaw_normalized(landmarks, img_w, img_h, interocular)
#         mratio = mouth_open_ratio(landmarks, img_w, img_h)

#         # Action detection with more lenient thresholds
#         action_ok = False
#         feedback = "CONTINUE"

#         if prompt_tag == 'neutral':
#             if abs(yaw) < 0.08 and ear > 0.20: 
#                 action_ok = True
#             else: 
#                 feedback = "LOOK_STRAIGHT"
#         elif prompt_tag == 'left':
#             if yaw < -0.06: 
#                 action_ok = True
#             else: 
#                 feedback = "TURN_MORE_LEFT"
#         elif prompt_tag == 'right':
#             if yaw > 0.06: 
#                 action_ok = True
#             else: 
#                 feedback = "TURN_MORE_RIGHT"
#         elif prompt_tag == 'blink':
#             if ear < 0.18: 
#                 feedback = "EYES_CLOSED"
#             else: 
#                 feedback = "BLINK_NOW"

#         if action_ok:
#             return jsonify({"status": "ACTION_OK"})
#         else:
#             return jsonify({"status": "CONTINUE", "feedback": feedback})
            
#     except Exception as e:
#         print(f"Error in /validate_frame: {e}")
#         print(traceback.format_exc())  # Print full stack trace for debugging
#         return jsonify({"status": "ERROR", "feedback": "SERVER_ERROR"}), 500

# @app.route('/enroll', methods=['POST'])
# def enroll():
#     try:
#         data = request.get_json()
#         if not data or 'userId' not in data or 'prompt' not in data or 'image' not in data:
#             return jsonify({"status": "error", "message": "Invalid request data"}), 400

#         user_id = data['userId']
#         prompt_tag = data['prompt']
        
#         try:
#             image_data = base64.b64decode(data['image'].split(',')[1])
#             nparr = np.frombuffer(image_data, np.uint8)
#             img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         except Exception as e:
#             print(f"Image decode error in enroll: {e}")
#             return jsonify({"status": "error", "message": "Image decode failed"})

#         if img is None:
#             return jsonify({"status": "error", "message": "Invalid image"})

#         embedding = compute_embedding(img)
#         if embedding is None:
#             return jsonify({"status": "error", "message": "Embedding failed"})

#         if SAVE_RAW:
#             timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
#             filename = f"{user_id}_{prompt_tag}_{timestamp}.jpg"
#             cv2.imwrite(os.path.join(DATA_DIR, filename), img)

#         meta = {
#             "user_id": user_id, 
#             "prompt": prompt_tag, 
#             "timestamp": datetime.utcnow().isoformat()
#         }
#         save_embedding_record(user_id, embedding, meta)
        
#         return jsonify({"status": "success"})
        
#     except Exception as e:
#         print(f"Error in /enroll: {e}")
#         print(traceback.format_exc())
#         return jsonify({"status": "error", "message": str(e)}), 500

# def preload_model():
#     """
#     Forces the DeepFace model to load into memory on server startup.
#     This prevents a long delay on the first /enroll request.
#     """
#     print("Pre-loading DeepFace model, this may take a moment...")
#     try:
#         # Create a small, black dummy image
#         dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
#         # This call will trigger the model download (if needed) and loading
#         compute_embedding(dummy_image)
#         print("✅ DeepFace model pre-loaded successfully.")
#     except Exception as e:
#         print(f"❗️ Error pre-loading model: {e}")
#         print("The application will still run, but the first enrollment may be slow.")

# # <<< --- MODIFY THIS FINAL BLOCK --- >>>
# if __name__ == '__main__':
#     # Call the pre-load function before starting the app
#     preload_model() 
    
#     # Now, run the Flask app
#     app.run(host='0.0.0.0', port=5000, debug=True)

# app.py - Fully Corrected Version


# app.py - Enhanced Version with Gallery Support
import os
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS 
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import base64
from deepface import DeepFace
import mediapipe as mp
import traceback
import glob

# --- CONFIGURATION ---
SAVE_RAW = True
DATA_DIR = "data"
EMB_DIR = "embeddings"
MODEL_NAME = "Facenet"
BLUR_THRESH = 35.0
EXPOSURE_MIN, EXPOSURE_MAX = 40, 220
MIN_INTER_OCULAR_PX = 45

# --- INITIALIZATION ---
app = Flask(__name__)
CORS(app)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EMB_DIR, exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, 
    max_num_faces=1, 
    refine_landmarks=True,
    min_detection_confidence=0.6
)

# --- HELPER & ANALYSIS FUNCTIONS ---
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
NOSE_TIP_IDX = 1

def landmark_to_xy(landmark, frame_w, frame_h):
    return int(landmark.x * frame_w), int(landmark.y * frame_h)

def euclidian_distance(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

def safe_landmark_access(landmarks, indices, frame_w, frame_h):
    try:
        if len(landmarks) <= max(indices):
            return None
        return [landmark_to_xy(landmarks[i], frame_w, frame_h) for i in indices]
    except (IndexError, AttributeError):
        return None

def eye_aspect_ratio(landmarks, eye_idx, frame_w, frame_h):
    try:
        pts = safe_landmark_access(landmarks, eye_idx, frame_w, frame_h)
        if pts is None or len(pts) < 6:
            return 0.25
        (x1,y1),(x2,y2),(x3,y3),(x4,y4),(x5,y5),(x6,y6) = pts
        ear = (euclidian_distance((x2,y2),(x6,y6)) + euclidian_distance((x3,y3),(x5,y5))) / \
              (2.0 * max(1e-6, euclidian_distance((x1,y1),(x4,y4))))
        return max(0.0, min(1.0, ear))
    except Exception:
        return 0.25

def mouth_open_ratio(landmarks, frame_w, frame_h):
    try:
        top_lip = landmark_to_xy(landmarks[13], frame_w, frame_h)
        bottom_lip = landmark_to_xy(landmarks, frame_w, frame_h)
        mouth_left = landmark_to_xy(landmarks, frame_w, frame_h)
        mouth_right = landmark_to_xy(landmarks, frame_w, frame_h)
        vertical_dist = euclidian_distance(top_lip, bottom_lip)
        horizontal_dist = euclidian_distance(mouth_left, mouth_right)
        return vertical_dist / max(1e-6, horizontal_dist)
    except Exception:
        return 0.0

def head_yaw_normalized(landmarks, frame_w, frame_h, interocular_dist):
    try:
        nose = landmark_to_xy(landmarks[NOSE_TIP_IDX], frame_w, frame_h)
        left_pts = safe_landmark_access(landmarks, LEFT_EYE_IDX, frame_w, frame_h)
        right_pts = safe_landmark_access(landmarks, RIGHT_EYE_IDX, frame_w, frame_h)
        if left_pts is None or right_pts is None:
            return 0.0
        left_eye_center = np.mean(left_pts, axis=0)
        right_eye_center = np.mean(right_pts, axis=0)
        eye_center = (left_eye_center + right_eye_center) / 2.0
        return (nose[0] - eye_center[0]) / max(1e-6, interocular_dist)
    except Exception:
        return 0.0

def quality_checks(face_crop):
    if face_crop is None or face_crop.size == 0: 
        return False, "empty_crop"
    try:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESH: 
            return False, "blurry"
        if not (EXPOSURE_MIN <= gray.mean() <= EXPOSURE_MAX): 
            return False, "bad_exposure"
        return True, "ok"
    except Exception: 
        return False, "qc_error"

def compute_embedding(face_crop):
    try:
        rep = DeepFace.represent(face_crop, model_name=MODEL_NAME, enforce_detection=False)
        if isinstance(rep, list) and len(rep) > 0:
            emb = np.array(rep[0]["embedding"], dtype=np.float32)
            return emb / np.linalg.norm(emb)
        return None
    except Exception: 
        return None

def save_embedding_record(user_id, embedding, meta):
    try:
        csv_path = os.path.join(EMB_DIR, "embeddings.csv")
        record = meta.copy()
        record["embedding"] = embedding.tolist()
        
        column_order = [
            "user_id", "prompt", "timestamp", 
            "interocular_px", "ear", "mouth_ratio", "embedding"
        ]
        
        df = pd.DataFrame([record])
        
        for col in column_order:
            if col not in df.columns:
                df[col] = None
        
        df = df[column_order]
        df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)
        
        npz_path = os.path.join(EMB_DIR, f"{user_id}.npz")
        if os.path.exists(npz_path):
            old_data = np.load(npz_path, allow_pickle=True)
            embs = list(old_data.get("embs", []))
            metas = list(old_data.get("metas", []))
            embs.append(embedding)
            metas.append(meta)
            np.savez(npz_path, embs=embs, metas=metas)
        else:
            np.savez(npz_path, embs=[embedding], metas=[meta])
            
        return True
    except Exception as e:
        print(f"Error in save_embedding_record: {e}")
        print(traceback.format_exc())
        return False

# --- FLASK ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_user', methods=['POST'])
def check_user():
    try:
        data = request.get_json()
        user_id = data.get('userId')
        if not user_id:
            return jsonify({"exists": False, "message": "User ID is required"}), 400

        csv_path = os.path.join(EMB_DIR, "embeddings.csv")
        if not os.path.exists(csv_path):
            return jsonify({"exists": False})

        df = pd.read_csv(csv_path)
        if user_id in df['user_id'].values:
            return jsonify({"exists": True})
        else:
            return jsonify({"exists": False})
            
    except Exception as e:
        print(f"Error in /check_user: {e}")
        return jsonify({"error": "An internal error occurred"}), 500

@app.route('/get_gallery', methods=['POST'])
def get_gallery():
    try:
        data = request.get_json()
        user_id = data.get('userId')
        if not user_id:
            return jsonify({"images": []}), 400

        # Find all image files for this user
        pattern = os.path.join(DATA_DIR, f"{user_id}_*.jpg")
        image_files = glob.glob(pattern)
        
        images = []
        prompt_names = {
            'neutral': 'Look straight (neutral)',
            'left': 'Turn head left',
            'right': 'Turn head right',
            'blink': 'Blink twice'
        }
        
        for img_path in sorted(image_files):
            filename = os.path.basename(img_path)
            # Extract prompt from filename (format: userid_prompt_timestamp.jpg)
            parts = filename.replace('.jpg', '').split('_')
            if len(parts) >= 2:
                prompt = parts[1]
                title = prompt_names.get(prompt, prompt.title())
                images.append({
                    'title': title,
                    'url': f'/image/{filename}',
                    'prompt': prompt,
                    'filename': filename
                })
        
        return jsonify({"images": images})
        
    except Exception as e:
        print(f"Error in /get_gallery: {e}")
        return jsonify({"error": "An internal error occurred"}), 500

@app.route('/image/<filename>')
def serve_image(filename):
    """Serve images from the data directory"""
    try:
        image_path = os.path.join(DATA_DIR, filename)
        if os.path.exists(image_path):
            return send_file(image_path)
        else:
            return "Image not found", 404
    except Exception as e:
        print(f"Error serving image {filename}: {e}")
        return "Error serving image", 500

@app.route('/validate_frame', methods=['POST'])
def validate_frame():
    try:
        data = request.get_json()
        if not data or 'image' not in data or 'prompt' not in data:
            return jsonify({"status": "CONTINUE", "feedback": "INVALID_REQUEST"}), 400
        
        image_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"status": "CONTINUE", "feedback": "INVALID_FRAME"})

        img_h, img_w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return jsonify({"status": "CONTINUE", "feedback": "NO_FACE_DETECTED"})

        landmarks = results.multi_face_landmarks[0].landmark
        
        left_pts = safe_landmark_access(landmarks, LEFT_EYE_IDX, img_w, img_h)
        right_pts = safe_landmark_access(landmarks, RIGHT_EYE_IDX, img_w, img_h)
        if left_pts is None or right_pts is None:
            return jsonify({"status": "CONTINUE", "feedback": "LANDMARK_ACCESS_ERROR"})
        
        left_center = np.mean(left_pts, axis=0)
        right_center = np.mean(right_pts, axis=0)
        interocular = euclidian_distance(left_center, right_center)
        
        if interocular < MIN_INTER_OCULAR_PX:
            return jsonify({"status": "CONTINUE", "feedback": "MOVE_CLOSER"})

        qc_pass, qc_reason = quality_checks(frame)
        if not qc_pass:
            return jsonify({"status": "CONTINUE", "feedback": qc_reason.upper()})

        prompt_tag = data['prompt']
        ear = (eye_aspect_ratio(landmarks, LEFT_EYE_IDX, img_w, img_h) + eye_aspect_ratio(landmarks, RIGHT_EYE_IDX, img_w, img_h)) / 2.0
        yaw = head_yaw_normalized(landmarks, img_w, img_h, interocular)
        mratio = mouth_open_ratio(landmarks, img_w, img_h)

        action_ok = False
        feedback = "CONTINUE"

        if prompt_tag == 'neutral':
            if abs(yaw) < 0.08 and ear > 0.20: 
                action_ok = True
            else: 
                feedback = "LOOK_STRAIGHT"
        elif prompt_tag == 'left':
            if yaw > 0.12:
                action_ok = True
            else: 
                feedback = "TURN_MORE_LEFT"
        elif prompt_tag == 'right':
            if yaw < -0.12:
                action_ok = True
            else: 
                feedback = "TURN_MORE_RIGHT"
        elif prompt_tag == 'blink':
            if ear < 0.18: 
                feedback = "EYES_CLOSED"
            else: 
                feedback = "BLINK_NOW"

        if action_ok:
            return jsonify({
                "status": "ACTION_OK",
                "feedback": "GOOD",
                "metadata": {
                    "interocular_px": round(interocular, 2),
                    "ear": round(ear, 4),
                    "mouth_ratio": round(mratio, 4)
                }
            })
        else:
            return jsonify({"status": "CONTINUE", "feedback": feedback})
            
    except Exception as e:
        print(f"Error in /validate_frame: {e}")
        print(traceback.format_exc())
        return jsonify({"status": "ERROR", "feedback": "SERVER_ERROR"}), 500

@app.route('/enroll', methods=['POST'])
def enroll():
    try:
        data = request.get_json()
        if not data or 'userId' not in data or 'prompt' not in data or 'image' not in data or 'metadata' not in data:
            return jsonify({"status": "error", "message": "Invalid request data, metadata missing"}), 400

        user_id = data['userId']
        prompt_tag = data['prompt']
        metadata_from_request = data['metadata']

        image_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"status": "error", "message": "Invalid image"})

        embedding = compute_embedding(img)
        if embedding is None:
            return jsonify({"status": "error", "message": "Embedding failed"})

        if SAVE_RAW:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{user_id}_{prompt_tag}_{timestamp}.jpg"
            cv2.imwrite(os.path.join(DATA_DIR, filename), img)

        full_meta_to_save = {
            "user_id": user_id, 
            "prompt": prompt_tag, 
            "timestamp": datetime.utcnow().isoformat(),
            "interocular_px": metadata_from_request.get("interocular_px"),
            "ear": metadata_from_request.get("ear"),
            "mouth_ratio": metadata_from_request.get("mouth_ratio")
        }
        
        save_embedding_record(user_id, embedding, full_meta_to_save)
        
        return jsonify({"status": "success"})
        
    except Exception as e:
        print(f"Error in /enroll: {e}")
        print(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500

def preload_model():
    print("Pre-loading DeepFace model, this may take a moment...")
    try:
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        compute_embedding(dummy_image)
        print("✅ DeepFace model pre-loaded successfully.")
    except Exception as e:
        print(f"❗️ Error pre-loading model: {e}")

if __name__ == '__main__':
    preload_model()
    # Use environment variable for port, default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)


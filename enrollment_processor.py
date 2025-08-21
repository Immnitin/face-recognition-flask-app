import cv2
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
import mediapipe as mp
from deepface import DeepFace

# -----------------------------------------------------------------------------
# --- All helper functions from your original enroll.py script ---
# -----------------------------------------------------------------------------

# MediaPipe landmark indices
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
UPPER_LIP_IDX = [13]
LOWER_LIP_IDX = [14]
MOUTH_RIGHT_CORNER_IDX = 291
MOUTH_LEFT_CORNER_IDX = 61
NOSE_TIP_IDX = 1

def landmark_to_xy(landmark, frame_w, frame_h):
    return int(landmark.x * frame_w), int(landmark.y * frame_h)

def euclidian_distance(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

def eye_aspect_ratio(landmarks, eye_idx, frame_w, frame_h):
    p = [landmarks[i] for i in eye_idx]
    pts = [landmark_to_xy(pt, frame_w, frame_h) for pt in p]
    (x1,y1),(x2,y2),(x3,y3),(x4,y4),(x5,y5),(x6,y6) = pts
    ear = (euclidian_distance((x2,y2),(x6,y6)) + euclidian_distance((x3,y3),(x5,y5))) / \
          (2.0 * max(1e-6, euclidian_distance((x1,y1),(x4,y4))))
    return ear

def mouth_open_ratio(landmarks, frame_w, frame_h):
    top_lip = landmark_to_xy(landmarks[UPPER_LIP_IDX[0]], frame_w, frame_h)
    bottom_lip = landmark_to_xy(landmarks[LOWER_LIP_IDX[0]], frame_w, frame_h)
    mouth_left = landmark_to_xy(landmarks[MOUTH_LEFT_CORNER_IDX], frame_w, frame_h)
    mouth_right = landmark_to_xy(landmarks[MOUTH_RIGHT_CORNER_IDX], frame_w, frame_h)
    vertical_dist = euclidian_distance(top_lip, bottom_lip)
    horizontal_dist = euclidian_distance(mouth_left, mouth_right)
    return vertical_dist / max(1e-6, horizontal_dist)

def head_yaw_normalized(landmarks, frame_w, frame_h, interocular_dist):
    nose = landmark_to_xy(landmarks[NOSE_TIP_IDX], frame_w, frame_h)
    left_eye_center = np.mean([landmark_to_xy(landmarks[i], frame_w, frame_h) for i in LEFT_EYE_IDX], axis=0)
    right_eye_center = np.mean([landmark_to_xy(landmarks[i], frame_w, frame_h) for i in RIGHT_EYE_IDX], axis=0)
    eye_center = (left_eye_center + right_eye_center) / 2.0
    return (nose[0] - eye_center[0]) / max(1e-6, interocular_dist)

def crop_face_from_bbox(frame, bbox):
    x1,y1,x2,y2 = bbox
    h,w = frame.shape[:2]
    x1, x2 = max(0,x1), min(w,x2)
    y1, y2 = max(0,y1), min(h,y2)
    return frame[y1:y2, x1:x2]

def quality_checks(face_crop, blur_thresh, exposure_min, exposure_max):
    if face_crop.size == 0:
        return False, "empty_crop"
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_val < blur_thresh:
        return False, "blurry"
    mean_pix = int(gray.mean())
    if not (exposure_min <= mean_pix <= exposure_max):
        return False, "bad_exposure"
    return True, ""

def get_face_bbox_from_landmarks(landmarks, frame_w, frame_h, padding=1.4):
    xs = [int(p.x * frame_w) for p in landmarks]
    ys = [int(p.y * frame_h) for p in landmarks]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w//2, y1 + h//2
    new_w, new_h = int(w*padding), int(h*padding)
    x1 = cx - new_w//2
    x2 = cx + new_w//2
    y1 = cy - new_h//2
    y2 = cy + new_h//2
    return int(x1), int(y1), int(x2), int(y2)

def compute_embedding(face_crop, model_name):
    try:
        rep = DeepFace.represent(face_crop, model_name=model_name, enforce_detection=False)
        if isinstance(rep, list) and len(rep) > 0:
            emb = rep[0]["embedding"]
        else:
            return None
        emb = np.array(emb, dtype=np.float32)
        emb = emb / np.linalg.norm(emb)
        return emb
    except Exception:
        return None

def save_embedding_record(user_id, embedding, meta, emb_dir):
    csv_path = os.path.join(emb_dir, "embeddings.csv")
    record = meta.copy()
    record["embedding"] = embedding.tolist()
    df = pd.DataFrame([record])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)
    npz_path = os.path.join(emb_dir, f"{user_id}.npz")
    if os.path.exists(npz_path):
        old_data = np.load(npz_path, allow_pickle=True)
        embs = list(old_data["embs"])
        metas = list(old_data["metas"])
        embs.append(embedding)
        metas.append(meta)
        np.savez(npz_path, embs=embs, metas=metas)
    else:
        np.savez(npz_path, embs=[embedding], metas=[meta])

# -----------------------------------------------------------------------------
# --- Main Enrollment Processor Class ---
# -----------------------------------------------------------------------------
class EnrollmentProcessor:
    def __init__(self, user_id):
        self.user_id = user_id

        # --- CONFIG ---
        self.SAVE_RAW = True
        self.DATA_DIR = "data"
        self.EMB_DIR = "embeddings"
        self.MODEL_NAME = "Facenet"
        self.CAPTURE_COOLDOWN = 1.5
        self.MIN_INTER_OCULAR_PX = 60
        self.BLUR_THRESH = 80.0
        self.EXPOSURE_MIN, self.EXPOSURE_MAX = 50, 205
        # --- End CONFIG ---

        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.EMB_DIR, exist_ok=True)

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.7, min_tracking_confidence=0.7
        )

        self.prompts = [
            {"text": "Look straight", "tag": "neutral"},
            {"text": "Turn head LEFT", "tag": "left"},
            {"text": "Turn head RIGHT", "tag": "right"},
            {"text": "Blink twice", "tag": "blink"},
            {"text": "Smile widely", "tag": "smile"},
        ]
        self.prompt_index = 0
        self.last_capture_time = 0
        self.collected_images_count = 0
        self.is_finished = False

        # State variables from your script
        self.EYE_CLOSED_THRESHOLD = 0.15
        self.EYE_OPEN_THRESHOLD = 0.25
        self.blink_state = "OPEN"
        self.blink_count = 0
        self.neutral_mouth_width = 0

    def get_current_prompt(self):
        if self.is_finished:
            return "Enrollment Complete! Thank you."
        return self.prompts[self.prompt_index]["text"]

    def process_frame(self, frame):
        if self.is_finished:
            return {"status": "✅ Enrollment Complete!", "finished": True}

        prompt_data = self.prompts[self.prompt_index]
        prompt_tag = prompt_data["tag"]

        feedback = {"status": self.get_current_prompt(), "finished": False}
        action_ok = False

        img_h, img_w = frame.shape[:2]
        # Flip the frame for a more natural mirror-like view
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            bbox = get_face_bbox_from_landmarks(landmarks, img_w, img_h, padding=1.5)
            # Use original (unflipped) frame for cropping to save correct orientation
            face_crop = crop_face_from_bbox(cv2.flip(frame, 1), bbox)

            left_center = np.mean([landmark_to_xy(landmarks[i], img_w, img_h) for i in LEFT_EYE_IDX], axis=0)
            right_center = np.mean([landmark_to_xy(landmarks[i], img_w, img_h) for i in RIGHT_EYE_IDX], axis=0)
            interocular = euclidian_distance(left_center, right_center)

            if interocular < self.MIN_INTER_OCULAR_PX:
                feedback["status"] = "MOVE CLOSER"
                return feedback

            # --- ✅ This is the logic adapted from your enroll.py script ---
            ear = (eye_aspect_ratio(landmarks, LEFT_EYE_IDX, img_w, img_h) + eye_aspect_ratio(landmarks, RIGHT_EYE_IDX, img_w, img_h)) / 2.0
            mratio = mouth_open_ratio(landmarks, img_w, img_h)
            normalized_yaw = head_yaw_normalized(landmarks, img_w, img_h, interocular)

            if prompt_tag == "neutral":
                if abs(normalized_yaw) < 0.05 and ear > self.EYE_OPEN_THRESHOLD and mratio < 0.01:
                    action_ok = True
                    feedback["status"] = "GOOD! HOLD IT!"
                    if self.neutral_mouth_width == 0:
                        mouth_left_xy = landmark_to_xy(landmarks[MOUTH_LEFT_CORNER_IDX], img_w, img_h)
                        mouth_right_xy = landmark_to_xy(landmarks[MOUTH_RIGHT_CORNER_IDX], img_w, img_h)
                        self.neutral_mouth_width = euclidian_distance(mouth_left_xy, mouth_right_xy)
                else:
                    feedback["status"] = "Look straight at the camera"

            elif prompt_tag == "left":
                if normalized_yaw > 0.10: # Flipped view means user turning left moves nose to image right (+)
                    action_ok = True
                    feedback["status"] = "GOOD! HOLD IT!"
                else:
                    feedback["status"] = "Turn head LEFT"

            elif prompt_tag == "right":
                if normalized_yaw < -0.10: # Flipped view means user turning right moves nose to image left (-)
                    action_ok = True
                    feedback["status"] = "GOOD! HOLD IT!"
                else:
                    feedback["status"] = "Turn head RIGHT"

            elif prompt_tag == "blink":
                if self.blink_state == "OPEN" and ear < self.EYE_CLOSED_THRESHOLD:
                    self.blink_state = "CLOSED"
                elif self.blink_state == "CLOSED" and ear > self.EYE_OPEN_THRESHOLD:
                    self.blink_state = "OPEN"
                    self.blink_count += 1
                
                if self.blink_count >= 2:
                    action_ok = True
                    feedback["status"] = "GOOD! Blinks detected!"
                else:
                    feedback["status"] = f"Blink now ({self.blink_count}/2)"

            elif prompt_tag == "smile":
                if self.neutral_mouth_width == 0:
                     feedback["status"] = "Please wait, calibrating..."
                else:
                    current_mouth_width = euclidian_distance(landmark_to_xy(landmarks[MOUTH_LEFT_CORNER_IDX], img_w, img_h), landmark_to_xy(landmarks[MOUTH_RIGHT_CORNER_IDX], img_w, img_h))
                    width_increase_ratio = current_mouth_width / max(1e-6, self.neutral_mouth_width)
                    if mratio > 0.035 and width_increase_ratio > 1.08:
                        action_ok = True
                        feedback["status"] = "GREAT SMILE! HOLD IT!"
                    else:
                        feedback["status"] = "Smile widely!"
            
            # --- End of adapted logic block ---

            if action_ok and (time.time() - self.last_capture_time) > self.CAPTURE_COOLDOWN:
                qc_pass, qc_reason = quality_checks(face_crop, self.BLUR_THRESH, self.EXPOSURE_MIN, self.EXPOSURE_MAX)
                if qc_pass:
                    timestamp = datetime.utcnow().isoformat()
                    emb = compute_embedding(face_crop, self.MODEL_NAME)
                    if emb is not None:
                        if self.SAVE_RAW:
                            fname = os.path.join(self.DATA_DIR, f"{self.user_id}_{prompt_tag}_{self.collected_images_count}.jpg")
                            cv2.imwrite(fname, face_crop)
                        
                        meta = {"user_id": self.user_id, "prompt": prompt_tag, "timestamp": timestamp}
                        save_embedding_record(self.user_id, emb, meta, self.EMB_DIR)
                        
                        self.last_capture_time = time.time()
                        self.collected_images_count += 1
                        
                        if prompt_tag == "blink": # Reset blink state for next time
                            self.blink_count = 0
                            self.blink_state = "OPEN"

                        self.prompt_index += 1
                        if self.prompt_index >= len(self.prompts):
                            self.is_finished = True
                            feedback["finished"] = True
                            feedback["status"] = "✅ Enrollment Complete! Thank you."
                        else:
                            feedback["status"] = f"✅ Captured! Next: {self.get_current_prompt()}"
                    else:
                        feedback["status"] = "Could not create embedding. Try again."
                else:
                    feedback["status"] = f"REJECTED: {qc_reason.upper()}"
        else:
            feedback["status"] = "No face detected"
            
        return feedback
# enroll.py (Corrected)
# A standalone script for guided face enrollment with liveness checks.
# This script uses OpenCV and MediaPipe for real-time processing.
#
# To run this:
# 1. Install dependencies:
#    pip install opencv-python mediapipe pandas deepface
# 2. Run from the command line with a user ID:
#    python enroll.py your_user_id

import cv2
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
import mediapipe as mp
from deepface import DeepFace
import argparse

# ---------------- CONFIG (Tuned Parameters) ----------------
SAVE_RAW = True
DATA_DIR = "data"
EMB_DIR = "embeddings"
MODEL_NAME = "Facenet"
CAMERA_INDEX = 0
CAPTURE_COOLDOWN = 1.2
MIN_INTER_OCULAR_PX = 45  # Lowered for farther distances
BLUR_THRESH = 30.0  # Much more lenient
EXPOSURE_MIN, EXPOSURE_MAX = 40, 220  # Wider range
# -----------------------------------------------------------

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EMB_DIR, exist_ok=True)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# MediaPipe landmark indices
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
UPPER_LIP_IDX = [13]
LOWER_LIP_IDX = [14]
MOUTH_RIGHT_CORNER_IDX = 291
MOUTH_LEFT_CORNER_IDX = 61
NOSE_TIP_IDX = 1

def landmark_to_xy(landmark, frame_w, frame_h):
    """Converts a MediaPipe landmark to (x, y) pixel coordinates."""
    return int(landmark.x * frame_w), int(landmark.y * frame_h)

def euclidian_distance(p1, p2):
    """Calculates the Euclidean distance between two 2D points."""
    return np.hypot(p1[0] - p2[0], p1 - p2)

def eye_aspect_ratio(landmarks, eye_idx, frame_w, frame_h):
    """Calculates the Eye Aspect Ratio (EAR)."""
    p = [landmarks[i] for i in eye_idx]
    pts = [landmark_to_xy(pt, frame_w, frame_h) for pt in p]
    (x1,y1),(x2,y2),(x3,y3),(x4,y4),(x5,y5),(x6,y6) = pts
    ear = (euclidian_distance((x2,y2),(x6,y6)) + euclidian_distance((x3,y3),(x5,y5))) / \
          (2.0 * max(1e-6, euclidian_distance((x1,y1),(x4,y4))))
    return ear

def mouth_open_ratio(landmarks, frame_w, frame_h):
    """Calculates a mouth openness ratio."""
    top_lip = landmark_to_xy(landmarks[UPPER_LIP_IDX[0]], frame_w, frame_h)
    bottom_lip = landmark_to_xy(landmarks[LOWER_LIP_IDX], frame_w, frame_h)
    mouth_left = landmark_to_xy(landmarks[MOUTH_LEFT_CORNER_IDX], frame_w, frame_h)
    mouth_right = landmark_to_xy(landmarks[MOUTH_RIGHT_CORNER_IDX], frame_w, frame_h)
    vertical_dist = euclidian_distance(top_lip, bottom_lip)
    horizontal_dist = euclidian_distance(mouth_left, mouth_right)
    return vertical_dist / max(1e-6, horizontal_dist)

def head_yaw_normalized(landmarks, frame_w, frame_h, interocular_dist):
    """Estimates crude head yaw, normalized by interocular distance."""
    nose = landmark_to_xy(landmarks[NOSE_TIP_IDX], frame_w, frame_h)
    left_eye_center = np.mean([landmark_to_xy(landmarks[i], frame_w, frame_h) for i in LEFT_EYE_IDX], axis=0)
    right_eye_center = np.mean([landmark_to_xy(landmarks[i], frame_w, frame_h) for i in RIGHT_EYE_IDX], axis=0)
    eye_center = (left_eye_center + right_eye_center) / 2.0
    return (nose[0] - eye_center) / max(1e-6, interocular_dist)

def crop_face_from_bbox(frame, bbox):
    """Crops the face region from the frame."""
    x1,y1,x2,y2 = bbox
    h,w = frame.shape[:2]
    x1, x2 = max(0,x1), min(w,x2)
    y1, y2 = max(0,y1), min(h,y2)
    return frame[y1:y2, x1:x2]

def quality_checks(face_crop):
    """Performs blur and exposure checks on the face crop."""
    if face_crop.size == 0:
        return False, "empty_crop"
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_val < BLUR_THRESH:
        return False, "blurry"
    mean_pix = int(gray.mean())
    if not (EXPOSURE_MIN <= mean_pix <= EXPOSURE_MAX):
        return False, "bad_exposure"
    return True, ""

def get_face_bbox_from_landmarks(landmarks, frame_w, frame_h, padding=1.4):
    """Calculates an expanded bounding box around the face."""
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

def compute_embedding(face_crop):
    """Computes face embedding using DeepFace."""
    try:
        rep = DeepFace.represent(face_crop, model_name=MODEL_NAME, enforce_detection=False)
        if isinstance(rep, list) and len(rep) > 0:
            emb = rep[0]["embedding"]
        else:
            return None
        emb = np.array(emb, dtype=np.float32)
        emb = emb / np.linalg.norm(emb)
        return emb
    except Exception:
        return None

def save_embedding_record(user_id, embedding, meta):
    """Saves embedding and metadata to CSV and NPZ files."""
    csv_path = os.path.join(EMB_DIR, "embeddings.csv")
    record = meta.copy()
    record["embedding"] = embedding.tolist()
    df = pd.DataFrame([record])
    df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)

    npz_path = os.path.join(EMB_DIR, f"{user_id}.npz")
    if os.path.exists(npz_path):
        old_data = np.load(npz_path, allow_pickle=True)
        embs = list(old_data["embs"])
        metas = list(old_data["metas"])
        embs.append(embedding)
        metas.append(meta)
        np.savez(npz_path, embs=embs, metas=metas)
    else:
        np.savez(npz_path, embs=[embedding], metas=[meta])

def run_enrollment(user_id):
    """Runs the guided face enrollment process."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {CAMERA_INDEX}.")
        print("CAMERA_NOT_OPENED")
        return

    with mp_face_mesh.FaceMesh(
        static_image_mode=False, 
        max_num_faces=1, 
        refine_landmarks=True,
        min_detection_confidence=0.6,  # More lenient
        min_tracking_confidence=0.6
    ) as face_mesh:

        prompts = [
            {"text": "Look straight (neutral)", "tag": "neutral", "duration": 12},
            {"text": "TURN HEAD LEFT", "tag": "left", "duration": 12},
            {"text": "TURN HEAD RIGHT", "tag": "right", "duration": 12},
            {"text": "Blink twice", "tag": "blink", "duration": 15},
            {"text": "Smile widely", "tag": "smile", "duration": 15},
        ]

        last_capture_time = 0
        collected_images_count = 0
        
        # State variables for blink detection - more lenient thresholds
        EYE_CLOSED_THRESHOLD = 0.18  # Higher threshold = easier to trigger
        EYE_OPEN_THRESHOLD = 0.22    # Lower threshold = easier to satisfy
        blink_state = "OPEN"
        blink_count = 0
        
        neutral_mouth_width = 0

        for prompt_data in prompts:
            prompt_text = prompt_data["text"]
            prompt_tag = prompt_data["tag"]
            current_prompt_duration = prompt_data["duration"]
            print(f"\n=== Prompt: {prompt_text} === (you have {current_prompt_duration}s)")
            prompt_start = time.time()
            action_captured = False

            if prompt_tag == "blink":
                blink_count = 0
                blink_state = "OPEN"

            while time.time() - prompt_start < current_prompt_duration:
                ret, frame = cap.read()
                if not ret:
                    print("CAMERA_READ_ERROR")
                    break
                
                frame = cv2.flip(frame, 1)
                img_h, img_w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                display = frame.copy()
                time_remaining = int(current_prompt_duration - (time.time() - prompt_start))
                instruction_text = f"Prompt: {prompt_text} ({time_remaining}s)"
                cv2.putText(display, instruction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                current_status_text = ""

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    bbox = get_face_bbox_from_landmarks(landmarks, img_w, img_h, padding=1.5)
                    cv2.rectangle(display, (bbox, bbox[1]), (bbox, bbox), (0, 128, 255), 2)
                    face_crop = crop_face_from_bbox(frame, bbox)

                    left_center = np.mean([landmark_to_xy(landmarks[i], img_w, img_h) for i in LEFT_EYE_IDX], axis=0)
                    right_center = np.mean([landmark_to_xy(landmarks[i], img_w, img_h) for i in RIGHT_EYE_IDX], axis=0)
                    interocular = euclidian_distance(left_center, right_center)

                    if interocular < MIN_INTER_OCULAR_PX:
                        current_status_text = "MOVE CLOSER"
                        action_ok = False
                    else:
                        ear = (eye_aspect_ratio(landmarks, LEFT_EYE_IDX, img_w, img_h) + 
                               eye_aspect_ratio(landmarks, RIGHT_EYE_IDX, img_w, img_h)) / 2.0
                        mratio = mouth_open_ratio(landmarks, img_w, img_h)
                        normalized_yaw = head_yaw_normalized(landmarks, img_w, img_h, interocular)
                        
                        cv2.putText(display, f"EAR:{ear:.2f} Yaw:{normalized_yaw:.2f}", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

                        action_ok = False
                        
                        # More lenient detection thresholds
                        if prompt_tag == "neutral":
                            if abs(normalized_yaw) < 0.08 and ear > EYE_OPEN_THRESHOLD and mratio < 0.04:
                                action_ok = True
                                current_status_text = "GOOD! HOLD IT!"
                                if neutral_mouth_width == 0:
                                    mouth_left_xy = landmark_to_xy(landmarks[MOUTH_LEFT_CORNER_IDX], img_w, img_h)
                                    mouth_right_xy = landmark_to_xy(landmarks[MOUTH_RIGHT_CORNER_IDX], img_w, img_h)
                                    neutral_mouth_width = euclidian_distance(mouth_left_xy, mouth_right_xy)
                                    if neutral_mouth_width < 10: 
                                        neutral_mouth_width = 50  # Fallback
                            else:
                                current_status_text = "LOOK STRAIGHT"
                        
                        elif prompt_tag == "left":
                            if normalized_yaw < -0.06:  # More lenient
                                action_ok = True
                                current_status_text = "GOOD! HOLD IT!"
                            else:
                                current_status_text = "TURN HEAD LEFT (MORE!)"

                        elif prompt_tag == "right":
                            if normalized_yaw > 0.06:  # More lenient
                                action_ok = True
                                current_status_text = "GOOD! HOLD IT!"
                            else:
                                current_status_text = "TURN HEAD RIGHT (MORE!)"

                        elif prompt_tag == "blink":
                            if blink_state == "OPEN" and ear < EYE_CLOSED_THRESHOLD:
                                blink_state = "CLOSED"
                                current_status_text = f"Blink {blink_count+1}/2 - Eyes Closed"
                            elif blink_state == "CLOSED" and ear > EYE_OPEN_THRESHOLD:
                                blink_state = "OPEN"
                                blink_count += 1
                                current_status_text = f"Blink {blink_count}/2 - Eyes Open"
                                if blink_count >= 2:
                                    action_ok = True
                                    current_status_text = "GOOD! HOLD IT!"
                            
                            if not action_ok:
                                current_status_text = f"Blink {blink_count+1}/2 - Blink Now!" if blink_count < 2 else "GOOD! HOLD IT!"

                        elif prompt_tag == "smile":
                            if neutral_mouth_width == 0: 
                                neutral_mouth_width = 50  # Fallback
                            current_mouth_width = euclidian_distance(
                                landmark_to_xy(landmarks[MOUTH_LEFT_CORNER_IDX], img_w, img_h), 
                                landmark_to_xy(landmarks[MOUTH_RIGHT_CORNER_IDX], img_w, img_h)
                            )
                            width_increase_ratio = current_mouth_width / max(1e-6, neutral_mouth_width)
                            # More lenient smile thresholds
                            SMILE_M_RATIO_THRESHOLD = 0.025
                            SMILE_WIDTH_INCREASE_THRESHOLD = 1.03
                            if mratio > SMILE_M_RATIO_THRESHOLD and width_increase_ratio > SMILE_WIDTH_INCREASE_THRESHOLD:
                                action_ok = True
                                current_status_text = "GREAT SMILE! HOLD IT!"
                            else:
                                current_status_text = "SMILE WIDER!"

                        cv2.putText(display, current_status_text, (10, img_h - 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                        if action_ok and (time.time() - last_capture_time) > CAPTURE_COOLDOWN:
                            qc_pass, qc_reason = quality_checks(face_crop)
                            if qc_pass:
                                timestamp = datetime.utcnow().isoformat()
                                fname = os.path.join(DATA_DIR, f"{user_id}_{prompt_tag}_{collected_images_count}.jpg")
                                if SAVE_RAW:
                                    try:
                                        cv2.imwrite(fname, face_crop)
                                        print(f"CAPTURED_IMAGE:{os.path.basename(fname)}")
                                    except Exception as e:
                                        print(f"ERROR_SAVING_IMAGE:{str(e)}")
                                
                                emb = compute_embedding(face_crop)
                                if emb is not None:
                                    meta = {
                                        "user_id": user_id, 
                                        "prompt": prompt_tag, 
                                        "timestamp": timestamp, 
                                        "interocular_px": int(interocular), 
                                        "ear": float(ear), 
                                        "mouth_ratio": float(mratio)
                                    }
                                    save_embedding_record(user_id, emb, meta)
                                    collected_images_count += 1
                                    last_capture_time = time.time()
                                    action_captured = True
                                    time.sleep(0.5)
                                    break
                                else:
                                    print("Failed to compute embedding.")
                            else:
                                qc_status_text = f"REJECTED: {qc_reason.upper()}!"
                                cv2.putText(display, qc_status_text, (10, img_h - 50), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(display, "NO FACE DETECTED", (10, img_h - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                cv2.imshow("Enroll - press q to cancel", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("ENROLLMENT_CANCELLED_BY_USER")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            if not action_captured:
                print(f"Prompt '{prompt_text}' timed out.")

        cap.release()
        cv2.destroyAllWindows()
        print(f"Enrollment finished. Collected {collected_images_count} embeddings for {user_id}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Guided-prompt auto-capture for face enrollment.")
    parser.add_argument("user_id", help="The user ID for enrollment.")
    args = parser.parse_args()
    run_enrollment(args.user_id)

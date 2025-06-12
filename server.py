import os
import io
import base64
import time
import threading

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import mediapipe as mp

app = Flask(__name__)
CORS(app)

# ─── POSE GAME STATE ──────────────────────────────────────────
game_state = {
    "success_count": 0,
    "fail_count": 0,
    "current_pose_idx": 0,
    "evaluation_period": 3.0,  # seconds
    "pose_start_time": None,
    "is_evaluating": False,
    "current_score": 0.0,
    "result": None,
}

# ─── POSE CONFIG ─────────────────────────────────────────────
POSES = [
    {
        "name": "RAISE_HANDS",
        "angles": [
            (("LEFT_SHOULDER","LEFT_ELBOW","LEFT_WRIST"), 180, 0.5),
            (("RIGHT_SHOULDER","RIGHT_ELBOW","RIGHT_WRIST"),180, 0.5),
        ],
        "threshold": 60,
    },
    {
        "name": "SIU",
        "angles": [
            (("LEFT_ELBOW","LEFT_SHOULDER","LEFT_HIP"),45,0.5),
            (("RIGHT_ELBOW","RIGHT_SHOULDER","RIGHT_HIP"),45,0.5),
        ],
        "threshold": 70,
    },
    {
        "name": "JOG",
        "angles": [
            (("LEFT_HIP","LEFT_KNEE","LEFT_ANKLE"),60,0.5),
            (("LEFT_SHOULDER","LEFT_ELBOW","LEFT_WRIST"),45,0.5),
        ],
        "threshold": 70,
    },
]

# ─── MEDIA PIPE SETUP ────────────────────────────────────────
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose_detector = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    v1 = a - b
    v2 = c - b
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

def evaluate_pose(landmarks, cfg):
    total_score, total_weight = 0.0, 0.0
    for pts, target, weight in cfg["angles"]:
        idxA = getattr(mp_pose.PoseLandmark, pts[0]).value
        idxB = getattr(mp_pose.PoseLandmark, pts[1]).value
        idxC = getattr(mp_pose.PoseLandmark, pts[2]).value
        ptA = [landmarks[idxA].x, landmarks[idxA].y]
        ptB = [landmarks[idxB].x, landmarks[idxB].y]
        ptC = [landmarks[idxC].x, landmarks[idxC].y]
        angle = calculate_angle(ptA, ptB, ptC)
        score = max(0, 100 - abs(angle - target))
        total_score  += score * weight
        total_weight += 100   * weight
    return (total_score/total_weight*100) if total_weight else 0

# ─── GAME CONTROL ────────────────────────────────────────────
@app.route('/start_game', methods=['POST'])
def start_game():
    game_state.update({
        "success_count": 0,
        "fail_count": 0,
        "current_pose_idx": 0,
        "pose_start_time": time.time(),
        "is_evaluating": True,
        "current_score": 0.0,
        "result": None,
    })
    return jsonify(status="started")

@app.route('/evaluate_frame', methods=['POST'])
def evaluate_frame():
    """
    Accepts a JSON body: { "image": "<base64-JPEG string>" }
    Returns updated game_state.
    """
    data = request.get_json()
    b64 = data.get("image", "").split(",",1)[-1]
    frame = cv2.imdecode(
        np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR
    )
    # flip/resize if needed
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(image_rgb)
    
    now = time.time()
    cfg = POSES[game_state["current_pose_idx"]]
    # live scoring
    if results.pose_landmarks and game_state["is_evaluating"]:
        lm = results.pose_landmarks.landmark
        game_state["current_score"] = evaluate_pose(lm, cfg)
    else:
        game_state["current_score"] = 0.0

    # period end?
    if now - game_state["pose_start_time"] >= cfg["threshold"] and game_state["is_evaluating"]:
        score = game_state["current_score"]
        if score >= cfg["threshold"]:
            game_state["success_count"] += 1
            game_state["result"] = "SUCCESS"
        else:
            game_state["fail_count"]    += 1
            game_state["result"] = "FAIL"
        # next pose
        game_state["current_pose_idx"]  = (game_state["current_pose_idx"]+1)%len(POSES)
        game_state["pose_start_time"]   = now
        game_state["is_evaluating"]     = True
        game_state["current_score"]     = 0.0

    # return the full state for the front-end to render
    return jsonify({
        "current_pose": cfg["name"],
        "current_score": round(game_state["current_score"],2),
        "success_count": game_state["success_count"],
        "fail_count": game_state["fail_count"],
        "result": game_state["result"]
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

"""
AI Physiotherapist - Main Flask Server Application.
Full-stack application delivering real-time camera posture tracking,
YOLO-pose estimation, exercise repetition counting, posture accuracy scoring,
daily analytics dashboards, and physiotherapist clinical reporting.
"""

import os
import cv2
import time
import base64
import json
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from config import Config, EXERCISES
from posture_engine import PostureEngine
import database as db

app = Flask(__name__)
app.config.from_object(Config)

# Initialize Posture Tracker Engine
engine = PostureEngine()

# Global Camera Object & Threading Lock
import threading
camera = None
camera_lock = threading.Lock()

def get_camera():
    """Safely retrieves or initializes the OpenCV webcam VideoCapture instance."""
    global camera
    if camera is None or not camera.isOpened():
        try:
            if camera is not None:
                camera.release()
            camera = cv2.VideoCapture(0)  # Laptop default webcam
            if not camera.isOpened():
                print("[WARNING] Primary camera (0) not opened. Retrying fallback camera index (1)...")
                camera = cv2.VideoCapture(1)
        except Exception as e:
            print(f"[ERROR] Failed to open camera device: {e}")
            camera = None
    return camera

def generate_video_stream():
    """Generator function for MJPEG live camera stream with YOLO pose overlay."""
    global camera
    consecutive_errors = 0

    while True:
        success = False
        frame = None

        with camera_lock:
            cam = get_camera()
            if cam and cam.isOpened():
                try:
                    success, frame = cam.read()
                except Exception as e:
                    print(f"[WARNING] Camera read error: {e}")
                    success = False

            if not success or frame is None:
                consecutive_errors += 1
                if consecutive_errors > 10 and camera is not None:
                    # Force release and re-attempt connection
                    try:
                        camera.release()
                    except Exception:
                        pass
                    camera = None
                    consecutive_errors = 0

        if not success or frame is None:
            # Generate blank canvas with friendly camera standby notice
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Webcam Standby / Connecting...", (100, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            time.sleep(0.05)
        else:
            consecutive_errors = 0

        try:
            # Process posture frame with YOLO engine
            telemetry, annotated_frame = engine.process_frame(frame)

            # Encode frame as JPEG
            ret, jpeg = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                continue

            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"[ERROR] Error in stream loop: {e}")
            time.sleep(0.05)

# -------------------------------------------------------------------
# VIEW ROUTES
# -------------------------------------------------------------------

@app.route('/')
def index():
    """Main Live Posture Tracking Dashboard."""
    profile = db.get_patient_profile()
    return render_template('index.html', exercises=EXERCISES, profile=profile, active_exercise=engine.active_exercise_id)

@app.route('/analytics')
def analytics():
    """Daily Analytics & Physiotherapist Tracking View."""
    profile = db.get_patient_profile()
    today_data = db.get_today_summary()
    history_data = db.get_recent_history(days=7)
    return render_template('analytics.html', profile=profile, today=today_data, history=history_data)

@app.route('/report_preview')
def report_preview():
    """Printable / Exportable Physiotherapist Daily Clinical Report."""
    profile = db.get_patient_profile()
    today_data = db.get_today_summary()
    return render_template('report_template.html', profile=profile, today=today_data)

# -------------------------------------------------------------------
# API ENDPOINTS
# -------------------------------------------------------------------

@app.route('/video_feed')
def video_feed():
    """MJPEG Live Camera Stream Endpoint."""
    return Response(generate_video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/exercises', methods=['GET'])
def api_get_exercises():
    """Get list of all 10 Exercise Modes."""
    return jsonify({
        "status": "success",
        "exercises": EXERCISES,
        "active_exercise_id": engine.active_exercise_id
    })

@app.route('/api/exercise/select', methods=['POST'])
def api_select_exercise():
    """Switch active exercise mode."""
    data = request.get_json() or {}
    exercise_id = data.get('exercise_id')
    if exercise_id and engine.set_exercise(exercise_id):
        return jsonify({
            "status": "success",
            "message": f"Switched to exercise {exercise_id}",
            "exercise": EXERCISES[exercise_id]
        })
    return jsonify({"status": "error", "message": "Invalid exercise ID"}), 400

@app.route('/api/session/start', methods=['POST'])
def api_start_session():
    """Start or reset active exercise session."""
    engine.reset_session()
    return jsonify({
        "status": "success",
        "message": f"Started session for {engine.exercise_config['name']}",
        "telemetry": {
            "exercise_id": engine.active_exercise_id,
            "target_reps": engine.target_reps,
            "rep_count": 0
        }
    })

@app.route('/api/session/stop', methods=['POST'])
def api_stop_session():
    """Stop session and persist session metrics into database."""
    duration_sec = int(time.time() - engine.session_start_time)
    reps = int(engine.rep_count)
    accuracy = float(round(engine.form_accuracy_score, 1))
    min_angle = float(round(engine.min_angle_achieved, 1)) if engine.min_angle_achieved < 900 else 0.0
    max_angle = float(round(engine.max_angle_achieved, 1))

    db.save_exercise_session(
        exercise_id=engine.active_exercise_id,
        exercise_name=engine.exercise_config["name"],
        reps=reps,
        target_reps=engine.target_reps,
        accuracy=accuracy,
        duration_sec=duration_sec,
        min_angle=min_angle,
        max_angle=max_angle,
        warnings=engine.warning_flags
    )

    return jsonify({
        "status": "success",
        "message": "Session recorded successfully!",
        "summary": {
            "exercise_name": engine.exercise_config["name"],
            "reps_completed": reps,
            "target_reps": engine.target_reps,
            "accuracy_score": accuracy,
            "duration_seconds": duration_sec
        }
    })

@app.route('/api/session/current', methods=['GET'])
def api_get_current_telemetry():
    """Get real-time posture telemetry for frontend client rendering."""
    active_duration = int(time.time() - engine.session_start_time)
    return jsonify({
        "status": "success",
        "exercise_id": str(engine.active_exercise_id),
        "exercise_name": str(engine.exercise_config["name"]),
        "rep_count": int(engine.rep_count),
        "target_reps": int(engine.target_reps),
        "stage": str(engine.stage),
        "current_angle": float(engine.current_angle),
        "angle_rest": float(engine.exercise_config["angle_rest"]),
        "angle_target": float(engine.exercise_config["angle_target"]),
        "form_score": float(round(engine.form_accuracy_score, 1)),
        "feedback": str(engine.current_feedback),
        "warnings": [str(w) for w in engine.warning_flags],
        "active_duration_seconds": int(active_duration)
    })

@app.route('/api/process_frame', methods=['POST'])
def api_process_frame():
    """Process base64 webcam frame sent from browser JavaScript."""
    data = request.get_json() or {}
    image_b64 = data.get('image')

    if not image_b64:
        return jsonify({"status": "error", "message": "No image data provided"}), 400

    try:
        # Decode base64 to OpenCV image
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]

        image_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"status": "error", "message": "Failed to decode frame"}), 400

        telemetry, annotated_frame = engine.process_frame(frame)

        # Encode annotated output frame back to base64
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        annotated_b64 = base64.b64encode(buffer).decode('utf-8')

        telemetry["annotated_image"] = f"data:image/jpeg;base64,{annotated_b64}"
        return jsonify({"status": "success", "data": telemetry})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/daily_analytics', methods=['GET'])
def api_get_daily_analytics():
    """Get daily summary and historical analytics data for charts."""
    days = request.args.get('days', 7, type=int)
    today = db.get_today_summary()
    history = db.get_recent_history(days=days)
    return jsonify({
        "status": "success",
        "today": today,
        "history": history
    })

@app.route('/api/send_report', methods=['POST'])
def api_send_report():
    """Generates and records daily report sent to physiotherapist."""
    data = request.get_json() or {}
    physio_email = data.get('physio_email') or 'dr.jenkins@physiorehab.com'
    patient_name = data.get('patient_name') or 'Alex Mercer'
    therapist_notes = data.get('therapist_notes') or ''

    today_summary = db.get_today_summary()
    report_id = db.save_physio_report(physio_email, patient_name, therapist_notes, today_summary)

    return jsonify({
        "status": "success",
        "message": f"Daily Rehab Report successfully transmitted to {physio_email}!",
        "report_id": report_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route('/api/patient_profile', methods=['GET', 'POST'])
def api_patient_profile():
    if request.method == 'POST':
        data = request.get_json() or {}
        db.update_patient_profile(
            patient_name=data.get('patient_name', 'Alex Mercer'),
            patient_id=data.get('patient_id', 'PAT-2026-8841'),
            physio_name=data.get('physio_name', 'Dr. Sarah Jenkins, PT'),
            physio_email=data.get('physio_email', 'dr.jenkins@physiorehab.com'),
            recovery_plan=data.get('recovery_plan', 'Post-Op Rehabilitation Protocol')
        )
        return jsonify({"status": "success", "message": "Profile updated successfully"})
    else:
        profile = db.get_patient_profile()
        return jsonify({"status": "success", "profile": profile})

if __name__ == '__main__':
    print("\n========================================================")
    print("  AI PHYSIOTHERAPIST - POST-OP REHAB POSTURE ENGINE    ")
    print("  Server running on http://127.0.0.1:5050               ")
    print("========================================================\n")
    app.run(host='0.0.0.0', port=5050, debug=True)

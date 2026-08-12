"""
AI Physiotherapist - Main Flask Server Application.

Full-stack application delivering:
- Real-time camera posture tracking
- YOLO pose estimation
- Exercise repetition counting
- Posture accuracy scoring
- Daily analytics
- Physiotherapist clinical reporting
"""

import base64
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from config import Config, EXERCISES
from posture_engine import PostureEngine
import database as db


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)
app.config.from_object(Config)


# ============================================================
# POSTURE ENGINE
# ============================================================

try:
    engine = PostureEngine()
    print("[SUCCESS] PostureEngine initialized.")
except Exception as exc:
    print(f"[ERROR] Failed to initialize PostureEngine: {exc}")
    raise


# ============================================================
# CAMERA
# ============================================================

camera = None


def get_camera():
    """
    Safely initialize and return the OpenCV webcam.
    """

    global camera

    if camera is None or not camera.isOpened():

        try:
            # Release previous camera if necessary
            if camera is not None:
                camera.release()

            # Try default camera
            camera = cv2.VideoCapture(0)

            if not camera.isOpened():

                print(
                    "[WARNING] Camera 0 could not be opened. "
                    "Trying camera 1..."
                )

                camera.release()
                camera = cv2.VideoCapture(1)

            if camera.isOpened():
                print("[SUCCESS] Camera initialized.")

            else:
                print("[WARNING] No webcam could be opened.")
                camera = None

        except Exception as exc:

            print(f"[ERROR] Failed to initialize camera: {exc}")
            camera = None

    return camera


# ============================================================
# VIDEO STREAM
# ============================================================

def generate_video_stream():
    """
    Generate an MJPEG camera stream with posture overlays.
    """

    global camera

    consecutive_errors = 0

    while True:

        cam = get_camera()

        success = False
        frame = None

        # --------------------------------------------------------
        # Read camera frame
        # --------------------------------------------------------

        if cam is not None and cam.isOpened():

            try:
                success, frame = cam.read()

            except Exception as exc:

                print(f"[WARNING] Camera read error: {exc}")
                success = False

        # --------------------------------------------------------
        # Camera failure
        # --------------------------------------------------------

        if not success or frame is None:

            consecutive_errors += 1

            if consecutive_errors > 10:

                if camera is not None:

                    try:
                        camera.release()

                    except Exception:
                        pass

                camera = None
                consecutive_errors = 0

            # Create standby frame
            frame = np.zeros(
                (480, 640, 3),
                dtype=np.uint8
            )

            cv2.putText(
                frame,
                "Webcam Standby / Connecting...",
                (100, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                2
            )

            time.sleep(0.05)

        else:

            consecutive_errors = 0

        # --------------------------------------------------------
        # Process posture
        # --------------------------------------------------------

        try:

            telemetry, annotated_frame = engine.process_frame(frame)

            # Encode frame as JPEG
            success_encode, jpeg = cv2.imencode(
                ".jpg",
                annotated_frame
            )

            if not success_encode:
                continue

            frame_bytes = jpeg.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )

        except Exception as exc:

            print(
                f"[ERROR] Error processing video frame: {exc}"
            )

            time.sleep(0.05)


# ============================================================
# WEB ROUTES
# ============================================================

@app.route("/")
def index():
    """
    Main live posture tracking dashboard.
    """

    profile = db.get_patient_profile()

    return render_template(
        "index.html",
        exercises=EXERCISES,
        profile=profile,
        active_exercise=engine.active_exercise_id
    )


@app.route("/analytics")
def analytics():
    """
    Daily analytics dashboard.
    """

    profile = db.get_patient_profile()

    today_data = db.get_today_summary()

    history_data = db.get_recent_history(
        days=7
    )

    return render_template(
        "analytics.html",
        profile=profile,
        today=today_data,
        history=history_data
    )


@app.route("/report_preview")
def report_preview():
    """
    Physiotherapist daily clinical report preview.
    """

    profile = db.get_patient_profile()

    today_data = db.get_today_summary()

    return render_template(
        "report_template.html",
        profile=profile,
        today=today_data
    )


# ============================================================
# CAMERA API
# ============================================================

@app.route("/video_feed")
def video_feed():
    """
    MJPEG live camera stream.
    """

    return Response(
        generate_video_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================
# EXERCISE API
# ============================================================

@app.route("/api/exercises", methods=["GET"])
def api_get_exercises():
    """
    Return all available exercises.
    """

    return jsonify(
        {
            "status": "success",
            "exercises": EXERCISES,
            "active_exercise_id": engine.active_exercise_id
        }
    )


@app.route("/api/exercise/select", methods=["POST"])
def api_select_exercise():
    """
    Change the active exercise.
    """

    data = request.get_json(silent=True) or {}

    exercise_id = data.get("exercise_id")

    if not exercise_id:

        return jsonify(
            {
                "status": "error",
                "message": "Exercise ID is required."
            }
        ), 400

    if engine.set_exercise(exercise_id):

        exercise = EXERCISES.get(
            exercise_id,
            engine.exercise_config
        )

        return jsonify(
            {
                "status": "success",
                "message": (
                    f"Switched to exercise {exercise_id}"
                ),
                "exercise": exercise
            }
        )

    return jsonify(
        {
            "status": "error",
            "message": "Invalid exercise ID."
        }
    ), 400


# ============================================================
# SESSION API
# ============================================================

@app.route("/api/session/start", methods=["POST"])
def api_start_session():
    """
    Start or reset the current exercise session.
    """

    engine.reset_session()

    return jsonify(
        {
            "status": "success",
            "message": (
                f"Started session for "
                f"{engine.exercise_config.get('name', 'Exercise')}"
            ),
            "telemetry": {
                "exercise_id": engine.active_exercise_id,
                "target_reps": engine.target_reps,
                "rep_count": 0
            }
        }
    )


@app.route("/api/session/stop", methods=["POST"])
def api_stop_session():
    """
    Stop the current session and save its metrics.
    """

    duration_sec = int(
        max(
            0,
            time.time() - engine.session_start_time
        )
    )

    reps = int(engine.rep_count)

    accuracy = float(
        round(
            engine.form_accuracy_score,
            1
        )
    )

    min_angle = (
        float(round(engine.min_angle_achieved, 1))
        if engine.min_angle_achieved < 900
        else 0.0
    )

    max_angle = float(
        round(
            engine.max_angle_achieved,
            1
        )
    )

    try:

        db.save_exercise_session(
            exercise_id=engine.active_exercise_id,
            exercise_name=engine.exercise_config.get(
                "name",
                engine.active_exercise_id
            ),
            reps=reps,
            target_reps=engine.target_reps,
            accuracy=accuracy,
            duration_sec=duration_sec,
            min_angle=min_angle,
            max_angle=max_angle,
            warnings=engine.warning_flags
        )

    except Exception as exc:

        print(
            f"[ERROR] Failed to save exercise session: {exc}"
        )

        return jsonify(
            {
                "status": "error",
                "message": (
                    f"Could not save session: {exc}"
                )
            }
        ), 500

    return jsonify(
        {
            "status": "success",
            "message": "Session recorded successfully!",
            "summary": {
                "exercise_name": engine.exercise_config.get(
                    "name",
                    engine.active_exercise_id
                ),
                "reps_completed": reps,
                "target_reps": engine.target_reps,
                "accuracy_score": accuracy,
                "duration_seconds": duration_sec
            }
        }
    )


@app.route("/api/session/current", methods=["GET"])
def api_get_current_telemetry():
    """
    Return current posture telemetry.
    """

    active_duration = int(
        max(
            0,
            time.time() - engine.session_start_time
        )
    )

    exercise_config = engine.exercise_config

    return jsonify(
        {
            "status": "success",
            "exercise_id": str(
                engine.active_exercise_id
            ),
            "exercise_name": str(
                exercise_config.get(
                    "name",
                    engine.active_exercise_id
                )
            ),
            "rep_count": int(
                engine.rep_count
            ),
            "target_reps": int(
                engine.target_reps
            ),
            "stage": str(
                engine.stage
            ),
            "current_angle": float(
                engine.current_angle
            ),
            "angle_rest": float(
                exercise_config.get(
                    "angle_rest",
                    180
                )
            ),
            "angle_target": float(
                exercise_config.get(
                    "angle_target",
                    90
                )
            ),
            "form_score": float(
                round(
                    engine.form_accuracy_score,
                    1
                )
            ),
            "feedback": str(
                engine.current_feedback
            ),
            "warnings": [
                str(warning)
                for warning in engine.warning_flags
            ],
            "active_duration_seconds": int(
                active_duration
            )
        }
    )


# ============================================================
# FRAME PROCESSING API
# ============================================================

@app.route("/api/process_frame", methods=["POST"])
def api_process_frame():
    """
    Process a base64 image sent from browser JavaScript.
    """

    data = request.get_json(silent=True) or {}

    image_b64 = data.get("image")

    if not image_b64:

        return jsonify(
            {
                "status": "error",
                "message": "No image data provided."
            }
        ), 400

    try:

        # --------------------------------------------------------
        # Remove data URL prefix if present
        # --------------------------------------------------------

        if "," in image_b64:

            image_b64 = image_b64.split(
                ",",
                1
            )[1]

        # --------------------------------------------------------
        # Decode base64
        # --------------------------------------------------------

        image_bytes = base64.b64decode(
            image_b64
        )

        nparr = np.frombuffer(
            image_bytes,
            np.uint8
        )

        frame = cv2.imdecode(
            nparr,
            cv2.IMREAD_COLOR
        )

        if frame is None:

            return jsonify(
                {
                    "status": "error",
                    "message": "Failed to decode frame."
                }
            ), 400

        # --------------------------------------------------------
        # Process through posture engine
        # --------------------------------------------------------

        telemetry, annotated_frame = (
            engine.process_frame(frame)
        )

        # --------------------------------------------------------
        # Encode processed frame
        # --------------------------------------------------------

        success, buffer = cv2.imencode(
            ".jpg",
            annotated_frame
        )

        if not success:

            return jsonify(
                {
                    "status": "error",
                    "message": "Failed to encode processed frame."
                }
            ), 500

        annotated_b64 = base64.b64encode(
            buffer.tobytes()
        ).decode("utf-8")

        telemetry["annotated_image"] = (
            "data:image/jpeg;base64,"
            + annotated_b64
        )

        return jsonify(
            {
                "status": "success",
                "data": telemetry
            }
        )

    except Exception as exc:

        print(
            f"[ERROR] Frame processing error: {exc}"
        )

        return jsonify(
            {
                "status": "error",
                "message": str(exc)
            }
        ), 500


# ============================================================
# ANALYTICS API
# ============================================================

@app.route("/api/daily_analytics", methods=["GET"])
def api_get_daily_analytics():
    """
    Return today's summary and historical analytics.
    """

    days = request.args.get(
        "days",
        default=7,
        type=int
    )

    # Prevent unreasonable requests
    days = max(
        1,
        min(days, 365)
    )

    today = db.get_today_summary()

    history = db.get_recent_history(
        days=days
    )

    return jsonify(
        {
            "status": "success",
            "today": today,
            "history": history
        }
    )


# ============================================================
# PHYSIOTHERAPIST REPORT API
# ============================================================

@app.route("/api/send_report", methods=["POST"])
def api_send_report():
    """
    Generate and save a physiotherapist report.
    """

    data = request.get_json(
        silent=True
    ) or {}

    physio_email = (
        data.get("physio_email")
        or "dr.jenkins@physiorehab.com"
    )

    patient_name = (
        data.get("patient_name")
        or "Alex Mercer"
    )

    therapist_notes = (
        data.get("therapist_notes")
        or ""
    )

    try:

        today_summary = (
            db.get_today_summary()
        )

        report_id = db.save_physio_report(
            physio_email,
            patient_name,
            therapist_notes,
            today_summary
        )

        return jsonify(
            {
                "status": "success",
                "message": (
                    "Daily Rehab Report successfully "
                    f"transmitted to {physio_email}!"
                ),
                "report_id": report_id,
                "timestamp": time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            }
        )

    except Exception as exc:

        print(
            f"[ERROR] Failed to generate report: {exc}"
        )

        return jsonify(
            {
                "status": "error",
                "message": str(exc)
            }
        ), 500


# ============================================================
# PATIENT PROFILE API
# ============================================================

@app.route(
    "/api/patient_profile",
    methods=["GET", "POST"]
)
def api_patient_profile():
    """
    Get or update patient profile.
    """

    if request.method == "POST":

        data = request.get_json(
            silent=True
        ) or {}

        try:

            db.update_patient_profile(
                patient_name=data.get(
                    "patient_name",
                    "Alex Mercer"
                ),
                patient_id=data.get(
                    "patient_id",
                    "PAT-2026-8841"
                ),
                physio_name=data.get(
                    "physio_name",
                    "Dr. Sarah Jenkins, PT"
                ),
                physio_email=data.get(
                    "physio_email",
                    "dr.jenkins@physiorehab.com"
                ),
                recovery_plan=data.get(
                    "recovery_plan",
                    "Post-Op Rehabilitation Protocol"
                )
            )

            return jsonify(
                {
                    "status": "success",
                    "message": "Profile updated successfully."
                }
            )

        except Exception as exc:

            print(
                f"[ERROR] Failed to update profile: {exc}"
            )

            return jsonify(
                {
                    "status": "error",
                    "message": str(exc)
                }
            ), 500

    # --------------------------------------------------------
    # GET profile
    # --------------------------------------------------------

    try:

        profile = db.get_patient_profile()

        return jsonify(
            {
                "status": "success",
                "profile": profile
            }
        )

    except Exception as exc:

        print(
            f"[ERROR] Failed to load profile: {exc}"
        )

        return jsonify(
            {
                "status": "error",
                "message": str(exc)
            }
        ), 500


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def page_not_found(error):
    """
    Handle missing routes.
    """

    return jsonify(
        {
            "status": "error",
            "message": "Route not found."
        }
    ), 404


@app.errorhandler(500)
def internal_server_error(error):
    """
    Handle unexpected server errors.
    """

    return jsonify(
        {
            "status": "error",
            "message": "Internal server error."
        }
    ), 500


# ============================================================
# APPLICATION START
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("       AI PHYSIOTHERAPIST")
    print("       POSTURE & EXERCISE TRACKER")
    print("=" * 60)
    print()
    print("Server running at:")
    print("http://127.0.0.1:5050")
    print()
    print("Press CTRL+C to stop the server.")
    print("=" * 60)
    print()

    app.run(
        host="0.0.0.0",
        port=5050,
        debug=True,
        use_reloader=False
    )
"""
YOLO-Pose Posture & Exercise Tracker Engine
============================================

Real-time posture/exercise tracking engine using Ultralytics YOLO Pose.

Features:
- YOLOv8 pose estimation
- Joint-angle calculation
- Exercise repetition counting
- Form/posture scoring
- Coaching feedback
- Skeleton and telemetry overlay
- Simulated pose fallback when YOLO is unavailable
"""

import math
import time
from pathlib import Path

# These packages are loaded dynamically so VS Code/Pyrefly does not show
# false "missing import" diagnostics when the virtual environment is not
# being picked up by the editor. They still must be installed in your venv.
import importlib


def _load_package(name):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeError(
            f"Missing Python package '{name}'. Activate your venv and run "
            f"'python -m pip install {name}'."
        ) from exc


cv2 = _load_package("cv2")
np = _load_package("numpy")
YOLO = getattr(_load_package("ultralytics"), "YOLO")

from config import EXERCISES


# Always locate the model relative to this file.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "yolov8n-pose.pt"


class PostureEngine:
    """Main posture and exercise tracking engine."""

    def __init__(self):
        self.model = None
        self.model_loaded = False

        # ---------------------------------------------------------
        # Active exercise
        # ---------------------------------------------------------
        self.active_exercise_id = "left_biceps_curl"

        if self.active_exercise_id not in EXERCISES:
            if not EXERCISES:
                raise ValueError("EXERCISES is empty in config.py")

            self.active_exercise_id = next(iter(EXERCISES))

        self.exercise_config = EXERCISES[self.active_exercise_id]

        # ---------------------------------------------------------
        # Repetition state
        # ---------------------------------------------------------
        self.rep_count = 0

        self.target_reps = int(
            self.exercise_config.get("recommended_reps", 10)
        )

        self.stage = "IDLE"

        self.last_rep_time = time.time()
        self.session_start_time = time.time()

        # ---------------------------------------------------------
        # Form / telemetry
        # ---------------------------------------------------------
        self.current_angle = 0.0
        self.left_angle = 0.0
        self.right_angle = 0.0

        self.form_accuracy_score = 100.0
        self.accuracy_history = []

        self.min_angle_achieved = 999.0
        self.max_angle_achieved = 0.0

        self.current_feedback = (
            "Position yourself in front of the camera to begin."
        )

        self.warning_flags = []

        # ---------------------------------------------------------
        # Simulation fallback
        # ---------------------------------------------------------
        self.sim_phase = 0.0

        # Load YOLO model.
        self._initialize_yolo()

    # =============================================================
    # YOLO INITIALIZATION
    # =============================================================

    def _initialize_yolo(self):
        """Load the local YOLO pose model."""

        try:
            print(f"[INFO] Looking for YOLO model: {MODEL_PATH}")

            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"YOLO model not found at: {MODEL_PATH}"
                )

            print("[INFO] Loading YOLOv8-pose model...")

            self.model = YOLO(str(MODEL_PATH))

            self.model_loaded = True

            print(
                "[SUCCESS] YOLOv8-pose model loaded successfully!"
            )

        except Exception as exc:
            self.model = None
            self.model_loaded = False

            print(
                f"[WARNING] Could not load YOLO model ({exc}). "
                "Using simulated pose fallback."
            )

    # =============================================================
    # EXERCISE MANAGEMENT
    # =============================================================

    def set_exercise(self, exercise_id):
        """Switch the active exercise."""

        if exercise_id not in EXERCISES:
            print(f"[WARNING] Unknown exercise: {exercise_id}")
            return False

        self.active_exercise_id = exercise_id
        self.exercise_config = EXERCISES[exercise_id]

        self.reset_session()

        exercise_name = self.exercise_config.get(
            "name",
            exercise_id
        )

        print(
            f"[INFO] Switched exercise to: {exercise_name}"
        )

        return True

    def reset_session(self):
        """Reset repetition counter and telemetry."""

        self.rep_count = 0

        self.target_reps = int(
            self.exercise_config.get(
                "recommended_reps",
                10
            )
        )

        self.stage = "IDLE"

        self.last_rep_time = time.time()
        self.session_start_time = time.time()

        self.current_angle = 0.0
        self.left_angle = 0.0
        self.right_angle = 0.0

        self.form_accuracy_score = 100.0
        self.accuracy_history = []

        self.min_angle_achieved = 999.0
        self.max_angle_achieved = 0.0

        self.warning_flags = []

        exercise_name = self.exercise_config.get(
            "name",
            self.active_exercise_id
        )

        self.current_feedback = (
            f"Ready for {exercise_name}. Start motion."
        )

    # =============================================================
    # ANGLE CALCULATION
    # =============================================================

    @staticmethod
    def calculate_angle(p1, p2, p3):
        """
        Calculate the angle at p2.

        p1, p2, p3:
            [x, y] coordinates

        Returns:
            Angle between 0 and 180 degrees.
        """

        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        x3, y3 = float(p3[0]), float(p3[1])

        radians = (
            math.atan2(
                y3 - y2,
                x3 - x2
            )
            -
            math.atan2(
                y1 - y2,
                x1 - x2
            )
        )

        angle = abs(
            radians * 180.0 / math.pi
        )

        if angle > 180.0:
            angle = 360.0 - angle

        return round(angle, 1)

    # =============================================================
    # FRAME PROCESSING
    # =============================================================

    def process_frame(self, frame):
        """
        Process one OpenCV BGR frame.

        Returns:
            telemetry, annotated_frame
        """

        if frame is None:
            raise ValueError(
                "process_frame received an empty frame"
            )

        if not isinstance(frame, np.ndarray):
            raise TypeError(
                "frame must be a numpy.ndarray"
            )

        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError(
                "frame must be a color image"
            )

        height, width = frame.shape[:2]

        keypoints = None
        keypoints_conf = None
        person_detected = False

        # ---------------------------------------------------------
        # YOLO inference
        # ---------------------------------------------------------

        if self.model_loaded and self.model is not None:

            try:

                results = self.model(
                    frame,
                    verbose=False
                )

                if len(results) > 0:

                    result = results[0]

                    kpts_data = result.keypoints

                    if (
                        kpts_data is not None
                        and kpts_data.xy is not None
                    ):

                        if len(kpts_data.xy) > 0:

                            keypoints = (
                                kpts_data.xy[0]
                                .cpu()
                                .numpy()
                                .astype(np.float32)
                            )

                            if kpts_data.conf is not None:

                                keypoints_conf = (
                                    kpts_data.conf[0]
                                    .cpu()
                                    .numpy()
                                    .astype(np.float32)
                                )

                            else:

                                keypoints_conf = np.ones(
                                    len(keypoints),
                                    dtype=np.float32
                                )

                            if len(keypoints) >= 17:

                                person_detected = True

            except Exception as exc:

                print(
                    f"[ERROR] YOLO inference error: {exc}"
                )

                person_detected = False

        # ---------------------------------------------------------
        # Fallback simulation
        # ---------------------------------------------------------

        if (
            not person_detected
            or keypoints is None
            or len(keypoints) < 17
        ):

            keypoints, keypoints_conf = (
                self._generate_simulated_keypoints(
                    width,
                    height
                )
            )

        keypoints = np.asarray(
            keypoints,
            dtype=np.float32
        )

        keypoints_conf = np.asarray(
            keypoints_conf,
            dtype=np.float32
        )

        if (
            len(keypoints) < 17
            or len(keypoints_conf) < 17
        ):
            raise RuntimeError(
                "Pose data must contain at least "
                "17 COCO keypoints."
            )

        # ---------------------------------------------------------
        # Determine exercise configuration
        # ---------------------------------------------------------

        is_bilateral = (
            "joints_left" in self.exercise_config
            and
            "joints_right" in self.exercise_config
        )

        if is_bilateral:

            j_l = self.exercise_config["joints_left"]
            j_r = self.exercise_config["joints_right"]

            p1_l = keypoints[int(j_l["p1"])]
            p2_l = keypoints[int(j_l["p2"])]
            p3_l = keypoints[int(j_l["p3"])]

            p1_r = keypoints[int(j_r["p1"])]
            p2_r = keypoints[int(j_r["p2"])]
            p3_r = keypoints[int(j_r["p3"])]

            conf_l = min(
                float(
                    keypoints_conf[
                        int(j_l["p1"])
                    ]
                ),
                float(
                    keypoints_conf[
                        int(j_l["p2"])
                    ]
                ),
                float(
                    keypoints_conf[
                        int(j_l["p3"])
                    ]
                )
            )

            conf_r = min(
                float(
                    keypoints_conf[
                        int(j_r["p1"])
                    ]
                ),
                float(
                    keypoints_conf[
                        int(j_r["p2"])
                    ]
                ),
                float(
                    keypoints_conf[
                        int(j_r["p3"])
                    ]
                )
            )

            self.left_angle = self.calculate_angle(
                p1_l,
                p2_l,
                p3_l
            )

            self.right_angle = self.calculate_angle(
                p1_r,
                p2_r,
                p3_r
            )

            angle = round(
                (
                    self.left_angle
                    +
                    self.right_angle
                ) / 2.0,
                1
            )

            p1 = p1_l
            p2 = p2_l
            p3 = p3_l

            joint_visibility = max(
                conf_l,
                conf_r
            )

        else:

            joints_cfg = (
                self.exercise_config.get("joints")
            )

            if (
                not joints_cfg
                and
                "joints_left" in self.exercise_config
            ):

                joints_cfg = (
                    self.exercise_config[
                        "joints_left"
                    ]
                )

            if not joints_cfg:

                raise ValueError(
                    "No joint configuration found "
                    f"for exercise: "
                    f"{self.active_exercise_id}"
                )

            idx1 = int(joints_cfg["p1"])
            idx2 = int(joints_cfg["p2"])
            idx3 = int(joints_cfg["p3"])

            p1 = keypoints[idx1]
            p2 = keypoints[idx2]
            p3 = keypoints[idx3]

            joint_visibility = min(
                float(keypoints_conf[idx1]),
                float(keypoints_conf[idx2]),
                float(keypoints_conf[idx3])
            )

            angle = self.calculate_angle(
                p1,
                p2,
                p3
            )

        self.current_angle = float(angle)

        # ---------------------------------------------------------
        # Visibility / rep logic
        # ---------------------------------------------------------

        if joint_visibility < 0.25:

            self.current_feedback = (
                "Target joints obscured. "
                "Position body in full camera view."
            )

            self.warning_flags = [
                "Low joint visibility"
            ]

        else:

            if angle < self.min_angle_achieved:
                self.min_angle_achieved = float(angle)

            if angle > self.max_angle_achieved:
                self.max_angle_achieved = float(angle)

            self._update_rep_state(
                angle,
                p1,
                p2,
                p3,
                keypoints,
                is_bilateral
            )

        # ---------------------------------------------------------
        # Draw output
        # ---------------------------------------------------------

        annotated_frame = self._draw_overlay(
            frame,
            keypoints,
            keypoints_conf,
            p1,
            p2,
            p3,
            angle
        )

        active_duration = int(
            time.time()
            -
            self.session_start_time
        )

        telemetry = {
            "exercise_id":
                str(self.active_exercise_id),

            "exercise_name":
                str(
                    self.exercise_config.get(
                        "name",
                        self.active_exercise_id
                    )
                ),

            "rep_count":
                int(self.rep_count),

            "target_reps":
                int(self.target_reps),

            "stage":
                str(self.stage),

            "current_angle":
                float(self.current_angle),

            "angle_rest":
                float(
                    self.exercise_config.get(
                        "angle_rest",
                        180
                    )
                ),

            "angle_target":
                float(
                    self.exercise_config.get(
                        "angle_target",
                        90
                    )
                ),

            "form_score":
                float(
                    round(
                        self.form_accuracy_score,
                        1
                    )
                ),

            "feedback":
                str(self.current_feedback),

            "warnings":
                [
                    str(w)
                    for w in self.warning_flags
                ],

            "active_duration_seconds":
                int(active_duration),

            "person_detected":
                bool(person_detected),

            "model_loaded":
                bool(self.model_loaded),

            "keypoints":
                keypoints.tolist()
        }

        return telemetry, annotated_frame

    # =============================================================
    # REPETITION STATE MACHINE
    # =============================================================

    def _update_rep_state(
        self,
        angle,
        p1,
        p2,
        p3,
        all_kpts,
        is_bilateral=False
    ):
        """Update repetition state and evaluate form."""

        rest_angle = float(
            self.exercise_config.get(
                "angle_rest",
                180
            )
        )

        target_angle = float(
            self.exercise_config.get(
                "angle_target",
                90
            )
        )

        start_thresh = float(
            self.exercise_config.get(
                "angle_threshold_start",
                rest_angle - 20
            )
        )

        finish_thresh = float(
            self.exercise_config.get(
                "angle_threshold_finish",
                target_angle + 10
            )
        )

        is_flexion_type = (
            target_angle < rest_angle
        )

        # ---------------------------------------------------------
        # State machine
        # ---------------------------------------------------------

        if self.stage == "IDLE":

            if (
                is_flexion_type
                and
                angle >= start_thresh
            ):

                self.stage = "EXTENDED"

                self.current_feedback = (
                    "Good starting posture. "
                    "Begin movement smoothly."
                )

            elif (
                not is_flexion_type
                and
                angle <= start_thresh
            ):

                self.stage = "EXTENDED"

                self.current_feedback = (
                    "Good starting posture. "
                    "Raise smoothly."
                )

            else:

                self.stage = "EXTENDED"

        elif self.stage == "EXTENDED":

            if (
                (
                    is_flexion_type
                    and
                    angle <= finish_thresh
                )
                or
                (
                    not is_flexion_type
                    and
                    angle >= finish_thresh
                )
            ):

                self.stage = "FLEXED"

                self.current_feedback = (
                    "Peak contraction reached! "
                    "Hold briefly, then lower controlled."
                )

        elif self.stage == "FLEXED":

            if (
                (
                    is_flexion_type
                    and
                    angle >= start_thresh - 10
                )
                or
                (
                    not is_flexion_type
                    and
                    angle <= start_thresh + 10
                )
            ):

                # Rep completed.
                self.rep_count += 1

                self.stage = "EXTENDED"

                self.last_rep_time = time.time()

                self.current_feedback = (
                    f"Great Rep! "
                    f"({self.rep_count}/{self.target_reps})"
                )

                rep_quality = (
                    self._evaluate_rep_quality(
                        angle,
                        p1,
                        p2,
                        p3,
                        all_kpts
                    )
                )

                self.accuracy_history.append(
                    rep_quality
                )

                self.form_accuracy_score = float(
                    sum(self.accuracy_history)
                    /
                    len(self.accuracy_history)
                )

        # ---------------------------------------------------------
        # Form checks
        # ---------------------------------------------------------

        self.warning_flags = []

        body_side = self.exercise_config.get(
            "body_side",
            "left"
        )

        # Elbow drift for biceps curls.
        if (
            "biceps_curl"
            in
            self.active_exercise_id
        ):

            shoulder_x = float(p1[0])
            elbow_x = float(p2[0])

            drift = abs(
                elbow_x - shoulder_x
            )

            if drift > 45:

                side_str = (
                    "arm"
                    if body_side == "both"
                    else f"{body_side} arm"
                )

                self.warning_flags.append(
                    "Elbow drifting! "
                    f"Keep {side_str} pinned to your torso."
                )

                self.current_feedback = (
                    f"Keep {side_str} elbow "
                    "steady against body."
                )

        # Bilateral symmetry.
        if is_bilateral:

            symmetry_diff = abs(
                self.left_angle
                -
                self.right_angle
            )

            if symmetry_diff > 22.0:

                self.warning_flags.append(
                    "Asymmetric effort! "
                    "Equalize left and right arm movement."
                )

        # Shoulder alignment.
        if len(all_kpts) > 6:

            l_shoulder_y = float(
                all_kpts[5][1]
            )

            r_shoulder_y = float(
                all_kpts[6][1]
            )

            if (
                abs(
                    l_shoulder_y
                    -
                    r_shoulder_y
                )
                > 35
            ):

                self.warning_flags.append(
                    "Shoulder uneven! "
                    "Keep spine upright."
                )

    # =============================================================
    # REP QUALITY
    # =============================================================

    def _evaluate_rep_quality(
        self,
        angle,
        p1,
        p2,
        p3,
        all_kpts
    ):
        """Calculate a conservative 0-100 rep score."""

        # Reserved for future scoring rules.
        del angle
        del p1
        del p2
        del p3
        del all_kpts

        base_score = 100.0

        if self.warning_flags:

            base_score -= (
                len(self.warning_flags)
                * 15.0
            )

        rep_duration = (
            time.time()
            -
            self.last_rep_time
        )

        if rep_duration < 1.0:
            base_score -= 10.0

        return float(
            max(
                50.0,
                min(
                    100.0,
                    base_score
                )
            )
        )

    # =============================================================
    # DRAW OVERLAY
    # =============================================================

    def _draw_overlay(
        self,
        frame,
        keypoints,
        keypoints_conf,
        p1,
        p2,
        p3,
        angle
    ):
        """Draw skeleton, joint angle, telemetry and feedback."""

        overlay = frame.copy()

        # COCO skeleton.
        skeleton_pairs = [
            (5, 7),
            (7, 9),

            (6, 8),
            (8, 10),

            (5, 6),

            (5, 11),
            (6, 12),

            (11, 12),

            (11, 13),
            (13, 15),

            (12, 14),
            (14, 16)
        ]

        # ---------------------------------------------------------
        # Skeleton lines
        # ---------------------------------------------------------

        for idx_a, idx_b in skeleton_pairs:

            pt_a = (
                int(keypoints[idx_a][0]),
                int(keypoints[idx_a][1])
            )

            pt_b = (
                int(keypoints[idx_b][0]),
                int(keypoints[idx_b][1])
            )

            conf_a = float(
                keypoints_conf[idx_a]
            )

            conf_b = float(
                keypoints_conf[idx_b]
            )

            if (
                conf_a > 0.3
                and
                conf_b > 0.3
            ):

                cv2.line(
                    overlay,
                    pt_a,
                    pt_b,
                    (214, 119, 90),
                    3
                )

        # ---------------------------------------------------------
        # Active joint
        # ---------------------------------------------------------

        pt1 = (
            int(p1[0]),
            int(p1[1])
        )

        pt2 = (
            int(p2[0]),
            int(p2[1])
        )

        pt3 = (
            int(p3[0]),
            int(p3[1])
        )

        cv2.line(
            overlay,
            pt1,
            pt2,
            (129, 185, 16),
            4
        )

        cv2.line(
            overlay,
            pt2,
            pt3,
            (129, 185, 16),
            4
        )

        # ---------------------------------------------------------
        # Keypoint nodes
        # ---------------------------------------------------------

        for i, pt in enumerate(
            keypoints[:17]
        ):

            cx = int(pt[0])
            cy = int(pt[1])

            conf_val = float(
                keypoints_conf[i]
            )

            if conf_val > 0.3:

                cv2.circle(
                    overlay,
                    (cx, cy),
                    6,
                    (255, 255, 255),
                    -1
                )

                cv2.circle(
                    overlay,
                    (cx, cy),
                    8,
                    (214, 119, 90),
                    2
                )

            else:

                cv2.circle(
                    overlay,
                    (cx, cy),
                    4,
                    (0, 165, 255),
                    -1
                )

        # Active joint center.
        cv2.circle(
            overlay,
            pt2,
            12,
            (16, 185, 129),
            -1
        )

        # ---------------------------------------------------------
        # Angle arc
        # ---------------------------------------------------------

        try:

            radius = 35

            start_angle = math.degrees(
                math.atan2(
                    pt1[1] - pt2[1],
                    pt1[0] - pt2[0]
                )
            )

            end_angle = math.degrees(
                math.atan2(
                    pt3[1] - pt2[1],
                    pt3[0] - pt2[0]
                )
            )

            cv2.ellipse(
                overlay,
                pt2,
                (radius, radius),
                0,
                start_angle,
                end_angle,
                (16, 185, 129),
                2
            )

        except Exception:
            pass

        # Angle text.
        angle_text = (
            f"{int(round(angle))} deg"
        )

        cv2.putText(
            overlay,
            angle_text,
            (
                pt2[0] + 15,
                pt2[1] - 15
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # ---------------------------------------------------------
        # Header
        # ---------------------------------------------------------

        cv2.rectangle(
            overlay,
            (0, 0),
            (
                frame.shape[1],
                70
            ),
            (30, 30, 30),
            -1
        )

        exercise_name = (
            self.exercise_config.get(
                "name",
                self.active_exercise_id
            )
        )

        header_str = (
            f"EXERCISE: "
            f"{exercise_name.upper()}"
        )

        reps_str = (
            f"REPS: "
            f"{self.rep_count}/"
            f"{self.target_reps}"
        )

        acc_str = (
            f"FORM ACCURACY: "
            f"{int(self.form_accuracy_score)}%"
        )

        cv2.putText(
            overlay,
            header_str,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (90, 190, 240),
            2
        )

        cv2.putText(
            overlay,
            reps_str,
            (20, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        acc_x = max(
            20,
            frame.shape[1] - 270
        )

        cv2.putText(
            overlay,
            acc_str,
            (acc_x, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (100, 230, 160),
            2
        )

        # ---------------------------------------------------------
        # Footer feedback
        # ---------------------------------------------------------

        if self.warning_flags:

            footer_color = (
                60,
                80,
                220
            )

            feedback = (
                self.warning_flags[0]
            )

        else:

            footer_color = (
                80,
                180,
                120
            )

            feedback = (
                self.current_feedback
            )

        max_chars = max(
            30,
            frame.shape[1] // 10
        )

        feedback = str(
            feedback
        )[:max_chars]

        footer_y = (
            frame.shape[0] - 20
        )

        cv2.putText(
            overlay,
            feedback,
            (20, footer_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            footer_color,
            2,
            cv2.LINE_AA
        )

        # ---------------------------------------------------------
        # Blend
        # ---------------------------------------------------------

        output = cv2.addWeighted(
            overlay,
            0.85,
            frame,
            0.15,
            0
        )

        return output

    # =============================================================
    # SIMULATED POSE FALLBACK
    # =============================================================

    def _generate_simulated_keypoints(
        self,
        width,
        height
    ):
        """
        Generate synthetic COCO keypoints.

        The left wrist rotates around the left elbow so the calculated
        elbow angle behaves more like an actual biceps curl.
        """

        self.sim_phase += 0.08

        center_x = width // 2
        center_y = height // 2

        sin_val = math.sin(
            self.sim_phase
        )

        body_scale = (
            min(width, height)
            / 600.0
        )

        body_scale = max(
            0.55,
            min(
                body_scale,
                1.35
            )
        )

        # ---------------------------------------------------------
        # Head
        # ---------------------------------------------------------

        head_y = (
            center_y
            -
            int(180 * body_scale)
        )

        # ---------------------------------------------------------
        # Shoulders
        # ---------------------------------------------------------

        left_shoulder = np.array(
            [
                center_x
                -
                int(70 * body_scale),

                center_y
                -
                int(110 * body_scale)
            ],
            dtype=np.float32
        )

        right_shoulder = np.array(
            [
                center_x
                +
                int(70 * body_scale),

                center_y
                -
                int(110 * body_scale)
            ],
            dtype=np.float32
        )

        # ---------------------------------------------------------
        # Elbows
        # ---------------------------------------------------------

        left_elbow = np.array(
            [
                center_x
                -
                int(85 * body_scale),

                center_y
                -
                int(10 * body_scale)
            ],
            dtype=np.float32
        )

        right_elbow = np.array(
            [
                center_x
                +
                int(85 * body_scale),

                center_y
                -
                int(10 * body_scale)
            ],
            dtype=np.float32
        )

        # ---------------------------------------------------------
        # Left biceps curl
        # ---------------------------------------------------------

        curl_angle_deg = (
            55.0
            +
            65.0 * sin_val
        )

        curl_angle = math.radians(
            curl_angle_deg
        )

        forearm_length = (
            95.0 * body_scale
        )

        left_wrist = np.array(
            [
                left_elbow[0]
                +
                forearm_length
                *
                math.cos(curl_angle),

                left_elbow[1]
                -
                forearm_length
                *
                math.sin(curl_angle)
            ],
            dtype=np.float32
        )

        # ---------------------------------------------------------
        # Right arm
        # ---------------------------------------------------------

        right_wrist = np.array(
            [
                center_x
                +
                int(75 * body_scale),

                center_y
                +
                int(80 * body_scale)
            ],
            dtype=np.float32
        )

        # ---------------------------------------------------------
        # Hips
        # ---------------------------------------------------------

        left_hip = np.array(
            [
                center_x
                -
                int(45 * body_scale),

                center_y
                +
                int(90 * body_scale)
            ],
            dtype=np.float32
        )

        right_hip = np.array(
            [
                center_x
                +
                int(45 * body_scale),

                center_y
                +
                int(90 * body_scale)
            ],
            dtype=np.float32
        )

        # ---------------------------------------------------------
        # Knees
        # ---------------------------------------------------------

        left_knee = np.array(
            [
                center_x
                -
                int(50 * body_scale),

                center_y
                +
                int(190 * body_scale)
            ],
            dtype=np.float32
        )

        right_knee = np.array(
            [
                center_x
                +
                int(50 * body_scale),

                center_y
                +
                int(190 * body_scale)
            ],
            dtype=np.float32
        )

        # ---------------------------------------------------------
        # Ankles
        # ---------------------------------------------------------

        left_ankle = np.array(
            [
                center_x
                -
                int(55 * body_scale),

                center_y
                +
                int(290 * body_scale)
            ],
            dtype=np.float32
        )

        right_ankle = np.array(
            [
                center_x
                +
                int(55 * body_scale),

                center_y
                +
                int(290 * body_scale)
            ],
            dtype=np.float32
        )

        # ---------------------------------------------------------
        # COCO keypoints
        # ---------------------------------------------------------

        kpts = np.array(
            [
                # 0 Nose
                [
                    center_x,
                    head_y
                ],

                # 1 Left eye
                [
                    center_x
                    -
                    int(10 * body_scale),
                    head_y
                    -
                    int(10 * body_scale)
                ],

                # 2 Right eye
                [
                    center_x
                    +
                    int(10 * body_scale),
                    head_y
                    -
                    int(10 * body_scale)
                ],

                # 3 Left ear
                [
                    center_x
                    -
                    int(25 * body_scale),
                    head_y
                    +
                    int(5 * body_scale)
                ],

                # 4 Right ear
                [
                    center_x
                    +
                    int(25 * body_scale),
                    head_y
                    +
                    int(5 * body_scale)
                ],

                # 5 Left shoulder
                left_shoulder,

                # 6 Right shoulder
                right_shoulder,

                # 7 Left elbow
                left_elbow,

                # 8 Right elbow
                right_elbow,

                # 9 Left wrist
                left_wrist,

                # 10 Right wrist
                right_wrist,

                # 11 Left hip
                left_hip,

                # 12 Right hip
                right_hip,

                # 13 Left knee
                left_knee,

                # 14 Right knee
                right_knee,

                # 15 Left ankle
                left_ankle,

                # 16 Right ankle
                right_ankle
            ],
            dtype=np.float32
        )

        # Keep points inside the image.
        kpts[:, 0] = np.clip(
            kpts[:, 0],
            0,
            max(width - 1, 0)
        )

        kpts[:, 1] = np.clip(
            kpts[:, 1],
            0,
            max(height - 1, 0)
        )

        conf = np.ones(
            17,
            dtype=np.float32
        )

        return kpts, conf


# =====================================================================
# STANDALONE TEST
# =====================================================================

if __name__ == "__main__":

    print(
        "[TEST] Initializing PostureEngine..."
    )

    engine = PostureEngine()

    # Synthetic camera frame.
    test_frame = np.zeros(
        (720, 1280, 3),
        dtype=np.uint8
    )

    telemetry, output = (
        engine.process_frame(
            test_frame
        )
    )

    print(
        "[TEST] Engine initialized successfully."
    )

    print(
        "[TEST] Model loaded:",
        telemetry["model_loaded"]
    )

    print(
        "[TEST] Exercise:",
        telemetry["exercise_name"]
    )

    print(
        "[TEST] Angle:",
        telemetry["current_angle"]
    )

    print(
        "[TEST] Stage:",
        telemetry["stage"]
    )

    print(
        "[TEST] Reps:",
        telemetry["rep_count"]
    )

    print(
        "[TEST] Output frame shape:",
        output.shape
    )
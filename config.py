"""
Configuration & 10 Post-Op Recovery Exercise Definitions for AI Physiotherapist.
Defines joint angle math targets, keypoint indices (COCO format), posture thresholds,
and clinical recommendations for each exercise mode.
"""

import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'ai-physio-secret-key-2026')
    DATABASE_PATH = os.path.join(os.path.dirname(__file__), 'physio_tracker.db')
    YOLO_MODEL_NAME = 'yolov8n-pose.pt'

# COCO Keypoint Indices Mapping for YOLO-Pose:
# 0: Nose, 1: L_Eye, 2: R_Eye, 3: L_Ear, 4: R_Ear
# 5: L_Shoulder, 6: R_Shoulder, 7: L_Elbow, 8: R_Elbow
# 9: L_Wrist, 10: R_Wrist, 11: L_Hip, 12: R_Hip
# 13: L_Knee, 14: R_Knee, 15: L_Ankle, 16: R_Ankle

EXERCISES = {
    "left_biceps_curl": {
        "id": "left_biceps_curl",
        "name": "Left Arm Biceps Curl",
        "category": "Upper Extremity Rehab",
        "description": "Post-elbow or shoulder recovery. Strengthens left biceps brachii with controlled elbow flexion.",
        "icon": "bi-flex",
        "body_side": "left",
        "joints": {
            "p1": 5,   # Left Shoulder
            "p2": 7,   # Left Elbow (Vertex)
            "p3": 9    # Left Wrist
        },
        "angle_rest": 155.0,        # Extension
        "angle_target": 40.0,       # Flexion peak
        "angle_threshold_start": 130.0,
        "angle_threshold_finish": 55.0,
        "recommended_reps": 10,
        "recommended_sets": 3,
        "posture_tips": [
            "Keep left elbow pinned to your side.",
            "Avoid swinging your torso or using momentum.",
            "Lower the weight smoothly over 3 seconds."
        ],
        "form_checks": {
            "elbow_drift_limit": 30.0 # Warning if elbow strays away from hip x-coordinate
        }
    },

    "right_biceps_curl": {
        "id": "right_biceps_curl",
        "name": "Right Arm Biceps Curl",
        "category": "Upper Extremity Rehab",
        "description": "Post-elbow or shoulder recovery. Strengthens right biceps brachii with controlled elbow flexion.",
        "icon": "bi-flex",
        "body_side": "right",
        "joints": {
            "p1": 6,   # Right Shoulder
            "p2": 8,   # Right Elbow (Vertex)
            "p3": 10   # Right Wrist
        },
        "angle_rest": 155.0,
        "angle_target": 40.0,
        "angle_threshold_start": 130.0,
        "angle_threshold_finish": 55.0,
        "recommended_reps": 10,
        "recommended_sets": 3,
        "posture_tips": [
            "Keep right elbow pinned close to torso.",
            "Maintain an upright spinal posture.",
            "Complete full extension at bottom."
        ],
        "form_checks": {
            "elbow_drift_limit": 30.0
        }
    },

    "bilateral_biceps_curl": {
        "id": "bilateral_biceps_curl",
        "name": "Bilateral Biceps Curl",
        "category": "Upper Extremity Rehab",
        "description": "Synchronized dual-arm flexion for upper body strength and bilateral symmetry assessment.",
        "icon": "bi-symmetry",
        "body_side": "both",
        "joints_left": {"p1": 5, "p2": 7, "p3": 9},
        "joints_right": {"p1": 6, "p2": 8, "p3": 10},
        "angle_rest": 155.0,
        "angle_target": 40.0,
        "angle_threshold_start": 130.0,
        "angle_threshold_finish": 55.0,
        "recommended_reps": 8,
        "recommended_sets": 3,
        "posture_tips": [
            "Curl both arms evenly without leaning left or right.",
            "Focus on equal velocity in both arms."
        ],
        "form_checks": {
            "symmetry_diff_limit": 25.0
        }
    },

    "seated_knee_extension": {
        "id": "seated_knee_extension",
        "name": "Seated Knee Extension (Right/Left)",
        "category": "Lower Extremity & ACL Rehab",
        "description": "Essential post-Knee Replacement / ACL rehab exercise. Strengthens quadriceps muscles safely.",
        "icon": "bi-leg",
        "body_side": "right",
        "joints": {
            "p1": 12,  # Right Hip
            "p2": 14,  # Right Knee (Vertex)
            "p3": 16   # Right Ankle
        },
        "angle_rest": 90.0,         # Bent seated knee
        "angle_target": 165.0,      # Straight leg extension
        "angle_threshold_start": 105.0,
        "angle_threshold_finish": 150.0,
        "recommended_reps": 12,
        "recommended_sets": 3,
        "posture_tips": [
            "Sit tall with back supported.",
            "Straighten leg fully and hold for 1 second at top.",
            "Lower slowly back to 90 degrees."
        ],
        "form_checks": {
            "slouch_limit": 20.0
        }
    },

    "front_arm_raise": {
        "id": "front_arm_raise",
        "name": "Front Arm Shoulder Flexion",
        "category": "Shoulder Mobility",
        "description": "Post-rotator cuff or shoulder surgery rehab. Improves forward arm elevation and shoulder range of motion.",
        "icon": "bi-arrow-up-circle",
        "body_side": "left",
        "joints": {
            "p1": 11,  # Left Hip
            "p2": 5,   # Left Shoulder (Vertex)
            "p3": 9    # Left Wrist
        },
        "angle_rest": 20.0,         # Arm down
        "angle_target": 140.0,      # Arm raised overhead
        "angle_threshold_start": 40.0,
        "angle_threshold_finish": 120.0,
        "recommended_reps": 10,
        "recommended_sets": 2,
        "posture_tips": [
            "Keep elbow straight during raising motion.",
            "Do not arch lower back as arm reaches top.",
            "Raise only within painless range."
        ],
        "form_checks": {
            "back_arch_limit": 15.0
        }
    },

    "side_lateral_raise": {
        "id": "side_lateral_raise",
        "name": "Side Shoulder Abduction",
        "category": "Shoulder Mobility",
        "description": "Post-op deltoid and rotator cuff rehab. Strengthens side shoulder stabilisers.",
        "icon": "bi-arrows-expand",
        "body_side": "right",
        "joints": {
            "p1": 12,  # Right Hip
            "p2": 6,   # Right Shoulder (Vertex)
            "p3": 10   # Right Wrist
        },
        "angle_rest": 20.0,         # Arm down by hip
        "angle_target": 90.0,       # Arm horizontal to shoulder level
        "angle_threshold_start": 35.0,
        "angle_threshold_finish": 80.0,
        "recommended_reps": 10,
        "recommended_sets": 3,
        "posture_tips": [
            "Raise right arm laterally to shoulder height.",
            "Pause briefly at peak shoulder level.",
            "Keep shoulders relaxed away from ears."
        ],
        "form_checks": {
            "shoulder_shrug_limit": 15.0
        }
    },

    "overhead_shoulder_press": {
        "id": "overhead_shoulder_press",
        "name": "Overhead Shoulder Press",
        "category": "Upper Body Functional Rehab",
        "description": "Restores overhead reaching strength and scapular movement pattern.",
        "icon": "bi-arrow-up-square",
        "body_side": "both",
        "joints": {
            "p1": 5,   # Left Shoulder
            "p2": 7,   # Left Elbow (Vertex)
            "p3": 9    # Left Wrist
        },
        "angle_rest": 75.0,         # Elbow bent at shoulder height
        "angle_target": 160.0,      # Press overhead straight
        "angle_threshold_start": 95.0,
        "angle_threshold_finish": 145.0,
        "recommended_reps": 8,
        "recommended_sets": 3,
        "posture_tips": [
            "Press arms straight up vertically.",
            "Keep neck neutral and look straight ahead.",
            "Avoid hyperextending low back."
        ],
        "form_checks": {
            "neck_strain_limit": 20.0
        }
    },

    "standing_mini_squat": {
        "id": "standing_mini_squat",
        "name": "Standing Mini-Squat",
        "category": "Lower Body & Joint Rehab",
        "description": "Post-hip/knee replacement rehabilitation. Improves quad strength and knee stability safely.",
        "icon": "bi-person-standing",
        "body_side": "both",
        "joints": {
            "p1": 12,  # Right Hip
            "p2": 14,  # Right Knee (Vertex)
            "p3": 16   # Right Ankle
        },
        "angle_rest": 170.0,        # Standing tall
        "angle_target": 115.0,      # Controlled partial squat depth
        "angle_threshold_start": 150.0,
        "angle_threshold_finish": 125.0,
        "recommended_reps": 10,
        "recommended_sets": 3,
        "posture_tips": [
            "Bend knees smoothly as if sitting on a high chair.",
            "Keep knees aligned over toes (avoid knee caving).",
            "Maintain chest open and spine straight."
        ],
        "form_checks": {
            "knee_valgus_limit": 15.0
        }
    },

    "trunk_side_bend": {
        "id": "trunk_side_bend",
        "name": "Trunk Side Bend & Spine Flex",
        "category": "Core & Spinal Rehab",
        "description": "Post-lumbar or thoracic rehab. Restores spinal lateral flexibility and core stability.",
        "icon": "bi-arrow-left-right",
        "body_side": "torso",
        "joints": {
            "p1": 0,   # Nose / Neck line
            "p2": 11,  # Left Hip (Vertex)
            "p3": 15   # Left Ankle
        },
        "angle_rest": 175.0,        # Vertical spine posture
        "angle_target": 150.0,      # Gentle lateral tilt (15-25 deg bend)
        "angle_threshold_start": 165.0,
        "angle_threshold_finish": 155.0,
        "recommended_reps": 8,
        "recommended_sets": 2,
        "posture_tips": [
            "Slide hand down outer thigh while bending sideways.",
            "Do not twist or lean forward.",
            "Keep feet grounded firmly."
        ],
        "form_checks": {
            "forward_twist_limit": 20.0
        }
    },

    "standing_hip_abduction": {
        "id": "standing_hip_abduction",
        "name": "Standing Hip Abduction",
        "category": "Hip & Pelvic Rehab",
        "description": "Post-hip arthroplasty / gluteus medius rehab. Improves gait stability and hip strength.",
        "icon": "bi-person-walking",
        "body_side": "right",
        "joints": {
            "p1": 11,  # Left Hip
            "p2": 12,  # Right Hip (Vertex)
            "p3": 14   # Right Knee
        },
        "angle_rest": 90.0,         # Straight vertical legs
        "angle_target": 125.0,      # Leg lifted outwards 35 degrees
        "angle_threshold_start": 100.0,
        "angle_threshold_finish": 118.0,
        "recommended_reps": 10,
        "recommended_sets": 3,
        "posture_tips": [
            "Keep body upright while lifting right leg outwards.",
            "Avoid tilting pelvis to compensate.",
            "Pause at peak lift before lowering leg."
        ],
        "form_checks": {
            "pelvic_tilt_limit": 15.0
        }
    }
}

🧍 AI Pose Corrector

AI Pose Corrector is a real-time posture analysis application that uses YOLO Pose and Flask to detect body keypoints, analyze posture, and provide corrective feedback.

The project combines computer vision and AI-powered pose estimation to help users identify incorrect posture and improve their body alignment.

✨ Features

* 🎥 Real-time camera-based pose detection
* 🧍 Human pose estimation using YOLO Pose
* 📍 Detection of body keypoints
* 📐 Posture and body alignment analysis
* ⚠️ Identification of incorrect posture
* 💡 Real-time corrective feedback
* 🌐 Flask-based web application

🛠️ Tech Stack

* Python — Core application logic
* Flask — Web application backend
* YOLO Pose — Human pose estimation and keypoint detection
* HTML / CSS / JavaScript — Frontend interface

🧠 How It Works

The application processes the user’s camera feed and uses the YOLO Pose model to identify important body keypoints.

Camera Feed
     ↓
Video Frames
     ↓
YOLO Pose Model
     ↓
Body Keypoint Detection
     ↓
Posture Analysis
     ↓
Posture Classification
     ↓
Corrective Feedback

Pose Detection

YOLO Pose detects key body points such as:

* Head
* Shoulders
* Elbows
* Wrists
* Hips
* Knees
* Ankles

These keypoints are then used by the posture analysis logic to determine whether the user’s posture is correct or incorrect.

📁 Project Structure

AI-Pose-Corrector/
│
├── app.py
├── posture_engine.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── templates/
│   └── index.html
│
└── static/
    ├── css/
    └── js/

🚀 Getting Started

1. Clone the Repository

git clone https://github.com/shaikroshni2008-coder/AI-Pose-Corrector.git

Navigate to the project directory:

cd AI-Pose-Corrector

2. Create a Virtual Environment

python3 -m venv venv

Activate the virtual environment on macOS/Linux:

source venv/bin/activate

3. Install Dependencies

pip install -r requirements.txt

4. Run the Application

python app.py

The Flask application will start locally.

Open the URL displayed in your terminal, usually:

http://127.0.0.1:5000

📋 Requirements

Before running the project, make sure you have:

* Python 3.9+
* A working webcam
* Required Python dependencies
* A compatible YOLO Pose model

Install the dependencies with:

pip install -r requirements.txt

🎯 Use Cases

AI Pose Corrector can be used for:

* 🪑 Desk and sitting posture monitoring
* 💻 Computer users
* 🏋️ Exercise posture analysis
* 🧘 Fitness and wellness applications
* 🏠 Home posture monitoring
* 🎓 Computer vision and AI projects

🔮 Future Improvements

* Support for multiple exercises
* More posture detection categories
* Personalized posture recommendations
* Posture history and analytics
* Voice-based corrective feedback
* Improved pose detection accuracy
* Mobile application
* Personalized AI posture coaching

⚠️ Disclaimer

AI Pose Corrector is an educational computer-vision project intended to provide general posture feedback. It is not a medical diagnostic tool and should not replace professional medical advice.

👩‍💻 Author

Roshni Shaik

B.Tech — Artificial Intelligence

⭐ If you find this project useful or interesting, consider giving the repository a star!

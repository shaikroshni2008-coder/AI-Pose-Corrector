🧘 AI Pose Corrector

An AI-powered posture analysis and correction system that uses computer vision and pose estimation to detect body posture in real time and provide feedback to help users maintain better posture.

✨ Features

* 🎥 Real-time pose detection
* 🧍 Posture analysis using body landmarks
* ⚠️ Detects incorrect posture
* ✅ Provides posture correction feedback
* 📊 Analyzes body alignment
* 💻 Web-based interface
* 🤖 Powered by AI and computer vision

🛠️ Tech Stack

* Python
* Flask
* OpenCV
* MediaPipe
* NumPy
* HTML / CSS / JavaScript

🧠 How It Works

The system uses a webcam to capture the user’s movements.

1. 📷 The webcam captures the user’s posture.
2. 🔍 MediaPipe detects important body landmarks.
3. 📐 The system calculates angles and body alignment.
4. 🧠 The posture engine analyzes the detected landmarks.
5. ⚠️ Incorrect posture is identified.
6. 💡 The application provides real-time corrective feedback.

Basic Workflow

Webcam
   ↓
Video Frame
   ↓
Pose Detection
   ↓
Body Landmarks
   ↓
Angle & Alignment Calculation
   ↓
Posture Analysis
   ↓
Feedback

📁 Project Structure

AI-Pose-Corrector/
│
├── app.py
├── posture_engine.py
├── requirements.txt
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── venv/
│
├── .gitignore
└── README.md

venv/ should remain local and should not be uploaded to GitHub.

🚀 Getting Started

1. Clone the repository

git clone https://github.com/shaikroshni2008-coder/AI-Pose-Corrector.git

Navigate into the project:

cd AI-Pose-Corrector

2. Create a virtual environment

python3 -m venv venv

Activate it on macOS/Linux:

source venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Run the application

python app.py

Open the local URL shown in the terminal, usually:

http://127.0.0.1:5000

📋 Requirements

Make sure you have:

* Python 3.9+
* A working webcam
* Internet connection for installing dependencies

Install the required Python packages using:

pip install -r requirements.txt

🎯 Use Cases

AI Pose Corrector can be useful for:

* 🪑 Office and desk posture monitoring
* 💻 Computer users
* 🏋️ Exercise posture checking
* 🧘 Fitness and wellness applications
* 🎓 AI/computer vision learning projects
* 🏠 Home posture monitoring

🔮 Future Improvements

Planned improvements include:

* Support for multiple exercise/posture types
* More accurate posture classification
* Posture history and analytics
* Personalized recommendations
* Voice-based feedback
* Mobile application
* Improved UI/UX
* AI-based personalized posture coaching

⚠️ Disclaimer

AI Pose Corrector is an educational and computer-vision project designed to provide general posture feedback. It is not a medical diagnostic tool and should not replace professional medical advice.

👩‍💻 Author

Roshni Shaik

B.Tech — Artificial Intelligence

⭐ If you find this project interesting, consider giving the repository a star!

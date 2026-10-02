# 🚀 Computer Vision Intelligence Suite

<p align="center">
  <img src="images/banner.png" alt="Computer Vision Intelligence Suite Banner" width="100%">
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)
![YOLO](https://img.shields.io/badge/YOLO-Object%20Detection-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

</p>

---

# 📖 Overview

**Computer Vision Intelligence Suite** is a collection of advanced **Python-based Computer Vision applications** developed for real-world visual intelligence and automated analytics.

The project demonstrates how modern AI and Computer Vision techniques can transform raw video streams into meaningful intelligence by detecting objects, tracking movement, recognizing patterns, generating analytics, and producing actionable reports.

The suite currently contains three independent intelligence systems:

* ✈️ Air Combat Intelligence System
* 🚗 License Plate Intelligence System
* 🛫 Airport Runway Intelligence System

Each project is modular and can run independently while sharing common computer vision components.

---

# 🎯 Objectives

The primary goals of this project are:

* Detect important objects in images and videos
* Perform real-time object tracking
* Generate movement trajectories
* Produce intelligence dashboards
* Generate reports automatically
* Visualize object behavior
* Demonstrate practical AI applications
* Provide educational examples for Computer Vision research

---

# 🖼 Project Architecture

```
Video / Camera Input
        │
        ▼
Frame Extraction
        │
        ▼
Object Detection (YOLO)
        │
        ▼
Object Tracking
        │
        ▼
OCR / Classification
        │
        ▼
Analytics Engine
        │
        ├───────────────► Dashboard
        ├───────────────► Reports
        ├───────────────► Heatmaps
        └───────────────► Annotated Video
```

---

# 📂 Repository Structure

```
Computer-Vision-Projects/
│
├── air_combat_intelligence/
│   ├── src/
│   │   ├── __init__.py
│   │   └── main.py
│   └── README.md
│
├── airport_runway_intelligence/
│   ├── src/
│   │   ├── __init__.py
│   │   └── main.py
│   └── README.md
│
├── license_plate_intelligence/
│   ├── src/
│   │   ├── __init__.py
│   │   └── main.py
│   └── README.md
│
├── common/
│   ├── __init__.py
│   ├── config.py
│   ├── tracker.py
│   ├── utils.py
│   └── visualization.py
│
├── web/
│   ├── server.py
│   ├── templates/index.html
│   └── static/
│       ├── css/style.css
│       └── js/app.js
│
├── models/
│   └── license_plate_detector.pt
│
├── outputs/
│   ├── air_combat/
│   ├── airport_runway/
│   └── license_plate/
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🛠 Technology Stack

| Category             | Technologies         |
| -------------------- | -------------------- |
| Programming          | Python 3.9+          |
| Computer Vision      | OpenCV               |
| Detection            | YOLOv8 (Ultralytics) |
| OCR                  | EasyOCR              |
| Numerical Computing  | NumPy                |
| Data Analysis        | Pandas               |
| Scientific Computing | SciPy                |
| Visualization        | Matplotlib           |
| Image Processing     | Pillow               |
| Tracking             | Unified Multi-Tracker|
| Web & Dashboard      | FastAPI, Uvicorn     |

---

# ⚙ Installation

## Clone Repository

```bash
git clone https://github.com/Livesh28/Computer-Vision-Projects.git
cd Computer-Vision-Projects
```

---

## Create Virtual Environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux / macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶ Usage

### 🌐 1. Interactive Tactical Web Dashboard & API (Recommended)

Launch the production web dashboard locally:

```bash
python main.py --serve --port 8080
```
Open **[http://localhost:8080](http://localhost:8080)** in your browser to run live simulations, upload custom videos, view tactical HUD playback, and download generated telemetry reports and heatmaps!

---

### 💻 2. Unified Command-Line Interface (CLI)

#### Air Combat Intelligence System:
```bash
# Run on a video file:
python main.py --system combat --input sample.mp4

# Run instant synthetic tactical simulation:
python main.py --system combat --synthetic
```

#### Airport Runway Surface Perception:
```bash
# Run on a video file:
python main.py --system runway --input airport.mp4

# Run instant synthetic runway simulation:
python main.py --system runway --synthetic
```

#### License Plate Intelligence (ANPR):
```bash
# Run on a video file or webcam:
python main.py --system anpr --input traffic.mp4

# Run instant synthetic traffic simulation:
python main.py --system anpr --synthetic
```

---

### 📦 3. Modular Direct Script Execution

You can also run each standalone module directly:

```bash
python air_combat_intelligence/src/main.py --input sample.mp4
python airport_runway_intelligence/src/main.py --input airport.mp4
python license_plate_intelligence/src/main.py --input traffic.mp4
```

---

# 📊 Generated Outputs

The suite automatically generates:

* Annotated Videos
* CSV Reports
* JSON Reports
* Excel Reports
* Detection Logs
* Heatmaps
* Analytics Dashboard
* Charts
* Object Tracking History

---

# 📈 Performance Features

✅ Real-Time Detection

✅ Multi-Object Tracking

✅ High Accuracy OCR

✅ Modular Architecture

✅ Easy Customization

✅ Lightweight Design

✅ Scalable Pipelines

✅ Research Friendly

---

# 📷 Sample Results

```
Original Video
      │
      ▼
YOLO Detection
      │
      ▼
Object Tracking
      │
      ▼
Analytics
      │
      ▼
Heatmaps
      │
      ▼
Dashboard
      │
      ▼
Final Annotated Video
```

---

# 🚀 Future Roadmap

* YOLOv11 Integration
* ByteTrack Support
* StrongSORT Tracking
* Real-Time RTSP Streaming
* Drone Surveillance Module
* Maritime Surveillance
* Face Recognition Module
* Industrial Safety Monitoring
* Smart City Traffic Analytics
* REST API
* Cloud Deployment
* Docker Support
* Kubernetes Deployment
* Mobile Dashboard
* AI Report Generation using LLMs

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Open a Pull Request

---

# 📄 License

This project is licensed under the MIT License.

---

# ⭐ Acknowledgements

Special thanks to the amazing open-source community:

* OpenCV
* Ultralytics YOLO
* NumPy
* Pandas
* SciPy
* Matplotlib
* Pillow
* EasyOCR
* Tesseract OCR
* Streamlit

---

# 💡 Applications

This project can be adapted for:

* Defense Intelligence
* Traffic Surveillance
* Airport Monitoring
* Border Security
* Smart Cities
* Industrial Automation
* Autonomous Vehicles
* Security Surveillance
* Research Projects
* Academic Learning

---

# 👨‍💻 Author

**Your Name**

GitHub: https://github.com/Livesh28

---

<p align="center">

### ⭐ If you found this project useful, please give it a Star!

**Made with ❤️ using Python, OpenCV, and Artificial Intelligence**

</p>

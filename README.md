# 🏏 BowlForm AI - Advanced Bowling Action Analysis System

![BowlForm AI Cover](https://via.placeholder.com/1200x400?text=BowlForm+AI+Advanced+Action+Analysis)

🔗 **Live Demo:** [https://bowling-frontend-vercel.vercel.app](https://bowling-frontend-vercel.vercel.app)

**BowlForm AI** is a state-of-the-art computer vision and machine learning-powered system designed to analyze cricket bowling actions. It verifies biomechanical parameters against ICC regulations (International Cricket Council) and provides AI-powered coaching feedback to help bowlers optimize their performance and prevent injuries.

## 📑 Table of Contents
1. [Introduction](#introduction)
2. [Key Features](#key-features)
3. [Technology Stack](#technology-stack)
4. [System Architecture](#system-architecture)
   - [Frontend Architecture](#frontend-architecture)
   - [Backend Architecture](#backend-architecture)
5. [Biomechanical Metrics Analyzed](#biomechanical-metrics-analyzed)
6. [ICC Compliance Rules](#icc-compliance-rules)
7. [Directory Structure](#directory-structure)
8. [Prerequisites](#prerequisites)
9. [Installation & Setup](#installation--setup)
   - [Local Development Setup](#local-development-setup)
   - [Docker Setup](#docker-setup)
10. [Environment Variables](#environment-variables)
11. [API Documentation](#api-documentation)
12. [Usage Guide](#usage-guide)
13. [Troubleshooting & FAQs](#troubleshooting--faqs)
14. [Contributing Developer Guidelines](#contributing-developer-guidelines)
15. [License](#license)
16. [Acknowledgments](#acknowledgments)

---

## 1. Introduction <a name="introduction"></a>

Cricket bowling is a highly technical physical motion that requires immense coordination. Slight deviations in form can result in severe injuries or result in an illegal delivery (often referred to as 'chucking').
BowlForm AI acts as an automated, highly precise digital coach that accepts video footage of a bowling delivery and identifies key events (e.g., arm horizontal, release point) using MediaPipe Pose estimation. By analyzing the angles and movements of the bowler's body, it provides:
- Real-time on-video HUD overlays.
- A biomechanical compliance check.
- Actionable AI-generated coaching tips.

---

## 2. Key Features <a name="key-features"></a>

- **Upload & Analyze**: Users can upload high-speed and standard video clips of their bowling action for immediate processing.
- **Biomechanical Angle Tracking**: Calculates core angles including elbow flexion, front knee extension, shoulder counter-rotation (SCR), hip-shoulder separation (HSS), and stride ratio.
- **Automated Phase Detection**: Intelligent finite-state machine tracks the bowling delivery through various phases: `idle` → `arm_horizontal` → `release`.
- **MediaPipe Pose Estimation**: 33-point 3D landmark tracking optimized for rapid CPU inference.
- **ICC Law 24 Verification**: Automatically checks elbow extension against the 15-degree limit rule for legal bowling.
- **Smart Asynchronous Backend**: Uses non-blocking threaded analysis for long videos, streaming updates via Server-Sent Events (SSE).
- **Dual Visual Outputs**: Generates both an annotated original video with a Heads-Up Display (HUD) and an isolated dark-mode skeleton video.
- **AI Coaching via OpenRouter**: Generates concise, highly targeted text feedback using an LLM.

---

## 3. Technology Stack <a name="technology-stack"></a>

### **Frontend**
- **HTML5 / CSS3 / JavaScript (Vanilla)**: Ensures a lightweight, dependency-free interface.
- **Tailwind CSS (or standard component CSS)**: Responsive, modern UI/UX design.
- **Server-Sent Events (SSE)**: Streams real-time processing progress directly to the browser UI without long-polling.

### **Backend**
- **Python 3.10+**: Core programming language.
- **Flask**: Lightweight WSGI web application framework for routing and API endpoints.
- **OpenCV (cv2)**: Video processing, frame manipulation, and HUD overlay rendering.
- **MediaPipe**: State-of-the-art pose estimation and landmark detection.
- **NumPy**: High-performance mathematical operations for geometry and angle calculations.
- **Python-Dotenv**: Secret management for external API Keys.
- **Requests**: HTTP client for communicating with external LLM APIs (OpenRouter).

---

## 4. System Architecture <a name="system-architecture"></a>

This application utilizes a decoupled client-server architecture. 

### Frontend Architecture <a name="frontend-architecture"></a>
The frontend operates purely in the browser. It consists of an `index.html` file housing the complete view structure, styled and enhanced with embedded JavaScript. 
- **Communication Layer**: Submits video files to the backend via `multipart/form-data` POST requests.
- **Progress Tracking Layer**: Establishes an `EventSource` connection to the backend `/progress/<job_id>` endpoint to listen for percentage updates and state changes.
- **Presentation Layer**: Receives final JSON metrics and updates DOM elements. Mounts `.webm` or `.mp4` video blobs into `<video>` tags for playback.

### Backend Architecture <a name="backend-architecture"></a>
The backend is a lightweight Flask application acting primarily as an orchestrator for computer vision logic.
- **Routing & State (app.py)**: Manages concurrent video uploads using thread-safe dictionaries to track job progress and results.
- **Vision & Logic Pipeline (backend.py)**:
  - Video decapsulation.
  - Phase state machine (`PhaseDetector`).
  - Rolling-window angle smoother (`AngleSmoother`).
  - Overlay Renderer (`draw_overlay`).
- **Storage Layer**: Processed videos and JSON payloads are temporarily saved onto the local filesystem or a bound Docker volume.

---

## 5. Biomechanical Metrics Analyzed <a name="biomechanical-metrics-analyzed"></a>

BowlForm AI deeply analyzes multiple kinematic sequences within the delivery:

1. **Elbow Flexion**: Monitored strictly from the "arm horizontal" phase until the "release" phase. Prevents illegal actions.
2. **Front Knee Angle**: Analyzed at the point of release. A hyper-extended knee (typically >175°) can cause ACL injuries, while a deeply bent knee (<130°) bleeds pace and momentum.
3. **Hip-Shoulder Separation (HSS)**: Analyzed on the transverse plane. An optimal HSS (between 15° and 45°) generates immense torque, defining the difference between an average bowler and a fast bowler.
4. **Shoulder Counter-Rotation (SCR)**: Tracked from delivery stride to release. SCR above 30° is flagged due to an elevated risk of lower lumbar stress fractures.
5. **Stride Length Ratio**: Extracted and normalized against the bowler's shoulder width. Ensures proper momentum transfer.

---

## 6. ICC Compliance Rules <a name="icc-compliance-rules"></a>

The International Cricket Council (ICC) enforces **Law 24**, specifically targeting unfair delivery actions. 
- **Rule Synopsis**: A bowler is permitted to bowl the ball with a bent arm, provided the elbow joint does not *straighten* by more than 15 degrees between the point where the bowling arm passes the horizontal and the point of ball release.
- **Implementation in Code**: 
  - The `PhaseDetector` captures `elbow_at_horizontal` and `elbow_at_release`.
  - The equation: `delta = elbow_at_release - elbow_at_horizontal`.
  - If `delta > 15.0`, the system explicitly flags the delivery as **ILLEGAL** and alters the HUD overlay accordingly.

---

## 7. Directory Structure <a name="directory-structure"></a>

```text
Bowling System/
└── v1/
    ├── frontend/
    │   └── index.html             # The primary Single Page Application template
    │
    └── backend/
        ├── app.py                 # Flask server, routing, and concurrent job tracking
        ├── backend.py             # Core MediaPipe logic, geometric calculations, AI queries
        ├── requirements.txt       # Python dependencies list
        ├── Dockerfile             # Container configuration for backend services
        ├── .env.example           # Example environment file (copy to .env)
        └── .env                   # Ignored file storing secrets (API_KEY)
```

---

## 8. Prerequisites <a name="prerequisites"></a>

Before you begin, ensure you have met the following requirements:
* **Python**: v3.10 or higher installed locally.
* **Node.js/npm**: (Optional) if serving frontend via a local dev server.
* **Docker**: (Optional) For containerized deployment.
* **OpenRouter Account**: You will need an API key to enable LLM-powered coaching tips.

---

## 9. Installation & Setup <a name="installation--setup"></a>

You can run the Bowling System locally or via Docker. The backend must be started before the frontend can process any videos.

### Local Development Setup <a name="local-development-setup"></a>

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/bowlform-ai.git
   cd bowlform-ai/v1/backend
   ```

2. **Set up a Python Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate       # On Linux/MacOS
   venv\Scripts\activate          # On Windows
   ```

3. **Install exactly pinned Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize Environment Variables**:
   ```bash
   # Create your .env file
   touch .env
   # Add your OpenRouter API Key to it (see Environment Variables section)
   ```

5. **Start the Flask Backend Server**:
   ```bash
   # Make sure you are in the backend directory
   flask run --host=0.0.0.0 --port=5000
   # OR
   python app.py
   ```

6. **Serve the Frontend**:
   Open a new terminal, navigate to the `frontend` folder, and serve `index.html`. You can use Python's built-in HTTP server:
   ```bash
   cd ../frontend
   python -m http.server 8000
   ```
   Now visit `http://localhost:8000` in your web browser.

### Docker Setup <a name="docker-setup"></a>

1. **Build the Docker Image**:
   ```bash
   cd v1/backend
   docker build -t bowlform-backend .
   ```

2. **Run the Docker Container**:
   ```bash
   docker run -p 5000:5000 --env-file .env bowlform-backend
   ```
   *Note: Ensure your `.env` file exists and contains the necessary API keys before running.*

---

## 10. Environment Variables <a name="environment-variables"></a>

To protect sensitive keys, the API configurations are loaded via standard `.env` files. Inside the `backend` directory, create a `.env` file and configure the following:

```env
# Server settings
FLASK_ENV=development
PORT=5000

# OpenRouter (DeepSeek LLM API connection)
API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxx
```

**Security Warning**: Never commit your `.env` file to version control. Let `.gitignore` handle it.

---

## 11. API Documentation <a name="api-documentation"></a>

The backend provides several JSON REST API endpoints and one EventStream endpoint.

### `POST /analyze`
- **Description**: Initiates the video analysis process.
- **Body**: `multipart/form-data` containing a `video` file.
- **Response**: `202 Accepted`
  ```json
  {
      "job_id": "uuid-1234-5678",
      "status": "processing"
  }
  ```

### `GET /progress/<job_id>`
- **Description**: Server-Sent Events (SSE) endpoint to listen for live processing progress.
- **Yields**: String payloads indicating percentages: `data: 45\n\n`

### `GET /results/<job_id>`
- **Description**: Fetches the finalized comprehensive JSON metrics once the job completes.
- **Response**: `200 OK`
  ```json
  {
      "status": "complete",
      "bowling_arm": "RIGHT",
      "icc_compliance": { ... },
      "ai_coaching_tips": "..."
  }
  ```

### `GET /video/annotated/<job_id>`
- **Description**: Returns the processed `.webm` video with HUD and tracked pose landmarks.
- **Response**: Blob payload (`video/webm`)

### `GET /video/skeleton/<job_id>`
- **Description**: Returns the processed `.webm` video containing only the dark-mode skeleton.
- **Response**: Blob payload (`video/webm`)

---

## 12. Usage Guide <a name="usage-guide"></a>

1. **Upload Video**: Click the large upload zone on the frontend webpage. For best results, use a high framerate video (60fps to 120fps) recorded from a completely side-on, 90-degree angle.
2. **Analysis Phase**: Wait as the progress bar fills up. The backend is analyzing each frame through MediaPipe.
3. **View Summary**: Once completed, the top panel will reveal your bowling arm, total frames, phase data, and a quick summary verdict.
4. **Playback**: 
   - **Original + HUD**: Displays the original video with joints overlaid in teal, accompanied by real-time floating metrics.
   - **Skeleton View**: Black background with isolated joint movements to eliminate visual noise.
5. **Review Tips**: The right-hand panel displays concise coaching tips generated by the LLM based directly on your biomechanical flaws.

---

## 13. Troubleshooting & FAQs <a name="troubleshooting--faqs"></a>

**Q: The video upload fails instantly.**
*A: Check that your Flask backend is running and that the `API_BASE` URL in `frontend/index.html` natively matches your current backend server route (e.g., `http://localhost:5000`). Make sure CORS is properly configured in `app.py`.*

**Q: The LLM coaching tips say "API key not configured."**
*A: Verify that the `.env` file exists in the `v1/backend/` directory, and that `python-dotenv` has been installed. Ensure `API_KEY` exactly matches your OpenRouter key.*

**Q: The phase detection does not detect the "release" phase.**
*A: The system relies on tracking the wrist's peak height relative to the shoulder. If the bowler's action is cut off by the top of the video frame, it will fail. Record with enough vertical headroom.*

**Q: My MediaPipe inference is too slow.**
*A: `backend.py` limits `MAX_WIDTH` to 640px. Make sure `model_complexity=1`. Processing video locally relies heavily on your CPU architecture. M-series Macs and modern x86 Core i7/i9s process rapidly. Threaded execution may help alleviate UI bottling.*

---

## 14. Contributing Developer Guidelines <a name="contributing-developer-guidelines"></a>

We welcome contributions to BowlForm AI. To keep the project structured:
1. **Fork the repository** on GitHub.
2. **Create a fresh branch** for your feature (`git checkout -b feature/AmazingFeature`).
3. **Follow PEP-8** standards strictly for backend Python code.
4. **Test properly**: Pass basic test clips before committing.
5. **Commit your changes**: (`git commit -m 'Add some AmazingFeature'`).
6. **Push to the branch**: (`git push origin feature/AmazingFeature`).
7. **Open a GitHub Pull Request**.

---

## 15. License <a name="license"></a>

This project is licensed under the MIT License - see the `LICENSE` file for details. You are free to use, modify, and distribute this software, but please include the original copyright notice.

---

## 16. Acknowledgments <a name="acknowledgments"></a>

- **MediaPipe Team (Google)** for providing an incredible CPU-optimized pose tracking neural network.
- **International Cricket Council (ICC)** for public biomechanical guidelines and testing specifications.
- **OpenRouter** for easy integration with state-of-the-art Large Language Models.
- Biomechanical standards inspired by independent research and the *University of Western Australia* biomechanics labs.

*BowlForm AI - Built for Bowlers, Engineered for Precision.*

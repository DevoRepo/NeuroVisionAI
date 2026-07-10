# NeuroVision AI

NeuroVision AI is a production-ready, full-stack medical application for Explainable AI Decision Support in Brain MRI Analysis. It leverages an EfficientNet PyTorch model, custom Image Quality Assessment modules, and Grad-CAM visualizations.

This repository contains two main parts:
1. **Backend**: A FastAPI application running PyTorch inference.
2. **Frontend**: A React/Vite web application using Tailwind CSS v4.

---

## 🚀 Setup Instructions

You need to run both the Backend and Frontend servers simultaneously. Please open **two separate terminal windows**.

### 1️⃣ Backend Setup (FastAPI + PyTorch)

**Prerequisites:** Python 3.9+ installed.

Open your first terminal and navigate to the `backend` directory:
```bash
cd neurovision_ai/backend
```

#### Step A: Create and Activate Virtual Environment

**For Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**For Windows (Command Prompt / PowerShell):**
```bash
python -m venv venv
venv\Scripts\activate
```

*(Note: You should see `(venv)` at the beginning of your terminal prompt indicating the virtual environment is active).*

#### Step B: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step C: Start the Backend Server

**For Mac/Linux:**
```bash
export PYTHONPATH=.
uvicorn app.main:app --reload --port 8000
```

**For Windows:**
```bash
set PYTHONPATH=.
uvicorn app.main:app --reload --port 8000
```
*The backend is now running at `http://localhost:8000/docs` (Swagger UI for testing).*

---

### 2️⃣ Frontend Setup (React + Vite)

**Prerequisites:** Node.js (v18+) and npm installed.

Open your **second terminal window** and navigate to the `frontend` directory:
```bash
cd neurovision_ai/frontend
```

#### Step A: Install Dependencies (Mac & Windows)
```bash
npm install
```

#### Step B: Start the Frontend Server (Mac & Windows)
```bash
npm run dev
```
*The frontend is now running! Look at your terminal for the Local URL (usually `http://localhost:5173/`).*

---

## 🎯 How to Use the Application

1. Open a web browser and go to the frontend URL (`http://localhost:5173/`).
2. Click **WELCOME** on the landing page.
3. On the Upload Page, enter any dummy Patient ID (e.g., `PID-983472`).
4. Upload a T1c or T2 Brain MRI slice (JPG or PNG) and submit.
5. The Processing screen will guide you through the AI analysis stages.
6. Once completed, explore the **Results Dashboard** for AI diagnostics, Image Quality scores, and Grad-CAM heatmaps.
7. Click **Export PDF** to download the professional medical report!

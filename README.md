<div align="center">
  <img src="https://img.icons8.com/nolan/96/artificial-intelligence.png" alt="AI Core" />
  
  # PP-Health-Tracker
  **Cinematic Predictive Clinical Intelligence**
  
  [![Vercel](https://img.shields.io/badge/Deployed_on-Vercel-black?logo=vercel)](https://pp-health-tracker.vercel.app/)
  [![Neon](https://img.shields.io/badge/Database-Neon_PostgreSQL-00e599?logo=postgresql)](https://neon.tech/)
  [![Python](https://img.shields.io/badge/Backend-Flask_Python-3776AB?logo=python&logoColor=white)]()
  [![TailwindCSS](https://img.shields.io/badge/UI-Tailwind_CSS-38B2AC?logo=tailwind-css&logoColor=white)]()

</div>

## 🌐 Overview
**PP-Health-Tracker** is an enterprise-grade, AI-powered metabolic and systemic health tracking platform. Built to feel like a high-end cinematic SaaS product, it leverages a multi-stage **LSTM Neural Network** to analyze 24 different systemic biomarkers and forecast wellness trajectories, specifically targeting the detection and progression of **Chronic Fatigue Syndrome (CFS)**.

---

## ✨ Core Features

*   **🧠 Neural Forecast Engine**: Integrates a pre-trained LSTM (Long Short-Term Memory) deep learning model to provide a forensic wellness projection based on temporal health markers.
*   **🎨 Cinematic SaaS UI/UX**: A fully responsive, dark-mode-first aesthetic ("Deep Carbon" & "Neural Green") featuring glassmorphism, GSAP scroll-driven animations, Lenis smooth scrolling, and an interactive canvas particle grid.
*   **🛡️ Self-Healing Data Pipeline**: Built-in algorithmic database synchronization that intercepts and auto-resolves PostgreSQL sequence desyncs (e.g., `predictions_pkey` violations) in real-time.
*   **🔐 Seamless Authentication**: Enterprise-grade Google OAuth 2.0 integration with automatic session persistence and strict HTTPS protocol enforcement.
*   **📄 Clinical Dossier Export**: One-click generation of beautifully formatted PDF health audits detailing probabilistic distributions and risk severities.

---

## 📸 Interface Showcases

*(Replace the paths below with actual screenshot URLs once uploaded to your repository's `/assets` folder)*

| 🎛️ Clinical Dashboard | 🔬 Neural Audit Pipeline |
| :---: | :---: |
| <img src="https://placehold.co/600x400/020202/22c55e?text=Clinical+Dashboard+Screenshot" alt="Dashboard" width="100%"/> | <img src="https://placehold.co/600x400/020202/22c55e?text=Multi-Step+Prediction+Form" alt="Audit Form" width="100%"/> |
| *Real-time analytics, trend variances, and risk severity indexes.* | *Interactive, multi-step biometric ingestion flow.* |

| 🚀 Landing Page | 📊 Prediction Results |
| :---: | :---: |
| <img src="https://placehold.co/600x400/020202/22c55e?text=Cinematic+Landing+Page" alt="Landing Page" width="100%"/> | <img src="https://placehold.co/600x400/020202/22c55e?text=Inference+Results" alt="Results" width="100%"/> |
| *Scroll-driven story telling with temporal intelligence insights.* | *Instant diagnostic classification and ML confidence metrics.* |

---

## 🏗️ Technical Architecture

### Backend Stack
*   **Framework**: Python Flask
*   **Database**: Neon PostgreSQL (Serverless)
*   **Driver**: `psycopg2-binary`
*   **Authentication**: `Authlib` (Google OAuth)
*   **PDF Generation**: `fpdf`

### Machine Learning
*   **Engine**: TensorFlow / Keras (`lstm_multiclass_model.h5`)
*   **Pre-processing**: Scikit-learn (`scaler.pkl`, `label_encoders.pkl`)
*   **Data Handling**: NumPy, Joblib

### Frontend Stack
*   **Styling**: Tailwind CSS (Vanilla via CDN)
*   **Typography**: Google Fonts (Outfit & Inter)
*   **Animations**: GSAP (ScrollTrigger) & HTML5 Canvas
*   **Smooth Scroll**: Lenis
*   **Icons**: Lucide Icons

---

## 🚀 Local Development Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Whitedevil2004r27/PP-Health-Tracker.git
cd PP-Health-Tracker
```

### 2. Set up a Virtual Environment
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Variables (`.env`)
Create a `.env` file in the root directory and populate it with your credentials:
```env
# Flask Core
SECRET_KEY=your_super_secret_flask_key

# Database
DATABASE_URL=postgresql://user:password@endpoint.neon.tech/neondb?sslmode=require

# Google OAuth
GOOGLE_CLIENT_ID=your_google_client_id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your_google_client_secret

# Enforce HTTPS locally if needed
PREFERRED_URL_SCHEME=http 
```

### 5. Run the Application
```bash
python app.py
```
The portal will be accessible at `http://127.0.0.1:5000`.

---

## ☁️ Deployment Guide (Vercel)

This application is heavily optimized for serverless deployment on **Vercel**.

1. **Vercel Project Setup**: Import the GitHub repository into your Vercel dashboard.
2. **Environment Variables**: Add all variables from your `.env` file into the Vercel project settings.
3. **Important Note on `vercel.json`**: The project includes a `vercel.json` that redirects all routes to `app.py`.
4. **Google Console Configuration**: 
   Ensure your Google Cloud Console Authorized Redirect URIs match your Vercel deployment:
   *   `https://<your-vercel-domain>.vercel.app/google/auth`

---

## 🛠️ Database Schema

The system automatically initializes and self-heals missing columns. The core schema includes:
*   `users`: ID, username, password hash, google_id, email, profile_pic, notifications.
*   `predictions`: ID, user_id, timestamp, prediction (Result), inputs_json (Raw biometrics).

---

<div align="center">
  <p>Engineered with precision for Clinical Intelligence.</p>
  <p>&copy; 2026 AI Health Intel Group</p>
</div>

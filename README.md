# AI-Powered Predictive Health Tracker | Clinical Intelligence 🚀🧠

An advanced, high-fidelity healthcare diagnostics platform utilizing **LSTM (Long Short-Term Memory)** neural networks for systemic wellness modeling and forensic risk assessment.

![Health Tracker Banner](https://images.unsplash.com/photo-1576091160550-2173dad99901?auto=format&fit=crop&q=80&w=1200)

## 💎 Liquid Glass Architecture
The platform features a signature **"Liquid Glass"** heritage aesthetic, characterized by high-index backdrop blurs, neon emerald accents, and a unified responsive navigation system.

- **Desktop**: Floating Top-Navigation Bar for synchronized workspace auditing.
- **Mobile**: Thumb-friendly Bottom-Navigation with a centralized "Analyze" action.

## ✨ V3.0 Clinical Features

- **🧠 Multi-Stage LSTM Engine**: Advanced predictive modeling for systemic health markers (Resting HR, HRV, BP Intensity).
- **🛡️ Stabilization & Security**: 
  - **Google OAuth 2.0**: Hardened authentication with session-state verification.
  - **Proxy Protocol**: Native **ProxyFix** integration for seamless deployment on Render.com (HTTPS).
- **📂 Forensic Documentation**: 
  - **Neural Specifications**: Detailed logic for the LSTM diagnostic sequences.
  - **Clinical Center**: Integrated Help Center, Privacy Policy, and Terms of Service.
- **⚙️ Integrated Identity Module**: Comprehensive profile management, UI density preferences, and persistent notification settings.
- **📊 Neural Forecasts**: Real-time visualization of systemic wellness progression over time.

## 🛠️ Technology Stack

- **Intelligence**: TensorFlow / Keras (LSTM models), Scikit-Learn (Clinical Encoders).
- **Backend**: Flask (Python 3.12+).
- **Database**: Self-healing SQLite3 with automated schema migration.
- **Security**: Authlib (OAuth), Werkzeug (ProxyFix).
- **Frontend**: TailwindCSS (Custom Logic), Lucide Icons, Glassmorphism CSS.

## 🚀 Deployment & Installation

### Prerequisites
- Python 3.8+ (Verified on Python 3.12/3.13)
- Google Cloud Console Credentials (for OAuth authorization)

### Local Configuration
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Whitedevil2004r27/PP-Health-Tracker.git
   cd PP-Health-Tracker
   ```

2. **Environment Setup**:
   Create a `.env` file in the root:
   ```env
   SECRET_KEY=your_clinical_key
   GOOGLE_CLIENT_ID=your_id.apps.googleusercontent.com
   GOOGLE_CLIENT_SECRET=your_client_secret
   ```

3. **Install Core Intelligence**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize System**:
   ```bash
   python app.py
   ```
   *The diagnostics portal will be accessible at `http://127.0.0.1:3000`.*

### Production Deployment (Netlify)
The application is pre-configured with `netlify.toml` for deployment.
- Netlify Functions require a WSGI adapter like `serverless-wsgi` to run Flask applications.
- Ensure your environment variables (like `SECRET_KEY` and Google OAuth credentials) are set in the Netlify dashboard.
- Note: SQLite databases do not persist across requests in serverless environments like Netlify. Consider migrating to a managed database like PostgreSQL for production.

## 📜 Neural Diagnostic Protocol
The system uses a multi-stage data synthesis protocol:
1. **Autonomic Markers**: Synthesis of resting vitals.
2. **Systemic Fatigue**: Modeling sleep flux and cognitive depth.
3. **Forensic Audit**: Generating clinical PDF reports for patient history.

## 📄 License
Distributed under the MIT License. Developed for Semester 8 - Final Project.

---
*Global Clinical Standard | AI Health Intel Group*

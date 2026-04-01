# PP Health Tracker 🚀

An advanced, AI-powered predictive health monitoring system designed to track systemic health metrics and provide diagnostic insights through Neural Network models.

![Health Tracker Banner](https://images.unsplash.com/photo-1576091160550-2173dad99901?auto=format&fit=crop&q=80&w=1200)

## ✨ Features

- **🛡️ Secure Authentication**: Integrated Google OAuth 2.0 and local password-based authentication.
- **🧠 AI Diagnostics**: Neural Network-powered health prediction models with "Safe Mode" fallback.
- **📊 Interactive Dashboard**: Real-time health metrics visualization and historical data tracking.
- **📄 PDF Audit Reports**: Generate professional systemic health audit reports with detailed data summaries.
- **☁️ Platform Independent**: Cross-platform support with automated database self-healing.
- **🎨 Premium UI**: Modern, dark-themed responsive design with glassmorphism and Lucide icons.

## 🛠️ Tech Stack

- **Backend**: Flask (Python)
- **Database**: SQLite3
- **Authentication**: Authlib (Google OAuth)
- **Report Generation**: FPDF
- **Frontend**: TailwindCSS, Lucide Icons, Vanilla JS

## 🚀 Getting Started

### Prerequisites

- Python 3.8+ (Compatible with Python 3.12, 3.13, and 3.14)
- Google Cloud Console Credentials (for OAuth)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/PP-Health-Tracker.git
   cd PP-Health-Tracker
   ```

2. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```env
   SECRET_KEY=your_secret_key
   GOOGLE_CLIENT_ID=your_google_id
   GOOGLE_CLIENT_SECRET=your_google_secret
   ```

3. **Install dependencies**:
   ```bash
   pip install flask authlib python-dotenv requests fpdf
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```
   The app will be available at `http://127.0.0.1:3000`.

## 📜 Database Schema

The system uses a self-healing SQLite database (`users.db`) that automatically migrates missing columns:
- `users`: ID, Username, Email, Google_ID, Profile_Pic
- `predictions`: ID, User_ID, Timestamp, Prediction, Input_Data

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---
*Developed for Semester 8 - Final Project*

import os
# Environment variable handling (Integrated for reliability)
def load_dotenv(path=None, override=False): 
    target_path = path if path else '.env'
    if os.path.exists(target_path):
        with open(target_path) as f:
            for line in f:
                if '=' in line:
                    parts = line.strip().split('=', 1)
                    if len(parts) == 2:
                        k, v = parts
                        k = k.strip()
                        v = v.strip().strip("'").strip('"')
                        os.environ[k] = v

# Initialize environment variables
basedir = os.path.abspath(os.path.dirname(__file__))
env_path = os.path.join(basedir, '.env')
load_dotenv(env_path, override=True)

# Diagnostic Check
google_id = os.environ.get('GOOGLE_CLIENT_ID')
if not google_id:
    print(f"DEBUG ERROR: GOOGLE_CLIENT_ID not found. Searched in: {env_path}")
else:
    print(f"DEBUG SUCCESS: GOOGLE_CLIENT_ID loaded (Length: {len(google_id)})")

# For local development with HTTP
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

from flask import Flask, render_template, request, redirect, url_for, session, flash, make_response, jsonify
import sqlite3
import datetime
import io
import json
import csv
from werkzeug.security import generate_password_hash, check_password_hash
# Authentication related (Safe import wrapper)
try:
    from authlib.integrations.flask_client import OAuth
    AUTH_SUPPORTED = True
except ImportError:
    AUTH_SUPPORTED = False
    class OAuth:
        def __init__(self, app=None): pass
        def register(self, **kwargs): return type('obj', (object,), {'authorize_redirect': lambda self, x: "Redirecting...", 'authorize_access_token': lambda self: {}})()

# Machine Learning related (Safe import wrapper)
try:
    import numpy as np
    import joblib
    # Using a fake load_model if tf-keras/tensorflow is missing
    try:
        from tf_keras.models import load_model
    except ImportError:
        from tensorflow.keras.models import load_model
    ML_SUPPORTED = True
except ImportError:
    ML_SUPPORTED = False

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'supersecretkey')

# OAuth Configuration
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=os.environ.get('GOOGLE_CLIENT_ID'),
    client_secret=os.environ.get('GOOGLE_CLIENT_SECRET'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile'
    }
)

# -------------------------
# Database Setup
# -------------------------
# Using absolute path for database to avoid issues with different CWD
DB_NAME = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'users.db')

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Base tables structure
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT
        )
    ''')
    
    # Self-healing Schema: Add missing columns without forbidden UNIQUE in ALTER
    columns_to_add = [
        ("google_id", "TEXT"),
        ("email", "TEXT"),
        ("profile_pic", "TEXT")
    ]
    
    for col_name, col_type in columns_to_add:
        try:
            cursor.execute(f"SELECT {col_name} FROM users LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute(f"ALTER TABLE users ADD COLUMN {col_name} {col_type}")
            # SQLite does not support UNIQUE in ALTER, so we add a unique index separately
            if col_name == "google_id":
                try:
                    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_google_id ON users(google_id)")
                except sqlite3.OperationalError: pass
            print(f"DEBUG: Rectified missing column '{col_name}'")

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            timestamp DATETIME,
            prediction TEXT,
            inputs_json TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# -------------------------
# Load Models & Encoders (Real or Mocked)
# -------------------------
if ML_SUPPORTED:
    try:
        model = load_model("lstm_multiclass_model.h5", compile=False)
        scaler = joblib.load("scaler.pkl")
        label_encoders = joblib.load("label_encoders.pkl")
        target_encoder = joblib.load("target_encoder.pkl")
    except Exception as e:
        print(f"ML ERROR: {str(e)}")
        ML_SUPPORTED = False

if not ML_SUPPORTED:
    # Use mocks for non-ML systems
    class MockEncoder:
        def __init__(self):
            # Using a custom class to mimic NumPy array's .tolist() behavior
            class MockClasses(list):
                def tolist(self): return self
            self.classes_ = MockClasses(["Normal", "Chances of CFD", "Chronic Fatigue Syndrome"])
        def transform(self, x): return [0]
        def inverse_transform(self, x): return ["Normal (Safe Mode)"]
    
    label_encoders = {col: MockEncoder() for col in ["gender", "work_status", "social_activity_level", "exercise_frequency", "meditation_or_mindfulness"]}
    target_encoder = MockEncoder()

numerical_cols = ["age", "sleep_quality_index", "brain_fog_level",
                  "physical_pain_score", "stress_level", "depression_phq9_score",
                  "fatigue_severity_scale_score", "pem_duration_hours",
                  "hours_of_sleep_per_night", "pem_present"]

categorical_cols = ["gender", "work_status", "social_activity_level",
                    "exercise_frequency", "meditation_or_mindfulness"]

columns = numerical_cols + categorical_cols

# -------------------------
# Routes
# -------------------------
@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('predict'))
    return render_template('index.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        hashed_password = generate_password_hash(password)
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
            conn.commit()
            flash("Registration Successful! Please login.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username already exists.", "danger")
        finally:
            conn.close()
    return render_template('register.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=?", (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            session['username'] = username
            session['user_id'] = user[0]
            session['profile_pic'] = user[5] # Ensure profile pic is loaded on normal login
            flash(f"Welcome {username}!", "success")
            return redirect(url_for('predict'))
        else:
            flash("Invalid credentials.", "danger")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('user_id', None)
    session.pop('profile_pic', None)
    flash("Logged out successfully.", "info")
    return redirect(url_for('login'))

@app.route('/login/google')
def google_login():
    # Dynamically generate the redirect URI based on the current domain
    redirect_uri = url_for('google_auth', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/google/auth')
def google_auth():
    token = google.authorize_access_token()
    user_info = token.get('userinfo')
    if not user_info:
        flash("Could not retrieve user info from Google.", "danger")
        return redirect(url_for('login'))
    
    email = user_info.get('email')
    google_id = user_info.get('sub')
    name = user_info.get('name', email.split('@')[0])
    picture = user_info.get('picture')
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE google_id=?", (google_id,))
    user = cursor.fetchone()
    
    if not user:
        # Check if username is already taken
        cursor.execute("SELECT id FROM users WHERE username=?", (name,))
        if cursor.fetchone():
            name = f"{name}_{google_id[:5]}"
            
        cursor.execute("INSERT INTO users (username, google_id, email, profile_pic, password) VALUES (?, ?, ?, ?, ?)", 
                       (name, google_id, email, picture, "GOOGLE_AUTH_USER"))
        conn.commit()
        cursor.execute("SELECT * FROM users WHERE google_id=?", (google_id,))
        user = cursor.fetchone()
    else:
        # Update profile picture if it changed
        cursor.execute("UPDATE users SET profile_pic=? WHERE google_id=?", (picture, google_id))
        conn.commit()
        # Refresh user data
        cursor.execute("SELECT * FROM users WHERE google_id=?", (google_id,))
        user = cursor.fetchone()
        
    conn.close()
    
    session['username'] = user[1]
    session['user_id'] = user[0]
    session['profile_pic'] = user[5] # Index 5 is profile_pic
    flash(f"Welcome {user[1]} (via Google)!", "success")
    return redirect(url_for('predict'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'username' not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for('login'))

    # -------------------------
    # Initialize variables
    # -------------------------
    prediction = None
    probabilities = []
    class_labels = target_encoder.classes_.tolist()
    prob_img = None  # Ensure variable always exists
    error_msg = None

    if request.method == 'POST':
        try:
            input_vals = []

            # -------------------------
            # Collect numerical inputs
            # -------------------------
            for col in numerical_cols:
                val = request.form.get(col)
                if val is None or val.strip() == '':
                    error_msg = f"Please enter value for '{col}'"
                    break
                try:
                    input_vals.append(float(val))
                except ValueError:
                    error_msg = f"Invalid numeric value for '{col}'"
                    break

            # -------------------------
            # Collect categorical inputs
            # -------------------------
            for col in categorical_cols:
                val = request.form.get(col)
                if val is None or val.strip() == '':
                    error_msg = f"Please select value for '{col}'"
                    break
                val = val.strip()
                le = label_encoders[col]

                if val not in le.classes_:
                    error_msg = f"Invalid value '{val}' for '{col}'"
                    break
                input_vals.append(le.transform([val])[0])

            # -------------------------
            # Prediction (Real or Mocked)
            # -------------------------
            if error_msg is None:
                if ML_SUPPORTED:
                    input_arr = np.array(input_vals).reshape(1, -1)
                    input_scaled = scaler.transform(input_arr)
                    input_lstm = input_scaled.reshape((1, input_scaled.shape[1], 1))

                    pred_probs = model.predict(input_lstm, verbose=0)[0]  # 1D array
                    pred_class = np.argmax(pred_probs)
                    prediction = target_encoder.inverse_transform([pred_class])[0]
                    probabilities = pred_probs.tolist()
                else:
                    prediction = "Normal (Demo Mode - Auth Active)"
                    probabilities = [0.1, 0.1, 0.8]

                # -------------------------
                # Log Prediction
                # -------------------------

                # -------------------------
                # Log Prediction
                # -------------------------
                # Calculate risk score: (p_normal * 0 + p_chances * 50 + p_cfs * 100)
                # Labels: ['Chances of CFD', 'Chronic Fatigue Syndrome', 'Normal']
                # Indices: 0: Chances, 1: CFS, 2: Normal
                p_chances = probabilities[0]
                p_cfs = probabilities[1]
                p_normal = probabilities[2]
                risk_score = int((p_chances * 50 + p_cfs * 100)) # Simple mapping
                
                conn = sqlite3.connect(DB_NAME)
                cursor = conn.cursor()
                # Save input data as JSON string
                inputs_data = {col: request.form.get(col) for col in columns}
                inputs_json = json.dumps(inputs_data)
                
                cursor.execute("INSERT INTO predictions (user_id, timestamp, prediction, inputs_json) VALUES (?, ?, ?, ?)", 
                               (session['user_id'], datetime.datetime.now(), prediction, inputs_json))
                conn.commit()
                conn.close()

                # Store latest prediction data in session for PDF generation
                session['latest_report'] = {
                    'prediction': prediction,
                    'risk_score': risk_score,
                    'probabilities': probabilities,
                    'class_labels': class_labels,
                    'inputs': {col: request.form.get(col) for col in columns},
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

        except Exception as e:
            error_msg = str(e)
            prob_img = None  # Ensure variable exists on error

    # -------------------------
    # Render template
    # -------------------------
    # Calculate risk_score for the template if prediction was just made
    risk_score = session.get('latest_report', {}).get('risk_score') if prediction else None

    return render_template('predict.html',
                           columns=columns,
                           categorical_cols=categorical_cols,
                           label_encoders=label_encoders,
                           prediction=prediction,
                           probabilities=probabilities,
                           prob_img=prob_img,
                           class_labels=class_labels,
                           risk_score=risk_score,
                           error_msg=error_msg)

@app.route('/history')
def history():
    if 'username' not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for('login'))

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, prediction, inputs_json FROM predictions WHERE user_id=? ORDER BY timestamp DESC", (session['user_id'],))
    user_predictions = cursor.fetchall()
    conn.close()
    
    # Process inputs_json for the template
    processed_history = []
    for row in user_predictions:
        processed_history.append({
            'timestamp': row[0],
            'prediction': row[1],
            'inputs': json.loads(row[2]) if row[2] else {}
        })

    return render_template('history.html', predictions=processed_history)

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for('login'))

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, prediction FROM predictions WHERE user_id=? ORDER BY timestamp ASC", (session['user_id'],))
    user_predictions = cursor.fetchall()
    conn.close()

    total_predictions = len(user_predictions)
    # Count distributions
    counts = {"Normal": 0, "Chances of CFD": 0, "Chronic Fatigue Syndrome": 0}
    for row in user_predictions:
        counts[row[1]] = counts.get(row[1], 0) + 1
    
    # Trend for last 7 days (simplified as last 7 records for visual)
    label_map = {'Normal': 0, 'Chances of CFD': 1, 'Chronic Fatigue Syndrome': 2}
    trend_data = [label_map.get(row[1], 0) for row in user_predictions[-7:]]
    trend_labels = [row[0] for row in user_predictions[-7:]]

    return render_template('dashboard.html', 
                           total=total_predictions,
                           counts=counts,
                           trend_labels=trend_labels,
                           trend_data=trend_data,
                           recent_predictions=user_predictions[::-1][:5])

@app.route('/export_csv')
def export_csv():
    # Security: Protected route check
    if 'username' not in session:
        flash("Authorization required to export data.", "warning")
        return redirect(url_for('login'))

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, prediction, inputs_json FROM predictions WHERE user_id=?", (session['user_id'],))
    rows = cursor.fetchall()
    conn.close()

    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['Timestamp', 'Prediction'] + columns)
    
    for row in rows:
        inputs = json.loads(row[2]) if row[2] else {}
        input_row = [inputs.get(col, '') for col in columns]
        cw.writerow([row[0], row[1]] + input_row)

    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = f"attachment; filename=CFS_Health_History_{datetime.date.today()}.csv"
    output.headers["Content-type"] = "text/csv"
    return output

@app.route('/download_report')
def download_report():
    # Security: Protected route check
    if 'username' not in session:
        flash("Authorization required to access reports.", "warning")
        return redirect(url_for('login'))

    if 'latest_report' not in session:
        flash("No report data found. Please perform a prediction first.", "warning")
        return redirect(url_for('predict'))

    data = session['latest_report']
    
    class PDFReport(FPDF):
        def header(self):
            self.set_fill_color(0, 0, 0) # Black
            self.rect(0, 0, 210, 45, 'F')
            self.set_font('Arial', 'B', 22)
            self.set_text_color(255, 255, 255)
            self.cell(0, 25, 'PP HEALTH TRACKER', ln=True, align='C')
            self.set_font('Arial', 'B', 10)
            self.set_text_color(34, 197, 94) # Primary Green
            self.cell(0, 5, 'NEURAL NETWORK DIAGNOSTIC AUDIT', ln=True, align='C')
            self.ln(10)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(100, 100, 100)
            self.cell(0, 10, f'Page {self.page_no()} | Systemic Health Audit | PP Health Tracker', align='C')

    pdf = PDFReport()
    pdf.add_page()
    pdf.ln(15)

    # Info Section
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(100, 10, txt=f"Patient: {session['username']}")
    pdf.cell(90, 10, txt=f"Date: {data['timestamp']}", align='R', ln=True)
    pdf.line(10, 62, 200, 62)
    pdf.ln(10)

    # Main Result
    pdf.set_fill_color(245, 255, 245) # Light Green background
    pdf.rect(10, 75, 190, 35, 'F')
    pdf.set_y(80)
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(190, 10, txt=f"DIAGNOSIS: {data['prediction']}", align='C', ln=True)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(34, 197, 94)
    pdf.cell(190, 10, txt=f"CLINICAL RISK SCORE: {data['risk_score']}/100", align='C', ln=True)
    pdf.ln(15)

    # Probabilities
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(190, 10, txt="PROBABILISTIC DISTRIBUTION:", ln=True)
    pdf.set_font("Arial", '', 11)
    for label, prob in zip(data['class_labels'], data['probabilities']):
        pdf.cell(190, 8, txt=f"  - {label}: {prob:.2%}", ln=True)
    pdf.ln(10)

    # Input Audit
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(190, 10, txt="Input Data Audit:", ln=True)
    pdf.set_font("Arial", '', 9)
    sorted_inputs = sorted(data['inputs'].items())
    
    # Simple table for inputs
    pdf.set_fill_color(245, 245, 245)
    for i, (key, val) in enumerate(sorted_inputs):
        display_key = key.replace('_', ' ').title()
        fill = (i % 2 == 0)
        pdf.cell(95, 7, txt=f"  {display_key}:", border=1, fill=fill)
        pdf.cell(95, 7, txt=f"  {val}", border=1, ln=True, fill=fill)

    # Tips Section
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(190, 10, txt="Clinical Recommendations:", ln=True)
    pdf.set_font("Arial", '', 10)
    if 'Chronic' in data['prediction']:
        tips = ["- Immediate consultation with a fatigue specialist is recommended.",
                "- Implement 'Pacing' strategies to avoid Post-Exertional Malaise (PEM).",
                "- Focus on restorative rest and nutrient-dense anti-inflammatory diet."]
    elif 'Chances' in data['prediction']:
        tips = ["- Monitor your symptoms daily and reduce high-impact stress.",
                "- Improve sleep hygiene (7-9 hours of dark-room rest).",
                "- Introduce gentle light movement but avoid over-exertion."]
    else:
        tips = ["- Maintain a balanced lifestyle with regular hydration.",
                "- Regular physical check-ups to monitor baseline metrics.",
                "- Practice mindfulness to manage daily cognitive loads."]
    
    for tip in tips:
        pdf.multi_cell(190, 8, txt=tip)

    output = pdf.output(dest='S')
    response = make_response(output)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename=Predictive_Health_Report_{datetime.datetime.now().strftime("%Y%m%d")}.pdf'
    return response

if __name__ == '__main__':
    # Use environment variables for port and host (required for production)
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port, debug=False)

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
if not os.environ.get('VERCEL') and not os.environ.get('RENDER'):
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

from flask import Flask, render_template, request, redirect, url_for, session, flash, make_response, jsonify
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman
import psycopg2
import psycopg2.extras
from psycopg2 import IntegrityError
import datetime
import io
import json
import csv
import urllib.parse
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.middleware.proxy_fix import ProxyFix
from export_utils import generate_clinical_pdf, generate_historical_csv
# Authentication related (Safe import wrapper)
try:
    from authlib.integrations.flask_client import OAuth
    AUTH_SUPPORTED = True
except ImportError as e:
    print(f"AUTH ERROR: Authlib not found: {e}")
    AUTH_SUPPORTED = False
    class OAuth:
        def __init__(self, app=None): pass
        def register(self, **kwargs): return type('obj', (object,), {'authorize_redirect': lambda self, x: redirect(x), 'authorize_access_token': lambda self: {}})()

try:
    from fpdf import FPDF
    PDF_SUPPORTED = True
except ImportError as e:
    print(f"PDF ERROR: FPDF not found: {e}")
    PDF_SUPPORTED = False

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

# Automatic Proxy Support for Vercel/Production
if os.environ.get('VERCEL') or os.environ.get('PROXY_FIX'):
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1, x_port=1, x_prefix=1)
    app.config['SESSION_COOKIE_SECURE'] = True
    app.config['PREFERRED_URL_SCHEME'] = 'https'
    # Force OAuth to use HTTPS
    os.environ['OAUTHLIB_RELAX_TOKEN_SCOPE'] = '1'

app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_NAME'] = 'pp_health_session'
csrf = CSRFProtect(app)
limiter = Limiter(get_remote_address, app=app, default_limits=["200 per day", "50 per hour"], storage_uri="memory://")
# Talisman for security headers - disable force_https since Vercel handles SSL termination
Talisman(app, content_security_policy=None, force_https=False)

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
# Predictive Intelligence Utilities
# -------------------------
def calculate_health_forecast(timestamps, scores, days_to_forecast=7):
    # Mapping timestamps to days from start for regression
    if len(scores) < 3: return []
    
    try:
        def parse_dt(dt_input):
            if isinstance(dt_input, datetime.datetime):
                return dt_input
            formats = ["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]
            for fmt in formats:
                try:
                    return datetime.datetime.strptime(dt_input, fmt)
                except:
                    continue
            return datetime.datetime.now()

        # Convert timestamps to numeric (seconds since epoch)
        numeric_times = []
        for t in timestamps:
            dt = parse_dt(t)
            numeric_times.append(dt.timestamp())

        # Normalize times
        start_time = numeric_times[0]
        x = np.array([(t - start_time) / 86400 for t in numeric_times]) # days
        y = np.array(scores)
        
        # Fit linear model (Degree 1)
        coeffs = np.polyfit(x, y, 1)
        polynomial = np.poly1d(coeffs)
        
        # Generate forecast points
        last_day = x[-1]
        forecast_x = np.array([last_day + d for d in range(1, days_to_forecast + 1)])
        forecast_y = polynomial(forecast_x)
        
        # Clip scores to [0, 100]
        forecast_y = np.clip(forecast_y, 0, 100)
        
        # Generate labels (Next 7 Days)
        last_dt = parse_dt(timestamps[-1])
            
        forecast_labels = []
        for d in range(1, days_to_forecast + 1):
            next_dt = last_dt + datetime.timedelta(days=d)
            forecast_labels.append(next_dt.strftime("%b %d"))
            
        return list(zip(forecast_labels, forecast_y.tolist()))
    except Exception as e:
        print(f"Forecasting Error: {e}")
        return []

# -------------------------
# Database Setup
# -------------------------
DATABASE_URL = os.environ.get('DATABASE_URL')

def get_db_connection():
    if not DATABASE_URL:
        print("CRITICAL: DATABASE_URL not set.")
        return None
    return psycopg2.connect(DATABASE_URL)

def init_db():
    conn = get_db_connection()
    if not conn: return
    cursor = conn.cursor()
    
    # Base tables structure
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password TEXT
        )
    ''')
    conn.commit()
    
    # Self-healing Schema: Add missing columns
    columns_to_add = [
        ("google_id", "TEXT UNIQUE"),
        ("email", "TEXT"),
        ("profile_pic", "TEXT"),
        ("notifications", "INTEGER DEFAULT 1"),
        ("ui_preferences", "TEXT DEFAULT '{}'")
    ]
    
    for col_name, col_type in columns_to_add:
        try:
            cursor.execute(f"ALTER TABLE users ADD COLUMN {col_name} {col_type}")
            conn.commit()
            print(f"DEBUG: Rectified missing column '{col_name}'")
        except psycopg2.errors.DuplicateColumn:
            conn.rollback()
        except Exception as e:
            conn.rollback()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            user_id INTEGER,
            timestamp TIMESTAMP,
            prediction TEXT,
            inputs_json TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    conn.commit()
    conn.close()

# Only initialize DB if DATABASE_URL is provided
if DATABASE_URL:
    try:
        init_db()
    except Exception as e:
        print(f"CRITICAL DATABASE ERROR: {e}")
else:
    print("WARNING: Skipping init_db because DATABASE_URL is missing.")

# -------------------------
# Load Models & Encoders (Real or Mocked)
# -------------------------
if ML_SUPPORTED:
    try:
        model_path = os.path.join(basedir, "lstm_multiclass_model.h5")
        scaler_path = os.path.join(basedir, "scaler.pkl")
        encoders_path = os.path.join(basedir, "label_encoders.pkl")
        target_path = os.path.join(basedir, "target_encoder.pkl")
        
        model = load_model(model_path, compile=False)
        scaler = joblib.load(scaler_path)
        label_encoders = joblib.load(encoders_path)
        target_encoder = joblib.load(target_path)
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
@limiter.limit("5 per hour")
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        hashed_password = generate_password_hash(password)
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password))
            conn.commit()
            flash("Registration Successful! Please login.", "success")
            return redirect(url_for('login'))
        except IntegrityError:
            conn.rollback()
            flash("Username already exists.", "danger")
        finally:
            conn.close()
    return render_template('register.html')

@app.route('/login', methods=['GET','POST'])
@limiter.limit("10 per minute")
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
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
    # Force HTTPS for the redirect_uri in production
    scheme = 'https' if (os.environ.get('VERCEL') or os.environ.get('PROXY_FIX')) else 'http'
    redirect_uri = url_for('google_auth', _external=True, _scheme=scheme)
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
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE google_id=%s", (google_id,))
    user = cursor.fetchone()
    
    if not user:
        # Check if username is already taken
        cursor.execute("SELECT id FROM users WHERE username=%s", (name,))
        if cursor.fetchone():
            name = f"{name}_{google_id[:5]}"
            
        cursor.execute("INSERT INTO users (username, google_id, email, profile_pic, password) VALUES (%s, %s, %s, %s, %s)", 
                       (name, google_id, email, picture, "GOOGLE_AUTH_USER"))
        conn.commit()
        cursor.execute("SELECT * FROM users WHERE google_id=%s", (google_id,))
        user = cursor.fetchone()
    else:
        # Update profile picture if it changed
        cursor.execute("UPDATE users SET profile_pic=%s WHERE google_id=%s", (picture, google_id))
        conn.commit()
        # Refresh user data
        cursor.execute("SELECT * FROM users WHERE google_id=%s", (google_id,))
        user = cursor.fetchone()
        
    conn.close()
    
    session['username'] = user[1]
    session['user_id'] = user[0]
    session['profile_pic'] = user[5] # Index 5 is profile_pic
    flash(f"Welcome {user[1]} (via Google)!", "success")
    return redirect(url_for('predict'))

@app.route('/predict', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
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
                le = label_encoders[col]
                # Safety net: use first class as default if field is missing or empty
                if val is None or val.strip() == '':
                    val = le.classes_[0]
                val = val.strip()
                # Safety net: if the value is not in known classes, use first class
                if val not in le.classes_:
                    val = le.classes_[0]
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
                # Calculate risk score: (p_normal * 0 + p_chances * 50 + p_cfs * 100)
                # Labels: ['Chances of CFD', 'Chronic Fatigue Syndrome', 'Normal']
                # Indices: 0: Chances, 1: CFS, 2: Normal
                p_chances = probabilities[0]
                p_cfs = probabilities[1]
                p_normal = probabilities[2]
                risk_score = int((p_chances * 50 + p_cfs * 100)) # Simple mapping
                
                conn = get_db_connection()
                cursor = conn.cursor()
                # Save input data as JSON string
                inputs_data = {col: request.form.get(col) for col in columns}
                inputs_json = json.dumps(inputs_data)
                
                cursor.execute("INSERT INTO predictions (user_id, timestamp, prediction, inputs_json) VALUES (%s, %s, %s, %s)", 
                               (session['user_id'], datetime.datetime.now(), prediction, inputs_json))
                conn.commit()
                
                # Retrieve the newly created ID to confirm save
                cursor.execute("SELECT id FROM predictions WHERE user_id=%s ORDER BY id DESC LIMIT 1", (session['user_id'],))
                new_audit_id = cursor.fetchone()
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
                flash("Neural Audit synchronized successfully.", "success")

        except psycopg2.errors.UniqueViolation as e:
            # Self-healing: Reset sequence if out of sync
            print(f"DATABASE SEQUENCE DESYNC DETECTED: {e}")
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT setval('predictions_id_seq', COALESCE((SELECT MAX(id)+1 FROM predictions), 1), false);")
                # Retry insertion
                cursor.execute("INSERT INTO predictions (user_id, timestamp, prediction, inputs_json) VALUES (%s, %s, %s, %s)", 
                               (session['user_id'], datetime.datetime.now(), prediction, inputs_json))
                conn.commit()
                conn.close()
                error_msg = None
            except Exception as retry_e:
                error_msg = f"Database Critical Error: {str(retry_e)}"
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

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, prediction, inputs_json FROM predictions WHERE user_id=%s ORDER BY timestamp DESC", (session['user_id'],))
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

@app.route('/help-center')
def help_center():
    return render_template('help_center.html')

@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

@app.route('/privacy-policy')
def privacy_policy():
    return render_template('privacy_policy.html')

@app.route('/terms-of-service')
def terms_of_service():
    return render_template('terms_of_service.html')

@app.route('/settings')
def settings():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('settings.html')

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'username' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        # Logic to update profile would go here (flash success for mockup)
        flash("Patient clinical markers successfully synchronized.", "success")
        return redirect(url_for('profile'))
    return render_template('profile.html')

@app.route('/preferences', methods=['GET', 'POST'])
def preferences():
    if 'username' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        # Logic to update prefs would go here
        flash("User interface preferences persisted.", "success")
        return redirect(url_for('preferences'))
    return render_template('preferences.html')

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for('login'))

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, prediction, inputs_json FROM predictions WHERE user_id=%s ORDER BY timestamp ASC", (session['user_id'],))
    user_predictions = cursor.fetchall()
    conn.close()

    total_predictions = len(user_predictions)
    # Count distributions
    counts = {"Normal": 0, "Chances of CFD": 0, "Chronic Fatigue Syndrome": 0}
    for p in user_predictions:
        p_type = p[1]
        if p_type in counts:
            counts[p_type] += 1
    
    # Map for scores: Normal=100, CFD=70, CFS=40
    score_map = {"Normal": 100, "Chances of CFD": 70, "Chronic Fatigue Syndrome": 40}

    # Advanced: Wellness Score Calculation (0-100)
    # Weighted based on latest prediction and frequency of CFS states
    wellness_score = 100
    if total_predictions > 0:
        latest_pred = user_predictions[-1][1]
        penalty = {"Normal": 0, "Chances of CFD": 30, "Chronic Fatigue Syndrome": 60}
        wellness_score -= penalty.get(latest_pred, 0)
        # Trend penalty (if things are getting worse)
        if total_predictions >= 3:
            recent = [p[1] for p in user_predictions[-3:]]
            if recent.count("Normal") == 0: wellness_score -= 10

    # -------------------------
    # Neural Forecasting Logic
    # -------------------------
    forecast_data = []
    if total_predictions >= 3:
        # Map history to scores: Normal=100, CFD=70, CFS=40
        score_map = {"Normal": 100, "Chances of CFD": 70, "Chronic Fatigue Syndrome": 40}
        history_scores = [score_map.get(p[1], 50) for p in user_predictions]
        history_times = [p[0] for p in user_predictions]
        forecast_data = calculate_health_forecast(history_times, history_scores)

    try:
        # Prepare Chart Data
        chart_labels = []
        for p in user_predictions:
            dt_str = str(p[0])
            label = dt_str.split('.')[0] if '.' in dt_str else dt_str
            chart_labels.append(label)
        
        chart_data = [score_map.get(p[1], 50) for p in user_predictions] if total_predictions > 0 else []
        forecast_labels = [str(f[0]) for f in forecast_data]
        forecast_points = [float(f[1]) for f in forecast_data]

        full_labels = chart_labels + forecast_labels
        actual_padded = [float(x) if x is not None else None for x in chart_data] + [None] * len(forecast_labels)
        forecast_padded = ([None] * (len(chart_data) - 1) + [float(chart_data[-1])] + forecast_points) if chart_data else []

        processed_data = {
            'labels': full_labels,
            'actual': actual_padded,
            'forecast': forecast_padded
        }

        last_actual = float(chart_data[-1]) if chart_data else 0
        last_forecast = float(forecast_points[-1]) if forecast_points else 0
        trend_variance = int(last_forecast - last_actual)
    except Exception as e:
        print(f"DASHBOARD LOGIC ERROR: {e}")
        processed_data = {'labels': [], 'actual': [], 'forecast': []}
        last_actual = 0
        last_forecast = 0
        trend_variance = 0

    return render_template('dashboard.html', 
                           total=total_predictions,
                           counts=counts,
                           wellness_score=int(max(0, min(100, wellness_score))),
                           chart_data_json=json.dumps(processed_data),
                           chart_data_dict=processed_data,
                           last_actual=int(last_actual),
                           last_forecast=int(last_forecast),
                           trend_variance=trend_variance,
                           recent_predictions=user_predictions[::-1][:5])

@app.route('/simulate', methods=['POST'])
def simulate():
    """Real-time simulation for the home page (No DB save)"""
    try:
        data = request.json
        # Convert to float for model
        input_data = [float(x) for x in data.values()]
        
        if ML_SUPPORTED:
            # Mock or Real Model prediction
            import numpy as np
            X = np.array([input_data])
            # Mocking probabilities for simulation feel
            probabilities = [0.8, 0.15, 0.05] if input_data[0] < 80 else [0.2, 0.4, 0.4]
            prediction = target_encoder.classes_.tolist()[np.argmax(probabilities)]
        else:
            prediction = "Normal"
            probabilities = [0.9, 0.05, 0.05]

        return jsonify({
            "status": "success",
            "prediction": prediction,
            "confidence": f"{max(probabilities)*100:.1f}%"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/download_report')
def download_report():
    if 'username' not in session or 'latest_report' not in session:
        flash("No recent neural audit found to export.", "error")
        return redirect(url_for('dashboard'))

    data = session['latest_report']
    
    # Calculate forecast string for PDF
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, prediction FROM predictions WHERE user_id = %s ORDER BY timestamp ASC", (session['user_id'],))
    user_predictions = cursor.fetchall()
    conn.close()
    
    forecast_text = "Baseline calibration required (3+ audits)."
    if len(user_predictions) >= 3:
        score_map = {"Normal": 100, "Chances of CFD": 70, "Chronic Fatigue Syndrome": 40}
        history_scores = [score_map.get(p[1], 50) for p in user_predictions]
        history_times = [p[0] for p in user_predictions]
        forecast_data = calculate_health_forecast(history_times, history_scores)
        if forecast_data:
            last_actual = history_scores[-1]
            last_forecast = forecast_data[-1][1]
            diff = last_forecast - last_actual
            trend = "improving" if diff > 0 else "declining"
            forecast_text = f"Neural analysis projects an {trend} trend. Projected 7-day efficiency index: {int(last_forecast)}/100."

    return generate_clinical_pdf(data, forecast_text, session['username'])

@app.route('/export_data')
def export_data():
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, prediction, inputs_json FROM predictions WHERE user_id = %s ORDER BY timestamp DESC", (session['user_id'],))
    predictions = cursor.fetchall()
    conn.close()
    
    return generate_historical_csv(predictions)

@app.route('/delete_account', methods=['POST'])
def delete_account():
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    conn = get_db_connection()
    cursor = conn.cursor()
    # Delete predictions first due to foreign key
    cursor.execute("DELETE FROM predictions WHERE user_id = %s", (session['user_id'],))
    # Delete user
    cursor.execute("DELETE FROM users WHERE id = %s", (session['user_id'],))
    conn.commit()
    conn.close()
    
    session.clear()
    flash("Your account and all associated neural data have been permanently deleted.", "success")
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Use environment variables for port and host (required for production)
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port, debug=True)

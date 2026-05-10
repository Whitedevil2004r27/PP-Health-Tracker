import sys
import os

# Ensure app can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app import app, get_db_connection
import uuid
from werkzeug.security import generate_password_hash

def run_tests():
    print("Starting Neon PostgreSQL Integration Verification...\n")
    
    conn = get_db_connection()
    if not conn:
        print("FAILED: Could not connect to Neon PostgreSQL. Check DATABASE_URL.")
        sys.exit(1)
        
    print("PASSED 1. Database Connection: SUCCESS")
    
    # Generate unique test user
    test_user = f"testuser_{uuid.uuid4().hex[:8]}"
    test_pass = "testpassword"
    hashed_pass = generate_password_hash(test_pass)
    
    cursor = conn.cursor()
    
    # 1. Test INSERT (Registration)
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s) RETURNING id", (test_user, hashed_pass))
        user_id = cursor.fetchone()[0]
        conn.commit()
        print("PASSED 2. CRUD - Create User (INSERT): SUCCESS")
    except Exception as e:
        print(f"FAILED to Insert User: {e}")
        sys.exit(1)
        
    # 2. Test SELECT (Login)
    try:
        cursor.execute("SELECT * FROM users WHERE username=%s", (test_user,))
        user = cursor.fetchone()
        if user and user[1] == test_user:
            print("PASSED 3. CRUD - Read User (SELECT): SUCCESS")
        else:
            print("FAILED: User not found after insert.")
            sys.exit(1)
    except Exception as e:
        print(f"FAILED to Select User: {e}")
        sys.exit(1)
        
    # 3. Test UPDATE (Profile Settings Update)
    try:
        cursor.execute("UPDATE users SET notifications=0 WHERE id=%s", (user_id,))
        conn.commit()
        
        cursor.execute("SELECT notifications FROM users WHERE id=%s", (user_id,))
        notif = cursor.fetchone()[0]
        if notif == 0:
            print("PASSED 4. CRUD - Update User Settings (UPDATE): SUCCESS")
        else:
            print("FAILED: Update did not persist.")
            sys.exit(1)
    except Exception as e:
        print(f"FAILED to Update User: {e}")
        sys.exit(1)
        
    # 4. Test Relationships & Foreign Keys (Predictions)
    try:
        from datetime import datetime
        cursor.execute("INSERT INTO predictions (user_id, timestamp, prediction, inputs_json) VALUES (%s, %s, %s, %s) RETURNING id",
                       (user_id, datetime.now(), "Normal", '{"test": true}'))
        pred_id = cursor.fetchone()[0]
        conn.commit()
        
        cursor.execute("SELECT * FROM predictions WHERE user_id=%s", (user_id,))
        preds = cursor.fetchall()
        if len(preds) == 1:
            print("PASSED 5. Relationships - Create Prediction with Foreign Key: SUCCESS")
        else:
            print("FAILED: Prediction not found.")
            sys.exit(1)
    except Exception as e:
        print(f"FAILED to Insert Prediction: {e}")
        sys.exit(1)

    # 5. Test DELETE (Cleanup)
    try:
        cursor.execute("DELETE FROM predictions WHERE user_id=%s", (user_id,))
        cursor.execute("DELETE FROM users WHERE id=%s", (user_id,))
        conn.commit()
        
        cursor.execute("SELECT * FROM users WHERE id=%s", (user_id,))
        if not cursor.fetchone():
            print("PASSED 6. CRUD - Delete and Cleanup (DELETE): SUCCESS")
        else:
            print("FAILED: Cleanup failed.")
            sys.exit(1)
    except Exception as e:
        print(f"FAILED to Delete User/Prediction: {e}")
        sys.exit(1)
        
    print("\nALL TESTS PASSED: Neon PostgreSQL is fully integrated and functional!")
    
if __name__ == "__main__":
    run_tests()

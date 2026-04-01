import sqlite3
import os

def update_db():
    # Using absolute path to ensure we target the same DB as app.py
    db_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'users.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN google_id TEXT UNIQUE")
        print("Added google_id column.")
    except sqlite3.OperationalError:
        print("google_id column already exists.")
    
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN email TEXT")
        print("Added email column.")
    except sqlite3.OperationalError:
        print("email column already exists.")

    try:
        cursor.execute("ALTER TABLE users ADD COLUMN profile_pic TEXT")
        print("Added profile_pic column.")
    except sqlite3.OperationalError:
        print("profile_pic column already exists.")
    
    # Make password nullable for OAuth users
    # In SQLite, we can't easily change NOT NULL to NULL without recreating the table,
    # but since we already updated the schema in app.py for NEW tables,
    # for existing users we'll just leave it as is for now or use a dummy hash if needed.
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    update_db()

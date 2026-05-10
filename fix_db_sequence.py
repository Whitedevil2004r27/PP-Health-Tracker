import os
import psycopg2

def load_dotenv():
    if os.path.exists('.env'):
        with open('.env') as f:
            for line in f:
                if '=' in line:
                    k, v = line.strip().split('=', 1)
                    os.environ[k.strip()] = v.strip().strip("'").strip('"')

load_dotenv()
database_url = os.environ.get('DATABASE_URL')

if not database_url:
    print("No DATABASE_URL found.")
    exit(1)

try:
    conn = psycopg2.connect(database_url)
    cursor = conn.cursor()
    
    # Check max ID
    cursor.execute("SELECT MAX(id) FROM predictions;")
    max_id = cursor.fetchone()[0]
    
    if max_id:
        print(f"Max ID in predictions table: {max_id}")
        # Reset sequence
        cursor.execute("SELECT setval('predictions_id_seq', %s);", (max_id,))
        conn.commit()
        print("Sequence successfully synchronized!")
    else:
        print("No rows in predictions table. Sequence is fine.")
        
    conn.close()
except Exception as e:
    print(f"Database error: {e}")

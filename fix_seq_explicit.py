import psycopg2
database_url = "postgresql://neondb_owner:npg_4vHx3ImSRcBr@ep-falling-math-aq90qlcp-pooler.c-8.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
conn = psycopg2.connect(database_url)
cursor = conn.cursor()
cursor.execute("SELECT MAX(id), COUNT(id) FROM predictions;")
print(cursor.fetchone())
cursor.execute("SELECT setval('predictions_id_seq', COALESCE((SELECT MAX(id)+1 FROM predictions), 1), false);")
conn.commit()
conn.close()
print("Fixed sequence.")

import os
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

client_id = os.environ.get('GOOGLE_CLIENT_ID')
print(f"DEBUG: GOOGLE_CLIENT_ID: {client_id}")
print(f"DEBUG: GOOGLE_CLIENT_ID length: {len(client_id) if client_id else 0}")

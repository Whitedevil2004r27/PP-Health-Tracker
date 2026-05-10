import sys
import os
import serverless_wsgi

# Add the root directory to the python path so it can find app.py
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root)

from app import app

def handler(event, context):
    return serverless_wsgi.handle_request(app, event, context)

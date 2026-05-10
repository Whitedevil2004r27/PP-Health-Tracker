import sys
import os
import serverless_wsgi

# Add the root directory to the python path so it can find app.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from app import app

def handler(event, context):
    return serverless_wsgi.handle_request(app, event, context)

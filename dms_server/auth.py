import os
from functools import wraps
from flask import request, jsonify

API_TOKEN = os.environ.get('DMS_API_TOKEN')
AUTH_REQUIRED = os.environ.get('DMS_REQUIRE_API_TOKEN', '0').lower() in {'1', 'true', 'yes'}

ALLOWED_ORIGINS = os.environ.get(
    'DMS_ALLOWED_ORIGINS', 'http://127.0.0.1:5000,http://localhost:5000'
).split(',')


def require_api_token(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not AUTH_REQUIRED:
            return f(*args, **kwargs)
        if not API_TOKEN:
            return jsonify({'error': 'DMS_API_TOKEN must be configured when authentication is enabled'}), 503
        token = request.headers.get('X-API-Token')
        if not token:
            return jsonify({'error': 'Missing API token'}), 401
        if token != API_TOKEN:
            return jsonify({'error': 'Invalid API token'}), 403
        return f(*args, **kwargs)
    return decorated


def get_client_ip():
    if request.headers.get('X-Forwarded-For'):
        return request.headers.get('X-Forwarded-For').split(',')[0].strip()
    return request.remote_addr

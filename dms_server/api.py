from flask import Blueprint, current_app, request

api_bp = Blueprint('api', __name__, url_prefix='/api')


@api_bp.route('/status', methods=['GET'])
def get_status():
    state = current_app.config['DMS_STATE']
    return {
        'model_loaded': state['model'] is not None,
        'yolo_loaded': state['yolo_model'] is not None,
        'tracker_initialized': state['facial_tracker'] is not None,
        'camera_connected': True,
    }


@api_bp.route('/telemetry', methods=['GET'])
def get_telemetry():
    import time
    telemetry = current_app.config['DMS_STATE']['telemetry']
    elapsed = int(time.time() - telemetry['session_start'])
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    session_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    resp = dict(telemetry)
    resp['session_time'] = session_time
    return resp


@api_bp.route('/sessions', methods=['GET'])
def get_sessions():
    telemetry = current_app.config['DMS_STATE']['telemetry']
    return {
        'sessions': [{
            'id': 'current',
            'start_time': telemetry.get('session_start', 0),
            'total_yawns': telemetry.get('total_yawns', 0),
            'total_drowsy': telemetry.get('total_drowsy', 0),
            'total_distractions': telemetry.get('total_distractions', 0),
        }]
    }


@api_bp.route('/alerts/config', methods=['GET', 'POST'])
def alerts_config():
    if request.method == 'GET':
        return {
            'thresholds': {
                'safe_score': 80,
                'warning_score': 45,
            },
            'audio_enabled': True,
        }

    config = request.get_json()
    return {'status': 'updated', 'config': config}


@api_bp.route('/health', methods=['GET'])
def health_check():
    return {'status': 'healthy', 'service': 'dms-api'}

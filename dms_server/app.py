"""WSGI access point for the canonical dashboard in the repository root."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_SERVER_PATH = Path(__file__).resolve().parents[1] / "dms_server.py"
_SPEC = spec_from_file_location("driver_monitoring_dashboard", _SERVER_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - impossible in a normal checkout
    raise RuntimeError(f"Unable to load dashboard module from {_SERVER_PATH}")
_dashboard = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_dashboard)

app = _dashboard.app
generate_frames = _dashboard.generate_frames
telemetry = _dashboard.telemetry


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False, threaded=True)

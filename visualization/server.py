import os
import time
import threading
import pandas as pd
from flask import Flask, render_template
from flask_socketio import SocketIO
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data.csv")

app = Flask(__name__)
app.config["SECRET_KEY"] = "csvchart"
socketio = SocketIO(app, cors_allowed_origins="*")

_last_payload = None
_lock = threading.Lock()


def read_csv_data():
    """Read the CSV and return a JSON-serializable payload, sorted by count desc."""
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    if "word" not in df.columns or "count" not in df.columns:
        raise ValueError("CSV must have 'word' and 'count' columns")
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0)
    df = df.sort_values("count", ascending=False).reset_index(drop=True)
    return {
        "words": df["word"].astype(str).tolist(),
        "counts": df["count"].tolist(),
        "total": int(df["count"].sum()),
        "updated_at": time.strftime("%H:%M:%S"),
    }


def broadcast_update():
    global _last_payload
    try:
        payload = read_csv_data()
    except Exception as e:
        socketio.emit("csv_error", {"error": str(e)})
        return
    with _lock:
        _last_payload = payload
    socketio.emit("csv_update", payload)


class CSVChangeHandler(FileSystemEventHandler):
    def __init__(self):
        self._last_fire = 0

    def _maybe_fire(self, event_path):
        if os.path.abspath(event_path) != os.path.abspath(CSV_PATH):
            return
        now = time.time()
        # debounce rapid-fire filesystem events
        if now - self._last_fire < 0.3:
            return
        self._last_fire = now
        time.sleep(0.05)  # let the writer finish flushing
        broadcast_update()

    def on_modified(self, event):
        self._maybe_fire(event.src_path)

    def on_created(self, event):
        self._maybe_fire(event.src_path)


def start_watcher():
    handler = CSVChangeHandler()
    observer = Observer()
    observer.schedule(handler, path=BASE_DIR, recursive=False)
    observer.daemon = True
    observer.start()
    return observer


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("connect")
def on_connect():
    global _last_payload
    with _lock:
        payload = _last_payload
    if payload is None:
        try:
            payload = read_csv_data()
            with _lock:
                _last_payload = payload
        except Exception as e:
            socketio.emit("csv_error", {"error": str(e)})
            return
    socketio.emit("csv_update", payload)


if __name__ == "__main__":
    start_watcher()
    print(f"Watching {CSV_PATH} for changes...")
    print("Open http://localhost:5000 in your browser")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)
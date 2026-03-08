"""
BowlForm AI — app.py
====================
Single entry point. Run:
    python app.py
Then open:  http://127.0.0.1:5000
"""

import os
import uuid
import json
import queue
import threading
import mimetypes

from flask import (
    Flask, request, jsonify, Response,
    send_file, render_template_string, stream_with_context
)
from flask_cors import CORS

# ── Import our analysis engine ──────────────────────────────────
from backend import analyze_bowling_video

# ── Flask app ───────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB upload limit

UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "uploads")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# In-memory job store: { job_id: { status, pct, result, orig_path, out_path } }
jobs: dict = {}
jobs_lock = threading.Lock()

# ── Helpers ─────────────────────────────────────────────────────

def set_job(job_id: str, **kwargs):
    with jobs_lock:
        if job_id not in jobs:
            jobs[job_id] = {}
        jobs[job_id].update(kwargs)


def get_job(job_id: str) -> dict:
    with jobs_lock:
        return dict(jobs.get(job_id, {}))


# ── Routes ───────────────────────────────────────────────────────

@app.route("/health")
def health_check():
    """Simple uptime check for Hugging Face."""
    return jsonify({"status": "ok", "app": "bowlform-ai"})

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Accept a video upload, start background analysis, return job_id immediately.
    Frontend polls /progress/<job_id> via SSE.
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]
    if video_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    job_id    = str(uuid.uuid4())
    orig_ext  = os.path.splitext(video_file.filename)[1].lower() or ".mp4"
    orig_path = os.path.join(UPLOADS_DIR, f"{job_id}_original{orig_ext}")
    out_path  = os.path.join(OUTPUTS_DIR, f"{job_id}_annotated.webm")
    skel_path = os.path.join(OUTPUTS_DIR, f"{job_id}_skeleton.webm")

    video_file.save(orig_path)

    set_job(job_id,
            status="queued",
            pct=0,
            result=None,
            orig_path=orig_path,
            out_path=out_path,
            skel_path=skel_path)

    # Run analysis in background thread
    thread = threading.Thread(
        target=_run_analysis,
        args=(job_id, orig_path, out_path, skel_path),
        daemon=True
    )
    thread.start()

    return jsonify({"job_id": job_id})


def _run_analysis(job_id: str, orig_path: str, out_path: str, skel_path: str):
    """Background worker: runs analyze_bowling_video and updates job state."""
    set_job(job_id, status="running", pct=5)

    def progress(pct: int):
        set_job(job_id, pct=pct)

    try:
        result = analyze_bowling_video(
            video_path=orig_path,
            output_path=out_path,
            skel_path=skel_path,
            progress_callback=progress
        )
        set_job(job_id, status="done", pct=100, result=result)
    except Exception as exc:
        set_job(job_id, status="error", pct=100,
                result={"error": str(exc)})


@app.route("/progress/<job_id>")
def progress(job_id: str):
    """
    Server-Sent Events stream.
    Emits JSON events: {"pct": 42}   (0-99 = in progress, 100 = done)
    """
    def generate():
        last_pct = -1
        while True:
            job = get_job(job_id)
            if not job:
                yield f"data: {json.dumps({'error': 'job not found'})}\n\n"
                return

            pct    = job.get("pct", 0)
            status = job.get("status", "queued")

            if pct != last_pct:
                yield f"data: {json.dumps({'pct': pct, 'status': status})}\n\n"
                last_pct = pct

            if status in ("done", "error"):
                # Send final event then close
                yield f"data: {json.dumps({'pct': 100, 'status': status})}\n\n"
                return

            # Small sleep to avoid busy-loop
            import time
            time.sleep(0.4)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # Nginx: disable buffering
        }
    )


@app.route("/results/<job_id>")
def results(job_id: str):
    """Return the full JSON analysis result."""
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job.get("status") != "done":
        return jsonify({"error": "Analysis not complete yet", "status": job.get("status")}), 202
    return jsonify(job.get("result", {}))


@app.route("/video/original/<job_id>")
def video_original(job_id: str):
    """Stream the original uploaded video."""
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    path = job.get("orig_path", "")
    if not path or not os.path.exists(path):
        return jsonify({"error": "Original video not found"}), 404
    return _stream_video(path)


@app.route("/video/annotated/<job_id>")
def video_annotated(job_id: str):
    """Stream the skeleton-annotated output video."""
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    path = job.get("out_path", "")
    if not path or not os.path.exists(path):
        return jsonify({"error": "Annotated video not found"}), 404
    return _stream_video(path)


@app.route("/video/skeleton/<job_id>")
def video_skeleton(job_id: str):
    """Stream the skeleton-only (black background) video."""
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    # Try skel_path from job store first, fall back to result dict
    path = job.get("skel_path") or (job.get("result") or {}).get("skeleton_video", "")
    if not path or not os.path.exists(path):
        return jsonify({"error": "Skeleton video not found"}), 404
    return _stream_video(path)


@app.route("/framemetrics/<job_id>")
def frame_metrics(job_id: str):
    """Return the per-frame metrics JSON for real-time frontend sync."""
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    result = job.get("result") or {}
    metrics = result.get("frame_metrics", {})
    fps     = result.get("fps", 30.0)
    return jsonify({"fps": fps, "metrics": metrics})


def _stream_video(path: str):
    """
    Range-aware video streaming so <video> controls (seek/scrub) work in browser.
    """
    file_size   = os.path.getsize(path)
    range_header = request.headers.get("Range", None)

    mime = mimetypes.guess_type(path)[0] or "video/mp4"

    if not range_header:
        return send_file(path, mimetype=mime)

    # Parse Range header  e.g. "bytes=0-1023"
    byte_range = range_header.replace("bytes=", "").split("-")
    start = int(byte_range[0])
    end   = int(byte_range[1]) if byte_range[1] else file_size - 1
    end   = min(end, file_size - 1)
    length = end - start + 1

    def generate_chunk():
        with open(path, "rb") as fh:
            fh.seek(start)
            remaining = length
            chunk = 65536  # 64 KB
            while remaining > 0:
                data = fh.read(min(chunk, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    resp = Response(
        stream_with_context(generate_chunk()),
        status=206,
        mimetype=mime,
        headers={
            "Content-Range":  f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges":  "bytes",
            "Content-Length": str(length),
        }
    )
    return resp


# ── Entry point ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  BowlForm AI Backend — Starting Server")
    print("  Listening on 0.0.0.0:7860")
    print("=" * 50 + "\n")
    # Hugging Face Spaces expose port 7860
    app.run(debug=False, host="0.0.0.0", port=7860, threaded=True)

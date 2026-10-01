import hmac
import json
import os

from flask import Flask, jsonify, request

import face_pipeline as fp

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB per request

API_KEY = os.environ.get("API_KEY", "")


@app.before_request
def require_api_key():
    if not API_KEY:
        return None
    if request.path in ("/", "/api/health"):
        return None
    sent = request.headers.get("X-API-Key", "")
    if not hmac.compare_digest(sent, API_KEY):
        return jsonify({"error": "Invalid or missing API key."}), 401
    return None


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "Upload too large."}), 413


@app.get("/")
def index():
    return jsonify({"service": "face-verification", "status": "ok"})


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/register")
def register():
    files = request.files.getlist("photos")
    if not files:
        return jsonify({"error": "No photos uploaded (use field 'photos')."}), 400

    samples = []
    skipped = []
    for f in files:
        image = fp.decode_image(f.read())
        if image is None:
            skipped.append({"filename": f.filename, "reason": "could not read image"})
            continue

        analysis = fp.analyze_image(image)
        if analysis is None:
            skipped.append({"filename": f.filename, "reason": "no face detected"})
            continue

        features, _, _ = analysis
        samples.append(features)

    if len(samples) < fp.MIN_SAMPLES:
        return jsonify({
            "error": f"Need at least {fp.MIN_SAMPLES} photos with a detectable face "
                     f"(got {len(samples)}). Face the camera in good light and retake.",
            "skipped": skipped,
        }), 400

    return jsonify({
        "reference": fp.build_reference(samples),
        "num_used": len(samples),
        "skipped": skipped,
    })


@app.post("/api/verify")
def verify():
    photo = request.files.get("photo")
    if photo is None:
        return jsonify({"error": "No photo uploaded (use field 'photo')."}), 400

    raw_reference = request.form.get("reference", "")
    if not raw_reference:
        return jsonify({"error": "Missing 'reference'. Register your face first."}), 400

    try:
        reference = json.loads(raw_reference)
        fp.validate_reference(reference)
    except ValueError as exc:  # includes json.JSONDecodeError
        return jsonify({"error": str(exc)}), 400

    image = fp.decode_image(photo.read())
    if image is None:
        return jsonify({"error": "Could not read the uploaded image."}), 400

    analysis = fp.analyze_image(image)
    if analysis is None:
        return jsonify({"face_detected": False})

    features, edges, gradient = analysis
    score, per_feature = fp.compare_to_reference(features, reference)

    return jsonify({
        "face_detected": True,
        "match": score <= fp.MATCH_THRESHOLD,
        "score": score,
        "threshold": fp.MATCH_THRESHOLD,
        "per_feature": per_feature,
        "edge_image": fp.png_base64(edges),
        "gradient_image": fp.png_base64(gradient),
    })


if __name__ == "__main__":
    # Local testing only. On Render, gunicorn runs the app (see the guide).
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)

"""
face_pipeline.py

The same edge-detection pipeline as get_face_reference_haar.py /
verify_face_haar.py (your teacher's steps), packaged for the web server:

    Original Image -> Grayscale -> Noise Reduction (Gaussian blur)
    -> Image Gradient (Sobel) -> Edge Detection (Canny) -> Edge Image
    -> features -> compare with the reference

Face detection is Haar Cascade with the original tight crop (no margin).
"""

import base64
import os

import cv2
import numpy as np

FACE_SIZE = (200, 200)

# Phone photos are large; shrink so the longest side is at most this many
# pixels before detection. Reference and verify both go through this, so
# they stay consistent, and it keeps the server fast on the free plan.
MAX_IMAGE_SIDE = 800

# Same scoring settings as verify_face_haar.py
MIN_REL_TOLERANCE = 0.05   # smallest allowed spread = 5% of the reference mean
MATCH_THRESHOLD = 2.5    # average deviation allowed for "MY FACE"
MIN_SAMPLES = 2

SCALAR_KEYS = ["mean_intensity", "std_intensity", "edge_density"]
QUADRANT_KEYS = ["top_left", "top_right", "bottom_left", "bottom_right"]

_cascade_path = os.path.join(
    cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
)
_face_cascade = cv2.CascadeClassifier(_cascade_path)
if _face_cascade.empty():
    raise RuntimeError(f"Could not load Haar cascade at {_cascade_path}")


# ---------------------------------------------------------
# Image input + face detection (Haar, tight crop)
# ---------------------------------------------------------
def decode_image(image_bytes):
    """Uploaded bytes -> BGR image (EXIF rotation applied), shrunk if huge."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        return None

    h, w = image.shape[:2]
    longest = max(h, w)
    if longest > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / longest
        image = cv2.resize(
            image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    return image


def find_face(image):
    """
    Detect the face with Haar Cascade and return the TIGHT crop (no extra
    margin) resized to FACE_SIZE, or None if no face is found.

    Phone cameras sometimes save pictures sideways, so if nothing is found
    upright we also try the image rotated 90 degrees each way.
    """
    for rotation in (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
        img = image if rotation is None else cv2.rotate(image, rotation)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = _face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: int(f[2]) * int(f[3]))
            return cv2.resize(img[y:y + h, x:x + w], FACE_SIZE)
    return None


# ---------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------
def step_convert_to_grayscale(face_bgr):
    return cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)


def step_noise_reduction(gray):
    return cv2.GaussianBlur(gray, (5, 5), 0)


def step_calculate_gradient(blurred):
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    return cv2.convertScaleAbs(cv2.magnitude(sobel_x, sobel_y))


def step_apply_edge_detection(blurred):
    return cv2.Canny(blurred, 50, 150)


def extract_features(face_gray):
    blurred = step_noise_reduction(face_gray)
    edges = step_apply_edge_detection(blurred)

    h, w = edges.shape
    half_h, half_w = h // 2, w // 2
    quadrants = {
        "top_left": edges[0:half_h, 0:half_w],
        "top_right": edges[0:half_h, half_w:w],
        "bottom_left": edges[half_h:h, 0:half_w],
        "bottom_right": edges[half_h:h, half_w:w],
    }

    features = {
        "mean_intensity": float(np.mean(blurred)),
        "std_intensity": float(np.std(blurred)),
        "edge_density": float(np.sum(edges == 255) / edges.size),
        "quadrant_densities": {
            name: float(np.sum(q == 255) / q.size) for name, q in quadrants.items()
        },
    }
    return features, edges


def analyze_image(image):
    """
    Full pipeline on one BGR image.
    Returns (features, edges, gradient) or None if no face was found.
    """
    face_bgr = find_face(image)
    if face_bgr is None:
        return None

    gray = step_convert_to_grayscale(face_bgr)
    features, edges = extract_features(gray)
    gradient = step_calculate_gradient(step_noise_reduction(gray))
    return features, edges, gradient


# ---------------------------------------------------------
# Reference (mean +/- std over several samples)
# ---------------------------------------------------------
def build_reference(samples):
    mean_out, std_out = {}, {}

    for key in SCALAR_KEYS:
        values = [s[key] for s in samples]
        mean_out[key] = float(np.mean(values))
        std_out[key] = float(np.std(values))

    mean_out["quadrant_densities"] = {}
    std_out["quadrant_densities"] = {}
    for qkey in QUADRANT_KEYS:
        values = [s["quadrant_densities"][qkey] for s in samples]
        mean_out["quadrant_densities"][qkey] = float(np.mean(values))
        std_out["quadrant_densities"][qkey] = float(np.std(values))

    return {"num_samples": len(samples), "mean": mean_out, "std": std_out}


def validate_reference(reference):
    """Raise ValueError if the reference sent by the app is malformed."""
    try:
        for part in ("mean", "std"):
            for key in SCALAR_KEYS:
                float(reference[part][key])
            for qkey in QUADRANT_KEYS:
                float(reference[part]["quadrant_densities"][qkey])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Reference data is missing or malformed.") from exc


# ---------------------------------------------------------
# Comparison
# ---------------------------------------------------------
def _deviation(live_value, ref_mean, ref_std):
    spread = max(ref_std, MIN_REL_TOLERANCE * abs(ref_mean))
    return abs(live_value - ref_mean) / spread


def compare_to_reference(features, reference):
    """Returns (overall_score, per_feature_deviation)."""
    mean, std = reference["mean"], reference["std"]

    per_feature = {}
    for key in SCALAR_KEYS:
        per_feature[key] = _deviation(features[key], mean[key], std[key])
    for qkey in QUADRANT_KEYS:
        per_feature[f"quadrant_{qkey}"] = _deviation(
            features["quadrant_densities"][qkey],
            mean["quadrant_densities"][qkey],
            std["quadrant_densities"][qkey],
        )

    overall = float(np.mean(list(per_feature.values())))
    return overall, per_feature


def png_base64(image):
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")

from pathlib import Path
import base64

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, request, send_from_directory


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.onnx"

app = Flask(__name__, static_folder=None)


# =========================================================
# Load ONNX model
# =========================================================

print("Loading ONNX model...")

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"]
)

input_info = session.get_inputs()[0]
output_info = session.get_outputs()[0]

input_name = input_info.name
output_name = output_info.name

print("ONNX model loaded!")
print(f"Input : {input_name} {input_info.shape}")
print(f"Output: {output_name} {output_info.shape}")


# =========================================================
# Image preprocessing
# =========================================================

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Convert an OpenCV image into an MNIST-style tensor.

    Output:
        shape = (1, 28, 28, 1)
        dtype = float32
        range = 0-1
    """

    if image is None:
        raise ValueError("Image could not be decoded.")

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # White digit on black background
    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Find digit
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError(
            "No digit detected. Draw a larger, clearer digit."
        )

    largest = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest) < 20:
        raise ValueError(
            "Digit is too small. Draw it larger."
        )

    # Crop digit
    x, y, w, h = cv2.boundingRect(largest)

    digit = thresh[
        y:y + h,
        x:x + w
    ]

    # =====================================================
    # Resize digit into a 20x20 region
    # =====================================================

    target = 20

    if w >= h:

        new_w = target

        new_h = max(
            1,
            round(h * target / w)
        )

    else:

        new_h = target

        new_w = max(
            1,
            round(w * target / h)
        )

    resized = cv2.resize(
        digit,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # =====================================================
    # Center inside 28x28 canvas
    # =====================================================

    canvas = np.zeros(
        (28, 28),
        dtype=np.uint8
    )

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    # Normalize
    normalized = (
        canvas.astype(np.float32) / 255.0
    )

    # IMPORTANT:
    # Our ONNX model expects NHWC
    #
    # (batch, height, width, channels)
    #
    # (1, 28, 28, 1)

    processed = normalized.reshape(
        1,
        28,
        28,
        1
    )

    return processed


# =========================================================
# Frontend
# =========================================================

@app.get("/")
def home():

    return send_from_directory(
        BASE_DIR,
        "index.html"
    )


# =========================================================
# Health check
# =========================================================

@app.get("/health")
def health():

    return jsonify({
        "status": "ok",
        "runtime": "ONNX Runtime"
    })


# =========================================================
# Prediction
# =========================================================

@app.post("/predict")
def predict():

    try:

        payload = request.get_json(
            silent=True
        ) or {}

        image_data = payload.get("image")

        if not image_data:

            return jsonify(
                error="No image provided."
            ), 400

        # Remove:
        #
        # data:image/png;base64,...

        if "," in image_data:

            image_data = image_data.split(
                ",",
                1
            )[1]

        # Base64 -> bytes

        try:

            raw = base64.b64decode(
                image_data,
                validate=True
            )

        except Exception:

            return jsonify(
                error="Invalid image data."
            ), 400

        # bytes -> OpenCV

        buffer = np.frombuffer(
            raw,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            buffer,
            cv2.IMREAD_COLOR
        )

        if image is None:

            return jsonify(
                error="Could not decode image."
            ), 400

        # Preprocess

        processed = preprocess_image(
            image
        )

        # =================================================
        # ONNX prediction
        # =================================================

        outputs = session.run(
            [output_name],
            {
                input_name: processed
            }
        )

        probabilities = np.asarray(
            outputs[0]
        )[0]

        digit = int(
            np.argmax(probabilities)
        )

        confidence = float(
            probabilities[digit]
        )

        # =================================================
        # Response
        # =================================================

        return jsonify(
            digit=digit,
            confidence=confidence,
            probabilities=[
                float(value)
                for value in probabilities
            ]
        )

    except ValueError as exc:

        return jsonify(
            error=str(exc)
        ), 400

    except Exception as exc:

        app.logger.exception(
            "Prediction failed"
        )

        return jsonify(
            error=f"Prediction failed: {exc}"
        ), 500


# =========================================================
# Local development
# =========================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )

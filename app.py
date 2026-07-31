from pathlib import Path
import base64

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.keras"

app = Flask(__name__, static_folder=None)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"Model loaded. Input: {model.input_shape}, Output: {model.output_shape}")


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Convert an OpenCV image into a centered MNIST-style (1, 28, 28, 1) tensor."""
    if image is None:
        raise ValueError("Image could not be decoded.")

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError("No digit detected. Draw a larger, clearer digit.")

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 20:
        raise ValueError("Digit is too small. Draw it larger.")

    x, y, w, h = cv2.boundingRect(largest)
    digit = thresh[y:y + h, x:x + w]

    # Fit the digit into a 20x20 box while preserving aspect ratio,
    # then center it on a 28x28 black canvas (MNIST-like layout).
    target = 20
    if w >= h:
        new_w = target
        new_h = max(1, round(h * target / w))
    else:
        new_h = target
        new_w = max(1, round(w * target / h))

    resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_off = (28 - new_w) // 2
    y_off = (28 - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    normalized = canvas.astype("float32") / 255.0
    return normalized.reshape(1, 28, 28, 1)


@app.get("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.post("/predict")
def predict():
    try:
        payload = request.get_json(silent=True) or {}
        image_data = payload.get("image")

        if not image_data:
            return jsonify(error="No image provided."), 400

        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        try:
            raw = base64.b64decode(image_data, validate=True)
        except Exception:
            return jsonify(error="Invalid image data."), 400

        buffer = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        processed = preprocess_image(image)

        probabilities = model.predict(processed, verbose=0)[0]
        digit = int(np.argmax(probabilities))

        return jsonify(
            digit=digit,
            confidence=float(probabilities[digit]),
            probabilities=[float(x) for x in probabilities],
        )
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify(error=f"Prediction failed: {exc}"), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

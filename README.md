# Handwritten Digit Recognition — Flask + Keras

This version does **not** use TensorFlow.js.

## Files

- `app.py` — Flask server and prediction API
- `index.html` — drawing UI
- `model.keras` — trained CNN
- `requirements.txt` — Python packages
- `train_model.py` — optional retraining script, when included
- `accuracy.png` / `loss.png` — optional training graphs, when included

## Windows setup

Open Command Prompt in this folder:

```bat
python -m venv mnist_env
mnist_env\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Then open:

http://127.0.0.1:5000

Do not open `index.html` directly. Flask serves the page and provides `/predict`.

## Retraining

If `train_model.py` is present, running it replaces `model.keras` with a newly trained model:

```bat
python train_model.py
```

Then restart `python app.py`.

# PRAICP-1002 · Traffic Sign Recognition (VS Code + Streamlit)

CNN that recognises 43 traffic signs, with a red / white / black Streamlit web app.
No dataset upload is needed - `train.py` downloads and extracts the data by itself.

## 1. Setup (VS Code terminal)
Use **Python 3.10 - 3.12** (TensorFlow does not support newer versions yet).

```bash
# open this folder in VS Code:  File > Open Folder...
python -m venv .venv

# activate it
.venv\Scripts\activate            # Windows (PowerShell / CMD)
source .venv/bin/activate         # macOS / Linux

pip install -r requirements.txt
```
In VS Code press `Ctrl+Shift+P` > *Python: Select Interpreter* > choose `.venv`.

## 2. Train the model (creates everything the app needs)
```bash
python train.py --quick          # 3-5 min dry-run: checks that everything works
python train.py --skip-tuning    # reuses the notebook's best hyperparameters (recommended)
python train.py                  # full run incl. Keras Tuner search (slow on CPU)
```
What it does: downloads the ZIP into `data/` -> extracts 400 images per class (memory-safe) ->
checks images -> 70/15/15 stratified split -> baseline CNN -> tuned CNN -> picks the best on
validation accuracy -> saves model, metrics, plots, and demo images.

Useful options: `--max-per-class 0` (use all images), `--final-epochs 60`, `--trials 12`,
`--delete-zip` (free disk space after extracting), `--zip-path your.zip` (if the auto-download is blocked,
download the ZIP manually and point to it).

## 3. Run the web app
```bash
streamlit run app.py
```
It opens at http://localhost:8501 . Tabs: **Predict** (upload / camera / test sample, top-5 chart,
confidence warning) · **Model performance** (metrics, baseline vs tuned, per-sign F1, confusion matrix) ·
**Sign library** (all 43 signs, searchable).

### Already trained in Colab?
Skip step 2: copy `TrafficSign_CNN_Model.keras` and `TrafficSign_class_names.json` from your notebook
into the `models/` folder and run the app. (Reference photos and the performance tab appear after running `train.py`.)

## Project layout
```
app.py            Streamlit frontend
train.py          download + extract + preprocess + train + evaluate + save
config.py         paths, dataset URL, sign names, default hyperparameters
requirements.txt  dependencies
models/           saved model, class names, metrics.json   (created by train.py)
assets/           reference photos, test samples, plots    (created by train.py)
data/             dataset ZIP + extracted subset           (created by train.py)
.streamlit/       red / white / black theme
.vscode/          one-click Run/Debug configurations (F5)
```

## Troubleshooting
- **`No matching distribution for tensorflow`** - your Python is too new; install Python 3.12 or 3.11.
- **Download blocked / fails** - download the ZIP from the URL printed by the script, save it as
  `data/PRAICP-1002-TrafSignDetc.zip`, and run `train.py` again.
- **Out of memory** - lower `--max-per-class` (e.g. 200).
- **App says model not found** - run `python train.py --skip-tuning` first.

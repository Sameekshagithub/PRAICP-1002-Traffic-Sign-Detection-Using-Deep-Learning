#!/usr/bin/env python
"""
PRAICP-1002 : Traffic Sign Recognition - full training pipeline (VS Code / terminal version of the notebook).

What this script does, in order
  1. Downloads the dataset ZIP automatically (no manual upload needed)
  2. Reads the ZIP index and extracts only MAX images per class (memory-safe)
  3. Checks images, resizes to 64x64 and splits 70 / 15 / 15 (stratified)
  4. Trains a baseline CNN, then tunes a CNN with Keras Tuner (or reuses the notebook's best hyperparameters)
  5. Evaluates both models, picks the best one on VALIDATION accuracy
  6. Saves the model, class names, metrics, plots, reference + sample images for the Streamlit app

Usage
  python train.py                  # full run like the notebook (tuning takes hours on CPU)
  python train.py --skip-tuning    # reuse the notebook's best hyperparameters (much faster)
  python train.py --quick          # tiny dry-run to check that everything works (a few minutes)
"""
import argparse
import gc
import json
import os
import posixpath
import random
import re
import shutil
import textwrap
import warnings
import zipfile

import matplotlib
matplotlib.use("Agg")                       # no pop-up windows: plots are saved to files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import Input, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten, MaxPooling2D, RandomContrast,
                                     RandomRotation, RandomTranslation, RandomZoom, Rescaling)
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt

import config as cfg

warnings.filterwarnings("ignore")
INPUT_SHAPE = (cfg.IMG_SIZE, cfg.IMG_SIZE, 3)


# ----------------------------------------------------------------------------------------------
# 1. Dataset: download -> index ZIP -> extract a subset
# ----------------------------------------------------------------------------------------------
def natural_key(s):
    """Sort '2' before '10'."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def download_file(url, dest):
    """Stream-download a file with a progress bar (never held fully in memory)."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60, headers={"User-Agent": "Mozilla/5.0"}) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="Downloading dataset") as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    os.replace(tmp, dest)                                  # only a complete download gets the final name


def ensure_zip(zip_path, url):
    if zip_path.exists() and zipfile.is_zipfile(zip_path):
        print(f"ZIP already present: {zip_path}")
        return
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        download_file(url, zip_path)
    except Exception as e:
        raise SystemExit(f"\nDownload failed: {e}\nDownload the file manually from:\n  {url}\n"
                         f"and save it as:\n  {zip_path}\nthen run this script again.")
    if not zipfile.is_zipfile(zip_path):
        raise SystemExit(f"The downloaded file is not a valid ZIP. Delete {zip_path} and try again.")


def index_zip(zip_path):
    """Read the ZIP index WITHOUT extracting: return (class_root, {class: [members]})."""
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist()
                   if m.lower().endswith(cfg.IMG_EXT) and not m.startswith("__MACOSX") and "/." not in m]
    by_parent = {}
    for m in members:
        by_parent.setdefault(posixpath.dirname(m), []).append(m)
    by_grand = {}
    for parent in by_parent:
        by_grand.setdefault(posixpath.dirname(parent), []).append(parent)
    class_root = max(by_grand, key=lambda g: len(by_grand[g]))
    return class_root, {posixpath.basename(p): by_parent[p] for p in by_grand[class_root]}


def extract_subset(zip_path, zip_members, max_per_class, extract_dir, seed):
    """Extract only `max_per_class` random images from every class."""
    marker = extract_dir / ".complete"
    if marker.exists():
        print(f'"{extract_dir.name}" already extracted, skipping.')
        return
    rng = random.Random(seed)
    selected = []
    for cls, files in zip_members.items():
        files = sorted(files)
        selected += files if not max_per_class or len(files) <= max_per_class else rng.sample(files, max_per_class)
    with zipfile.ZipFile(zip_path) as zf:
        for m in tqdm(selected, desc="Extracting selected images"):
            zf.extract(m, extract_dir)
    marker.write_text("ok")
    print(f"Extracted {len(selected):,} images into {extract_dir}")


# ----------------------------------------------------------------------------------------------
# 2. Preprocessing
# ----------------------------------------------------------------------------------------------
def preprocess_image(path, img_size=cfg.IMG_SIZE):
    """Load, convert to RGB, resize -> uint8 array (img_size, img_size, 3). Normalisation is inside the model."""
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB").resize((img_size, img_size), Image.LANCZOS), dtype=np.uint8)


# ----------------------------------------------------------------------------------------------
# 3. Models
# ----------------------------------------------------------------------------------------------
def make_augmentation():
    """Random changes active only during training. NO horizontal flip (left/right signs are mirror images)."""
    return Sequential([
        RandomRotation(factor=0.03, fill_mode="nearest"),
        RandomTranslation(height_factor=0.10, width_factor=0.10, fill_mode="nearest"),
        RandomZoom(height_factor=0.10, width_factor=0.10, fill_mode="nearest"),
        RandomContrast(factor=0.2),
    ], name="data_augmentation")


def build_cnn(num_classes, filters_1=32, filters_2=64, dense_units=128, dropout=0.5, learning_rate=0.001, name=None):
    model = Sequential([
        Input(shape=INPUT_SHAPE),
        make_augmentation(),
        Rescaling(1.0 / 255),
        Conv2D(filters_1, (3, 3), activation="relu"), MaxPooling2D((2, 2)),
        Conv2D(filters_2, (3, 3), activation="relu"), MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"), MaxPooling2D((2, 2)),
        Flatten(),
        Dense(dense_units, activation="relu"),
        Dropout(dropout),
        Dense(num_classes, activation="softmax"),
    ], name=name)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def make_hypermodel(num_classes):
    class TrafficSignHyperModel(kt.HyperModel):
        def build(self, hp):
            hp.Choice("batch_size", [16, 32])               # applied in fit() below
            return build_cnn(num_classes,
                             filters_1=hp.Choice("filters_1", [32, 64]),
                             filters_2=hp.Choice("filters_2", [64, 128]),
                             dense_units=hp.Choice("dense_units", [128, 256]),
                             dropout=hp.Float("dropout", 0.3, 0.5, step=0.1),
                             learning_rate=hp.Choice("learning_rate", [0.001, 0.0001]))

        def fit(self, hp, model, *args, **kwargs):
            return model.fit(*args, batch_size=hp.get("batch_size"), **kwargs)
    return TrafficSignHyperModel()


def get_callbacks(checkpoint_path, patience=5):
    return [EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
            ModelCheckpoint(str(checkpoint_path), monitor="val_loss", save_best_only=True)]


def evaluate_model(model, X, y):
    prob = model.predict(X, verbose=0)
    pred = np.argmax(prob, axis=1)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    return {"Accuracy": accuracy_score(y, pred), "Precision (macro)": p,
            "Recall (macro)": r, "F1-Score (macro)": f}, pred, prob


# ----------------------------------------------------------------------------------------------
# 4. Reports / plots
# ----------------------------------------------------------------------------------------------
def save_history_plot(history, title, path):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(history.history["accuracy"], label="Training", color="#111111")
    ax[0].plot(history.history["val_accuracy"], label="Validation", color="#D7141A")
    ax[0].set(title=f"{title} - Accuracy", xlabel="Epoch", ylabel="Accuracy"); ax[0].legend()
    ax[1].plot(history.history["loss"], label="Training", color="#111111")
    ax[1].plot(history.history["val_loss"], label="Validation", color="#D7141A")
    ax[1].set(title=f"{title} - Loss", xlabel="Epoch", ylabel="Loss"); ax[1].legend()
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)


def save_confusion_matrix(cm, class_names, title, path):
    plt.figure(figsize=(18, 15))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 6}, cbar=False)
    plt.xlabel("Predicted sign", fontsize=13); plt.ylabel("Actual sign", fontsize=13)
    plt.title(title, fontsize=15)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def save_app_images(df, paths_test, y_test, class_names):
    """One reference photo per sign + one unseen test photo per sign, for the Streamlit app."""
    for d in (cfg.REF_DIR, cfg.SAMPLES_DIR):
        shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True, exist_ok=True)
    for cls in class_names:
        first = df[df["class"] == cls]["path"].iloc[0]
        with Image.open(first) as im:
            im.convert("RGB").resize((128, 128), Image.LANCZOS).save(cfg.REF_DIR / f"{cls}.png")
    for k, cls in enumerate(class_names):
        idx = np.where(y_test == k)[0]
        if len(idx):
            with Image.open(paths_test[idx[0]]) as im:
                im.convert("RGB").save(cfg.SAMPLES_DIR / f"{cls}.png")


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train the traffic sign CNN (PRAICP-1002).")
    p.add_argument("--max-per-class", type=int, default=400, help="images used per class; 0 = whole dataset (default 400)")
    p.add_argument("--baseline-epochs", type=int, default=30)
    p.add_argument("--trials", type=int, default=8, help="Keras Tuner random-search trials")
    p.add_argument("--tuner-epochs", type=int, default=15)
    p.add_argument("--final-epochs", type=int, default=40)
    p.add_argument("--skip-tuning", action="store_true", help="reuse the notebook's best hyperparameters instead of searching")
    p.add_argument("--quick", action="store_true", help="tiny dry-run (60 images/class, very few epochs)")
    p.add_argument("--zip-path", default=str(cfg.ZIP_PATH), help="use an already downloaded dataset ZIP")
    p.add_argument("--delete-zip", action="store_true", help="delete the ZIP after extracting to save disk space")
    a = p.parse_args()
    if a.quick:
        a.max_per_class, a.baseline_epochs, a.trials, a.tuner_epochs, a.final_epochs = 60, 3, 2, 2, 3
    return a


def main():
    args = parse_args()
    seed = cfg.SEED
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
    for d in (cfg.DATA_DIR, cfg.MODELS_DIR, cfg.REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    print(f"TensorFlow {tf.__version__} | Keras Tuner {kt.__version__} | GPU: {bool(tf.config.list_physical_devices('GPU'))}")

    # ---- dataset -------------------------------------------------------------------------
    print("\n[1/7] Dataset")
    from pathlib import Path
    zip_path = Path(args.zip_path)
    ensure_zip(zip_path, cfg.DATASET_URL)
    class_root, zip_members = index_zip(zip_path)
    full_counts = pd.Series({c: len(v) for c, v in zip_members.items()})
    full_counts = full_counts.reindex(sorted(full_counts.index, key=natural_key))
    print(f"Classes: {len(full_counts)} | images in FULL dataset: {full_counts.sum():,} | class folder: {class_root!r}")

    extract_dir = cfg.DATA_DIR / f"TrafSign_data_{args.max_per_class or 'all'}"
    extract_subset(zip_path, zip_members, args.max_per_class, extract_dir, seed)
    if args.delete_zip and zip_path.exists():
        zip_path.unlink(); print("ZIP deleted to save disk space.")

    dataset_path = extract_dir / class_root
    class_names = sorted((d.name for d in dataset_path.iterdir() if d.is_dir()), key=natural_key)
    num_classes = len(class_names)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    use_names = all(c.isdigit() and int(c) in cfg.GTSRB_NAMES for c in class_names)
    sign_names = {c: (cfg.GTSRB_NAMES[int(c)] if use_names else c) for c in class_names}

    # ---- basic checks --------------------------------------------------------------------
    print("\n[2/7] Basic checks")
    rows = [{"class": c, "path": str(f)} for c in class_names
            for f in sorted((dataset_path / c).iterdir()) if f.suffix.lower() in cfg.IMG_EXT]
    df = pd.DataFrame(rows)
    bad = []
    for p in tqdm(df["path"], desc="Checking images"):
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception:
            bad.append(p)
    if bad:
        df = df[~df["path"].isin(bad)].reset_index(drop=True)
    print(f"Working images: {len(df):,} | corrupt removed: {len(bad)}")

    # ---- preprocessing + split -----------------------------------------------------------
    print("\n[3/7] Preprocessing and 70/15/15 stratified split")
    X = np.empty((len(df), cfg.IMG_SIZE, cfg.IMG_SIZE, 3), dtype=np.uint8)
    for k, p in enumerate(tqdm(df["path"], desc="Preprocessing images")):
        X[k] = preprocess_image(p)
    y = df["class"].map(class_to_idx).to_numpy()
    paths = df["path"].to_numpy()

    idx_all = np.arange(len(y))
    idx_train, idx_tmp = train_test_split(idx_all, test_size=0.30, stratify=y, random_state=seed)
    idx_val, idx_test = train_test_split(idx_tmp, test_size=0.50, stratify=y[idx_tmp], random_state=seed)
    X_train, y_train = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]
    X_test, y_test = X[idx_test], y[idx_test]
    paths_test = paths[idx_test]
    del X; gc.collect()
    print(f"Train {X_train.shape} | Validation {X_val.shape} | Test {X_test.shape}")

    cw = compute_class_weight("balanced", classes=np.arange(num_classes), y=y_train)
    class_weights = {i: float(w) for i, w in enumerate(cw)}

    # ---- baseline ------------------------------------------------------------------------
    print("\n[4/7] Baseline CNN")
    baseline = build_cnn(num_classes, name="baseline_cnn")
    hist_base = baseline.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=args.baseline_epochs,
                             batch_size=32, class_weight=class_weights,
                             callbacks=get_callbacks(cfg.MODELS_DIR / "best_baseline_cnn.keras"))
    base_val, _, _ = evaluate_model(baseline, X_val, y_val)
    base_test, pred_base, prob_base = evaluate_model(baseline, X_test, y_test)
    print(f"Baseline - validation acc {base_val['Accuracy']:.4f} | test acc {base_test['Accuracy']:.4f}")

    # ---- tuning / final model ------------------------------------------------------------
    print("\n[5/7] Final CNN")
    if args.skip_tuning:
        best_params = dict(cfg.DEFAULT_BEST_HP)
        print("Tuning skipped - using the notebook's best hyperparameters:", best_params)
    else:
        tuner = kt.RandomSearch(make_hypermodel(num_classes), objective="val_accuracy", max_trials=args.trials,
                                seed=seed, overwrite=True, directory=str(cfg.ROOT / "trafsign_tuning"),
                                project_name="trafsigndetc")
        tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=args.tuner_epochs,
                     class_weight=class_weights,
                     callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)])
        best_params = dict(tuner.get_best_hyperparameters(1)[0].values)
        print("Best hyperparameters:", best_params)

    final = build_cnn(num_classes, filters_1=best_params["filters_1"], filters_2=best_params["filters_2"],
                      dense_units=best_params["dense_units"], dropout=best_params["dropout"],
                      learning_rate=best_params["learning_rate"], name="final_cnn")
    hist_final = final.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=args.final_epochs,
                           batch_size=best_params["batch_size"], class_weight=class_weights,
                           callbacks=get_callbacks(cfg.MODELS_DIR / "best_final_cnn.keras", patience=6))
    final_val, _, _ = evaluate_model(final, X_val, y_val)
    final_test, pred_final, prob_final = evaluate_model(final, X_test, y_test)
    print(f"Final - validation acc {final_val['Accuracy']:.4f} | test acc {final_test['Accuracy']:.4f}")

    # ---- select best on VALIDATION accuracy ----------------------------------------------
    print("\n[6/7] Evaluation of the best model")
    if final_val["Accuracy"] >= base_val["Accuracy"]:
        best_name, best_model, pred_best, hist_best = "Final Tuned CNN", final, pred_final, hist_final
    else:
        best_name, best_model, pred_best, hist_best = "Baseline CNN", baseline, pred_base, hist_base
    print("Selected model:", best_name)

    names_full = [f"{c}: {sign_names[c]}" for c in class_names]
    report_txt = classification_report(y_test, pred_best, target_names=names_full, zero_division=0)
    (cfg.REPORT_DIR / "classification_report.txt").write_text(report_txt, encoding="utf-8")
    print(report_txt)

    cm = confusion_matrix(y_test, pred_best, labels=np.arange(num_classes))
    save_confusion_matrix(cm, class_names, f"Confusion Matrix - {best_name} (Test Set)", cfg.REPORT_DIR / "confusion_matrix.png")
    save_history_plot(hist_best, best_name, cfg.REPORT_DIR / "training_curves.png")

    _, _, f1_cls, _ = precision_recall_fscore_support(y_test, pred_best, labels=np.arange(num_classes), zero_division=0)
    confused = sorted(({"actual": names_full[a], "predicted": names_full[p], "count": int(cm[a, p])}
                       for a in range(num_classes) for p in range(num_classes) if a != p and cm[a, p] > 0),
                      key=lambda d: -d["count"])[:10]
    best_test = {"Accuracy": accuracy_score(y_test, pred_best)}
    pr, rc, f1m, _ = precision_recall_fscore_support(y_test, pred_best, average="macro", zero_division=0)
    best_test.update({"Precision (macro)": pr, "Recall (macro)": rc, "F1-Score (macro)": f1m})

    # ---- save everything the app needs ---------------------------------------------------
    print("\n[7/7] Saving model, metrics and app images")
    best_model.save(cfg.MODEL_PATH)
    with open(cfg.CLASS_JSON, "w") as f:
        json.dump({"class_folders": class_names, "sign_names": sign_names, "img_size": cfg.IMG_SIZE}, f, indent=2)
    metrics = {
        "selected_model": best_name,
        "test": best_test,
        "comparison": {"Baseline CNN - Validation": base_val, "Baseline CNN - Test": base_test,
                       "Final Tuned CNN - Validation": final_val, "Final Tuned CNN - Test": final_test},
        "per_class_f1": {names_full[i]: float(f1_cls[i]) for i in range(num_classes)},
        "most_confused": confused,
        "best_hyperparameters": best_params,
        "dataset": {"classes": num_classes, "full_images": int(full_counts.sum()), "working_images": int(len(df)),
                    "train": int(len(y_train)), "validation": int(len(y_val)), "test": int(len(y_test)),
                    "img_size": cfg.IMG_SIZE, "misclassified_test": int((pred_best != y_test).sum())},
    }
    with open(cfg.METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    save_app_images(df, paths_test, y_test, class_names)

    reloaded = tf.keras.models.load_model(cfg.MODEL_PATH)
    same = np.allclose(best_model.predict(X_test[:20], verbose=0), reloaded.predict(X_test[:20], verbose=0), atol=1e-5)

    print("=" * 62)
    print("  PRAICP-1002 : TRAFFIC SIGN RECOGNITION - RESULTS")
    print("=" * 62)
    print(f"Classes             : {num_classes}")
    print(f"Images train/val/test: {len(y_train)} / {len(y_val)} / {len(y_test)}")
    print(f"Selected model      : {best_name}")
    print(f"Test accuracy       : {best_test['Accuracy'] * 100:.2f} %")
    print(f"Test F1 (macro)     : {best_test['F1-Score (macro)']:.4f}")
    print(f"Model saved         : {cfg.MODEL_PATH} (reload check: {same})")
    print("=" * 62)
    print("Now run the web app:   streamlit run app.py")


if __name__ == "__main__":
    main()

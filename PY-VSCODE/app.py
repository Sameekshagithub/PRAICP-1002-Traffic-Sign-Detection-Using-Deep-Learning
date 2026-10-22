"""
PRAICP-1002 : Traffic Sign Recognition - Streamlit frontend (red / white / black road-sign theme).

Run with:   streamlit run app.py
Needs the files created by train.py (or copied from the notebook run) in the  models/  folder:
    TrafficSign_CNN_Model.keras   and   TrafficSign_class_names.json
"""
import html
import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps

import config as cfg

st.set_page_config(page_title="Traffic Sign Recognition", page_icon="🛑", layout="wide")

RED, BLACK, WHITE = "#D7141A", "#111111", "#FFFFFF"

# ------------------------------------------------------------------------------------------------
# Styling (road-sign look: red / white / black)
# ------------------------------------------------------------------------------------------------
CSS = """
<style>
.block-container{padding-top:1.4rem;max-width:1200px;}
header[data-testid="stHeader"]{background:transparent;}

.hero{background:#111;color:#fff;border-radius:16px;padding:22px 28px;display:flex;align-items:center;gap:22px;
      border-bottom:7px solid #D7141A;}
.hero h1{margin:0;padding:0;font-size:2rem;letter-spacing:.09em;font-weight:800;color:#fff !important;}
.hero p{margin:6px 0 0;color:#e2e2e2 !important;font-size:.98rem;}
.noentry{width:68px;height:68px;border-radius:50%;background:#D7141A;border:4px solid #fff;
         display:flex;align-items:center;justify-content:center;flex:0 0 auto;box-shadow:0 0 0 3px #D7141A;}
.noentry::after{content:'';width:36px;height:9px;background:#fff;border-radius:2px;}
.road{height:12px;margin:14px 0 22px;border-radius:6px;border:1px solid #111;
      background:repeating-linear-gradient(90deg,#fff 0 30px,#111 30px 52px);}

.card{border:2px solid #111;border-radius:16px;padding:20px;background:#fff;}
.card-red{border-color:#D7141A;box-shadow:6px 6px 0 #111;}
.card-wait{border:3px dashed #111;text-align:center;padding:46px 20px;color:#111;}
.card-wait .big{font-size:3rem;line-height:1;}

.result{display:flex;align-items:center;gap:26px;flex-wrap:wrap;}
.sign{border-radius:50%;border:15px solid #D7141A;background:#fff;color:#111;display:flex;align-items:center;
      justify-content:center;font-weight:900;flex:0 0 auto;box-shadow:0 0 0 4px #111;}
.result .label{font-size:.8rem;letter-spacing:.16em;color:#D7141A;font-weight:800;}
.result .name{font-size:1.75rem;font-weight:800;color:#111;line-height:1.15;margin:4px 0 8px;}
.conf{display:inline-block;background:#111;color:#fff;border-radius:999px;padding:6px 16px;font-weight:700;}
.conf b{color:#fff;}

.caution{margin-top:16px;border:3px solid #111;border-radius:12px;overflow:hidden;}
.caution .stripe{height:12px;background:repeating-linear-gradient(135deg,#111 0 14px,#fff 14px 28px);}
.caution .txt{padding:10px 14px;font-weight:600;color:#111;background:#fff;}
.ok{margin-top:16px;border-left:8px solid #D7141A;background:#F3F3F3;padding:10px 14px;border-radius:8px;
    font-weight:600;color:#111;}

.bar-row{display:flex;align-items:center;gap:12px;margin:9px 0;}
.bar-label{flex:0 0 46%;font-size:.92rem;font-weight:600;color:#111;}
.bar-track{flex:1;height:16px;background:#E6E6E6;border-radius:8px;overflow:hidden;border:1px solid #111;}
.bar-fill{height:100%;background:#111;}
.bar-fill.top{background:#D7141A;}
.bar-val{flex:0 0 58px;text-align:right;font-weight:700;color:#111;font-size:.9rem;}

.metric{background:#111;color:#fff;border-radius:14px;padding:16px 10px;text-align:center;border-bottom:6px solid #D7141A;}
.metric .v{display:block;font-size:1.9rem;font-weight:800;color:#fff;}
.metric .k{display:block;font-size:.82rem;letter-spacing:.08em;color:#d9d9d9;margin-top:2px;}

.stTabs [data-baseweb="tab-list"]{gap:6px;border-bottom:3px solid #111;}
.stTabs [data-baseweb="tab"]{font-weight:800;padding:10px 20px;color:#111;border-radius:10px 10px 0 0;}
.stTabs [aria-selected="true"]{background:#111;color:#fff !important;}
.stTabs [data-baseweb="tab-highlight"]{background:#D7141A;height:4px;}

[data-testid="stSidebar"]{background:#111;}
[data-testid="stSidebar"] *{color:#fff;}
[data-testid="stSidebar"] hr{border-color:#D7141A;}
h2,h3{color:#111;font-weight:800;}
.small{font-size:.85rem;color:#444;}
.footer{margin-top:30px;padding:14px 18px;background:#111;color:#fff;border-radius:12px;border-top:5px solid #D7141A;
        text-align:center;font-size:.88rem;}
.footer *{color:#fff;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def H(s: str):
    """Render HTML (blank lines and indentation removed so Markdown never treats it as a code block)."""
    st.markdown("\n".join(l.strip() for l in s.splitlines() if l.strip()), unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the CNN model...")
def load_model(path: str, mtime: float):
    import tensorflow as tf                      # imported here so the page appears instantly
    return tf.keras.models.load_model(path)


@st.cache_data
def load_classes(mtime: float):
    with open(cfg.CLASS_JSON) as f:
        meta = json.load(f)
    return meta["class_folders"], meta["sign_names"], int(meta.get("img_size", cfg.IMG_SIZE))


@st.cache_data
def load_metrics(mtime: float):
    with open(cfg.METRICS_JSON) as f:
        return json.load(f)


def placeholder_sign(cls: str, size: int = 128) -> Image.Image:
    """Drawn red-ring sign with the class number (used when no reference photo is available)."""
    im = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(im)
    d.ellipse((4, 4, size - 4, size - 4), outline=RED, width=size // 9, fill="white")
    try:
        d.text((size / 2, size / 2), cls, fill=BLACK, anchor="mm", font_size=size // 3)   # Pillow >= 10.1
    except TypeError:
        d.text((size / 2 - 6, size / 2 - 6), cls, fill=BLACK)                              # older Pillow
    return im


def reference_image(cls: str) -> Image.Image:
    p = cfg.REF_DIR / f"{cls}.png"
    return Image.open(p) if p.exists() else placeholder_sign(cls)


def preprocess(img: Image.Image, size: int) -> np.ndarray:
    """Identical to training: RGB + resize to size x size (normalisation happens inside the model)."""
    return np.asarray(img.convert("RGB").resize((size, size), Image.Resampling.LANCZOS), dtype=np.uint8)


def open_image(file) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(file)).convert("RGB")


# ------------------------------------------------------------------------------------------------
# Header
# ------------------------------------------------------------------------------------------------
H("""
<div class="hero">
  <div class="noentry"></div>
  <div>
    <h1>TRAFFIC SIGN RECOGNITION</h1>
    <p>PRAICP-1002 &nbsp;|&nbsp; Convolutional Neural Network &nbsp;|&nbsp; 43 sign classes &nbsp;|&nbsp; Upload a sign, get the meaning in a blink</p>
  </div>
</div>
<div class="road"></div>
""")

if not cfg.CLASS_JSON.exists():
    st.error("The trained model files were not found in the `models/` folder.")
    st.markdown("Run **`python train.py`** (or `python train.py --quick` for a fast test) first, **or** copy "
                "`TrafficSign_CNN_Model.keras` and `TrafficSign_class_names.json` from your notebook run into `models/`.")
    st.stop()

class_names, sign_names, img_size = load_classes(cfg.CLASS_JSON.stat().st_mtime)
metrics = load_metrics(cfg.METRICS_JSON.stat().st_mtime) if cfg.METRICS_JSON.exists() else None
label_of = lambda i: f"{class_names[i]} · {sign_names[class_names[i]]}"

# ------------------------------------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------------------------------------
with st.sidebar:
    H('<div style="text-align:center"><div class="noentry" style="margin:0 auto 10px"></div></div>')
    st.markdown("## Control panel")
    threshold = st.slider("Confidence warning level (%)", 30, 95, 70, 5,
                          help="Below this confidence the app shows a caution notice.")
    st.markdown("---")
    st.markdown(f"**Signs known:** {len(class_names)}  \n**Input size:** {img_size} × {img_size}  \n"
                f"**Model file:** {cfg.MODEL_PATH.name}")
    if metrics:
        st.markdown(f"**Model:** {metrics['selected_model']}  \n**Test accuracy:** {metrics['test']['Accuracy'] * 100:.2f}%")
    st.markdown("---")
    st.markdown("**Tips**  \n• Use a close, centred photo of ONE sign.  \n• Blurry, dark or partly hidden signs "
                "lower the confidence.  \n• The model only knows the 43 signs of the dataset.")

tab_predict, tab_perf, tab_lib = st.tabs(["🛑  Predict", "📊  Model performance", "🚸  Sign library"])

# ================================================================================================
# TAB 1 - PREDICT
# ================================================================================================
with tab_predict:
    if not cfg.MODEL_PATH.exists():
        st.error(f"Model file not found: `{cfg.MODEL_PATH}`. Run `python train.py` first.")
    else:
        model = load_model(str(cfg.MODEL_PATH), cfg.MODEL_PATH.stat().st_mtime)
        left, right = st.columns([1, 1.2], gap="large")

        img, actual_idx = None, None
        with left:
            st.markdown("### 1 · Give the model a sign")
            source = st.radio("Image source", ["Upload image", "Use camera", "Test sample"],
                              horizontal=True, label_visibility="collapsed")
            if source == "Upload image":
                f = st.file_uploader("Drop a traffic sign photo here", type=["png", "jpg", "jpeg", "bmp", "ppm", "webp"])
                img = open_image(f) if f else None
            elif source == "Use camera":
                f = st.camera_input("Point the camera at a traffic sign")
                img = open_image(f) if f else None
            else:
                samples = sorted(cfg.SAMPLES_DIR.glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else 999)
                if not samples:
                    st.info("No sample images found. They are created by `train.py`.")
                else:
                    choice = st.selectbox("Pick an unseen test photo",
                                          samples, format_func=lambda p: f"{p.stem} · {sign_names.get(p.stem, p.stem)}")
                    img = open_image(choice)
                    actual_idx = class_names.index(choice.stem) if choice.stem in class_names else None

            if img is not None:
                c1, c2 = st.columns(2)
                c1.image(img, caption="Your image", width="stretch")
                c2.image(Image.fromarray(preprocess(img, img_size)).resize((256, 256), Image.Resampling.NEAREST),
                         caption=f"What the CNN sees ({img_size}×{img_size})", width="stretch")

        with right:
            st.markdown("### 2 · Prediction")
            if img is None:
                H("""
                <div class="card card-wait">
                  <div class="big">🚦</div>
                  <h3>Waiting for a traffic sign…</h3>
                  <p>Upload a photo, use your camera, or choose a test sample on the left.</p>
                </div>""")
            else:
                with st.spinner("Reading the sign..."):
                    probs = model.predict(preprocess(img, img_size)[None], verbose=0)[0]
                order = np.argsort(probs)[::-1][:5]
                best, conf = int(order[0]), float(probs[order[0]])

                H(f"""
                <div class="card card-red">
                  <div class="result">
                    <div class="sign" style="width:150px;height:150px;font-size:{3.4 if len(class_names[best]) < 3 else 2.6}rem">{html.escape(class_names[best])}</div>
                    <div>
                      <div class="label">PREDICTED SIGN</div>
                      <div class="name">{html.escape(sign_names[class_names[best]])}</div>
                      <span class="conf">Confidence&nbsp;<b>{conf * 100:.1f}%</b></span>
                    </div>
                  </div>""")
                if conf * 100 < threshold:
                    H("""<div class="caution"><div class="stripe"></div><div class="txt">
                    CAUTION - low confidence. The image may be blurred, dark, partly hidden, or not one of the 43 known signs.
                    Please double-check this result.</div></div>""")
                else:
                    H('<div class="ok">High confidence - the model is quite sure about this sign.</div>')
                if actual_idx is not None:
                    verdict = "CORRECT ✔" if actual_idx == best else f"WRONG ✘ (actual: {label_of(actual_idx)})"
                    H(f'<div class="ok">Test-sample check: <b>{html.escape(verdict)}</b></div>')
                H("</div>")

                st.markdown("#### Top-5 most likely signs")
                rows = "".join(
                    f'<div class="bar-row"><div class="bar-label">{html.escape(label_of(int(i)))}</div>'
                    f'<div class="bar-track"><div class="bar-fill {"top" if k == 0 else ""}" style="width:{probs[i] * 100:.1f}%"></div></div>'
                    f'<div class="bar-val">{probs[i] * 100:.1f}%</div></div>'
                    for k, i in enumerate(order))
                H(f'<div class="card">{rows}</div>')

                r1, r2 = st.columns([1, 3])
                r1.image(reference_image(class_names[best]), width=110)
                r2.markdown(f"**Reference photo** of sign **{label_of(best)}** from the dataset - "
                            "compare it with your image to confirm the result.")

# ================================================================================================
# TAB 2 - MODEL PERFORMANCE
# ================================================================================================
with tab_perf:
    if not metrics:
        st.info("No `metrics.json` found. It is created when you run `python train.py`. "
                "(The Predict tab works without it.)")
    else:
        t, ds = metrics["test"], metrics["dataset"]
        st.markdown(f"### Results of the selected model - **{metrics['selected_model']}** (unseen test set)")
        cols = st.columns(4)
        for col, (k, v) in zip(cols, [("ACCURACY", t["Accuracy"]), ("PRECISION (macro)", t["Precision (macro)"]),
                                      ("RECALL (macro)", t["Recall (macro)"]), ("F1-SCORE (macro)", t["F1-Score (macro)"])]):
            col.markdown(f'<div class="metric"><span class="v">{v * 100:.2f}%</span><span class="k">{k}</span></div>',
                         unsafe_allow_html=True)

        st.markdown("&nbsp;")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Sign classes", ds["classes"])
        d2.metric("Images used (train / val / test)", f"{ds['train']} / {ds['validation']} / {ds['test']}")
        d3.metric("Full dataset size", f"{ds['full_images']:,}")
        d4.metric("Misclassified test images", f"{ds['misclassified_test']} of {ds['test']}")

        H('<div class="road"></div>')
        st.markdown("### Baseline vs tuned model")
        comp = pd.DataFrame(metrics["comparison"]).T.rename_axis("Model / split")
        st.dataframe((comp * 100).round(2).astype(str) + " %", width="stretch")
        st.markdown("**Best hyperparameters:** " + ", ".join(f"`{k} = {v}`" for k, v in metrics["best_hyperparameters"].items()))

        st.markdown("### F1-score of every sign (weakest first)")
        f1 = pd.Series(metrics["per_class_f1"]).sort_values()
        st.bar_chart(f1, color=RED, horizontal=True, height=760)

        st.markdown("### Most confused pairs of signs")
        if metrics["most_confused"]:
            st.dataframe(pd.DataFrame(metrics["most_confused"]).rename(
                columns={"actual": "Actual sign", "predicted": "Predicted as", "count": "Images"}),
                width="stretch", hide_index=True)
        else:
            st.success("No misclassified test images.")

        a, b = st.columns(2)
        curves, cmat = cfg.REPORT_DIR / "training_curves.png", cfg.REPORT_DIR / "confusion_matrix.png"
        if curves.exists():
            a.markdown("### Training curves"); a.image(str(curves), width="stretch")
        if cmat.exists():
            b.markdown("### Confusion matrix"); b.image(str(cmat), width="stretch")

# ================================================================================================
# TAB 3 - SIGN LIBRARY
# ================================================================================================
with tab_lib:
    st.markdown("### The 43 signs the model knows")
    q = st.text_input("Search (name or number)", placeholder="e.g. speed, stop, 14").strip().lower()
    shown = [c for c in class_names if q in f"{c} {sign_names[c]}".lower()]
    if not shown:
        st.warning("No sign matches your search.")
    for start in range(0, len(shown), 6):
        cols = st.columns(6)
        for col, c in zip(cols, shown[start:start + 6]):
            with col:
                st.image(reference_image(c), width="stretch")
                st.markdown(f"<div style='text-align:center;font-weight:700;font-size:.85rem;color:#111'>"
                            f"<span style='color:#D7141A'>{html.escape(c)}</span> · {html.escape(sign_names[c])}</div>",
                            unsafe_allow_html=True)

H("""
<div class="footer">PRAICP-1002 &nbsp;·&nbsp; Traffic Sign Detection &amp; Classification &nbsp;·&nbsp; Team PTID-AIE-JUL-26-11142
&nbsp;·&nbsp; Submitted by Sameeksha<br>Educational project - not for real driving decisions.</div>
""")

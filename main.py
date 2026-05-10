import streamlit as st
import numpy as np
from PIL import Image
import time
import io

# ── TensorFlow imports ────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.models      import Sequential
from tensorflow.keras.layers      import (Input, Dense, Dropout,
                                           BatchNormalization, GlobalAveragePooling2D)
from tensorflow.keras.applications import VGG16

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DermAI — Skin Cancer Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS  — Medical-editorial dark theme
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root variables ── */
:root {
    --bg-base:      #060d1a;
    --bg-card:      #0d1c2e;
    --bg-card2:     #0a1628;
    --accent-teal:  #00d4c8;
    --accent-blue:  #3b82f6;
    --accent-green: #10b981;
    --accent-red:   #ef4444;
    --text-primary: #e8f0fe;
    --text-muted:   #64748b;
    --border:       rgba(0,212,200,0.12);
    --glow-teal:    0 0 40px rgba(0,212,200,0.15);
    --glow-green:   0 0 40px rgba(16,185,129,0.2);
    --glow-red:     0 0 40px rgba(239,68,68,0.2);
}

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary);
}
.stApp {
    background: var(--bg-base);
    background-image:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(0,212,200,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(59,130,246,0.06) 0%, transparent 60%);
}
.main .block-container { padding: 2rem 3rem 4rem; max-width: 1400px; }
section[data-testid="stSidebar"] { display: none; }

/* ── Hero ── */
.hero-wrap {
    text-align: center;
    padding: 3.5rem 1rem 2.5rem;
    position: relative;
}
.hero-badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent-teal);
    border: 1px solid rgba(0,212,200,0.3);
    border-radius: 20px;
    padding: 6px 18px;
    margin-bottom: 1.4rem;
    background: rgba(0,212,200,0.06);
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2.4rem, 5vw, 4rem);
    font-weight: 400;
    line-height: 1.15;
    color: #fff;
    margin: 0 0 1rem;
    letter-spacing: -0.5px;
}
.hero-title em {
    font-style: italic;
    color: var(--accent-teal);
}
.hero-sub {
    font-size: 1.05rem;
    color: var(--text-muted);
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.7;
    font-weight: 300;
}
.hero-line {
    width: 60px;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-teal), transparent);
    margin: 2rem auto 0;
}

/* ── Divider ── */
.section-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 2.5rem 0;
}

/* ── Upload Panel ── */
.upload-panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: var(--glow-teal);
    position: relative;
    overflow: hidden;
}
.upload-panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-teal), transparent);
}
.panel-label {
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent-teal);
    font-weight: 600;
    margin-bottom: 1.2rem;
}
.panel-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: #fff;
    margin: 0 0 0.4rem;
}
.panel-desc {
    font-size: 0.85rem;
    color: var(--text-muted);
    margin-bottom: 1.5rem;
    line-height: 1.6;
}

/* ── Image preview ── */
.img-preview-wrap {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid var(--border);
    margin-top: 1.2rem;
    background: var(--bg-card2);
}

/* ── Result Panel ── */
.result-panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2rem;
    height: 100%;
    position: relative;
    overflow: hidden;
}
.result-panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-blue), transparent);
}

/* ── Result verdict ── */
.verdict-benign {
    background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.35);
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
    box-shadow: var(--glow-green);
    margin: 1.2rem 0;
}
.verdict-malignant {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.35);
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
    box-shadow: var(--glow-red);
    margin: 1.2rem 0;
}
.verdict-icon { font-size: 2.8rem; margin-bottom: 0.4rem; }
.verdict-label-benign  { font-family: 'DM Serif Display', serif; font-size: 2rem; color: var(--accent-green); }
.verdict-label-malignant { font-family: 'DM Serif Display', serif; font-size: 2rem; color: var(--accent-red); }
.verdict-sub { font-size: 0.82rem; color: var(--text-muted); margin-top: 0.3rem; letter-spacing: 0.5px; }

/* ── Confidence bar ── */
.conf-wrap { margin: 1.4rem 0; }
.conf-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-bottom: 6px;
    font-weight: 500;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.conf-track {
    background: rgba(255,255,255,0.06);
    border-radius: 99px;
    height: 8px;
    overflow: hidden;
}
.conf-fill-green {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #059669, #10b981);
    transition: width 1s ease;
}
.conf-fill-red {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #dc2626, #ef4444);
    transition: width 1s ease;
}

/* ── Stats row ── */
.stats-row {
    display: flex;
    gap: 12px;
    margin-top: 1.2rem;
}
.stat-chip {
    flex: 1;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 12px;
    text-align: center;
}
.stat-val {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: #fff;
}
.stat-key {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 2px;
}

/* ── Medical alert ── */
.med-alert {
    background: rgba(245,158,11,0.07);
    border: 1px solid rgba(245,158,11,0.25);
    border-radius: 12px;
    padding: 14px 18px;
    font-size: 0.82rem;
    color: #fbbf24;
    margin-top: 1.2rem;
    line-height: 1.6;
    display: flex;
    gap: 10px;
    align-items: flex-start;
}

/* ── Placeholder state ── */
.placeholder-state {
    text-align: center;
    padding: 3rem 1rem;
    color: var(--text-muted);
}
.placeholder-icon { font-size: 3rem; opacity: 0.3; margin-bottom: 0.8rem; }
.placeholder-text { font-size: 0.9rem; line-height: 1.7; }

/* ── Info grid ── */
.info-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-top: 1rem;
}
.info-tile {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
}
.info-tile::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent-teal), transparent);
    opacity: 0.4;
}
.tile-icon { font-size: 1.6rem; margin-bottom: 0.7rem; }
.tile-head {
    font-family: 'DM Serif Display', serif;
    font-size: 1.05rem;
    color: #fff;
    margin-bottom: 0.5rem;
}
.tile-body {
    font-size: 0.82rem;
    color: var(--text-muted);
    line-height: 1.65;
}

/* ── Tech pills ── */
.tech-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 0.8rem;
}
.tech-pill {
    background: rgba(0,212,200,0.08);
    border: 1px solid rgba(0,212,200,0.2);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.78rem;
    color: var(--accent-teal);
    font-weight: 500;
}

/* ── Footer ── */
.footer-wrap {
    text-align: center;
    padding: 2.5rem 0 1rem;
    color: var(--text-muted);
    font-size: 0.8rem;
    letter-spacing: 0.3px;
}
.footer-brand {
    font-family: 'DM Serif Display', serif;
    font-size: 1rem;
    color: var(--accent-teal);
    margin-bottom: 0.3rem;
}

/* ── Streamlit widget overrides ── */
.stFileUploader > div {
    background: rgba(255,255,255,0.03) !important;
    border: 1.5px dashed rgba(0,212,200,0.35) !important;
    border-radius: 14px !important;
    padding: 1.5rem !important;
}
.stFileUploader label { color: var(--text-muted) !important; font-size: 0.88rem !important; }
div[data-testid="stImage"] img { border-radius: 12px; }
.stSpinner > div { border-color: var(--accent-teal) transparent transparent transparent !important; }
.stSuccess {
    background: rgba(16,185,129,0.1) !important;
    border: 1px solid rgba(16,185,129,0.3) !important;
    color: #6ee7b7 !important;
    border-radius: 10px !important;
}
.stProgress > div > div {
    background: linear-gradient(90deg, var(--accent-teal), var(--accent-blue)) !important;
    border-radius: 99px !important;
}
.stProgress { background: rgba(255,255,255,0.06) !important; border-radius: 99px !important; }
button[kind="primary"] {
    background: var(--accent-teal) !important;
    color: #000 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
# ── FIX: weights-only file needs architecture rebuilt first ──────────────────
# The notebook saved model1.save_weights('skin_cancer_model.weights.h5')
# so we rebuild the exact same architecture, then load the weights.

IMAGE_SIZE = 128

@st.cache_resource(show_spinner=False)
def load_skin_cancer_model():
    """
    Rebuild the Improved VGG16 architecture (matching the notebook's model1)
    and load the saved weights from 'skin_cancer_model.weights.h5'.
    """
    base = VGG16(weights=None, include_top=False,
                 input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    mdl = Sequential([
        Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        base,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(256, activation='relu'),
        Dropout(0.4),
        BatchNormalization(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')   # Benign=0 / Malignant=1
    ], name='Improved_VGG16')

    mdl.load_weights('skin_cancer_model.weights.h5')
    return mdl


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION  (fixed for softmax + 128×128)
# ══════════════════════════════════════════════════════════════════════════════
def predict_skin_cancer(pil_image, mdl):
    """
    Preprocess a PIL image and return (label, confidence_pct).
    Fixes:
      • Resize to 128×128 (matches training)
      • softmax output  →  argmax instead of threshold on [0][0]
    """
    img = pil_image.resize((IMAGE_SIZE, IMAGE_SIZE)).convert('RGB')
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)           # (1, 128, 128, 3)

    probs     = mdl.predict(arr, verbose=0)[0]  # [p_benign, p_malignant]
    class_idx = int(np.argmax(probs))
    label      = 'Benign' if class_idx == 0 else 'Malignant'
    confidence = float(probs[class_idx]) * 100

    return label, confidence, probs


# ══════════════════════════════════════════════════════════════════════════════
# HERO SECTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">🔬 AI-Powered Dermoscopy Analysis</div>
    <h1 class="hero-title">Skin Lesion<br><em>Intelligence</em></h1>
    <p class="hero-sub">
        Upload a dermoscopy image. Our deep learning model — trained on
        thousands of clinical images — instantly classifies it as
        <strong style="color:#10b981">Benign</strong> or
        <strong style="color:#ef4444">Malignant</strong>.
    </p>
    <div class="hero-line"></div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL  (with friendly error)
# ══════════════════════════════════════════════════════════════════════════════
try:
    model = load_skin_cancer_model()
    model_ready = True
except Exception as e:
    model_ready = False
    st.error(f"⚠️ Could not load model weights: `{e}`\n\n"
             "Make sure **skin_cancer_model.weights.h5** is in the same folder as main.py.")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT  — Upload  |  Results
# ══════════════════════════════════════════════════════════════════════════════
left, right = st.columns([1, 1], gap="large")

# ── LEFT — Upload & Preview ───────────────────────────────────────────────────
with left:
    st.markdown("""
    <div class="upload-panel">
        <div class="panel-label">Step 01</div>
        <div class="panel-title">Upload Image</div>
        <div class="panel-desc">
            Supported formats: JPG, JPEG, PNG<br>
            Best results with clear, close-up dermoscopy photos.
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_image = st.file_uploader(
        "Drop your image here or click to browse",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_image is not None:
        pil_img = Image.open(uploaded_image)
        st.markdown('<div class="img-preview-wrap">', unsafe_allow_html=True)
        st.image(pil_img, caption="Uploaded Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Image metadata
        w, h = pil_img.size
        st.markdown(f"""
        <div class="stats-row">
            <div class="stat-chip">
                <div class="stat-val">{w}px</div>
                <div class="stat-key">Width</div>
            </div>
            <div class="stat-chip">
                <div class="stat-val">{h}px</div>
                <div class="stat-key">Height</div>
            </div>
            <div class="stat-chip">
                <div class="stat-val">{pil_img.mode}</div>
                <div class="stat-key">Mode</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── RIGHT — Results ───────────────────────────────────────────────────────────
with right:
    st.markdown("""
    <div class="result-panel">
        <div class="panel-label">Step 02</div>
        <div class="panel-title">Analysis Results</div>
    </div>
    """, unsafe_allow_html=True)

    if uploaded_image is None:
        st.markdown("""
        <div class="placeholder-state">
            <div class="placeholder-icon">🔍</div>
            <div class="placeholder-text">
                No image uploaded yet.<br>
                Upload a skin lesion image on the left<br>to see the AI prediction here.
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif not model_ready:
        st.error("Model not loaded — see error above.")

    else:
        with st.spinner("Analysing image…"):
            time.sleep(1.2)
            label, confidence, probs = predict_skin_cancer(pil_img, model)

        st.success("✅ Analysis complete")

        # ── Verdict box ──────────────────────────────────────────────────────
        if label == "Benign":
            st.markdown(f"""
            <div class="verdict-benign">
                <div class="verdict-icon">✅</div>
                <div class="verdict-label-benign">Benign</div>
                <div class="verdict-sub">Non-Cancerous Lesion</div>
            </div>
            """, unsafe_allow_html=True)
            fill_class = "conf-fill-green"
        else:
            st.markdown(f"""
            <div class="verdict-malignant">
                <div class="verdict-icon">⚠️</div>
                <div class="verdict-label-malignant">Malignant</div>
                <div class="verdict-sub">Cancerous Lesion Detected</div>
            </div>
            """, unsafe_allow_html=True)
            fill_class = "conf-fill-red"

        # ── Confidence bar ────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="conf-wrap">
            <div class="conf-header">
                <span>Confidence Score</span>
                <span>{confidence:.1f}%</span>
            </div>
            <div class="conf-track">
                <div class="{fill_class}" style="width:{confidence:.1f}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Probability chips ─────────────────────────────────────────────────
        st.markdown(f"""
        <div class="stats-row">
            <div class="stat-chip">
                <div class="stat-val" style="color:#10b981">{probs[0]*100:.1f}%</div>
                <div class="stat-key">Benign</div>
            </div>
            <div class="stat-chip">
                <div class="stat-val" style="color:#ef4444">{probs[1]*100:.1f}%</div>
                <div class="stat-key">Malignant</div>
            </div>
            <div class="stat-chip">
                <div class="stat-val">128px</div>
                <div class="stat-key">Model Input</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Medical disclaimer ────────────────────────────────────────────────
        st.markdown("""
        <div class="med-alert">
            <span>⚕️</span>
            <span>This prediction is generated by an AI model and is
            <strong>not a medical diagnosis</strong>. Always consult a
            qualified dermatologist for clinical evaluation.</span>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# INFO SECTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; margin-bottom:1.5rem;">
    <div class="hero-badge">About This System</div>
</div>
<div class="info-grid">
    <div class="info-tile">
        <div class="tile-icon">🧠</div>
        <div class="tile-head">Deep Learning Model</div>
        <div class="tile-body">
            Built on VGG16 transfer learning with a custom
            classification head. Trained on thousands of
            dermoscopy images from the Melanoma Cancer Dataset.
        </div>
    </div>
    <div class="info-tile">
        <div class="tile-icon">⚡</div>
        <div class="tile-head">Real-time Inference</div>
        <div class="tile-body">
            Images are resized, normalised, and passed through
            the model in under a second. Both Benign and Malignant
            probabilities are returned.
        </div>
    </div>
    <div class="info-tile">
        <div class="tile-icon">🎯</div>
        <div class="tile-head">Binary Classification</div>
        <div class="tile-body">
            The model classifies lesions into two categories —
            <strong style="color:#10b981">Benign</strong> (non-cancerous) and
            <strong style="color:#ef4444">Malignant</strong> (cancerous) —
            with a softmax confidence score.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<br>', unsafe_allow_html=True)

st.markdown("""
<div class="info-tile" style="margin-top:0.5rem;">
    <div class="tile-head" style="margin-bottom:0.8rem;">🛠 Technology Stack</div>
    <div class="tech-row">
        <span class="tech-pill">Python 3.12</span>
        <span class="tech-pill">Streamlit</span>
        <span class="tech-pill">TensorFlow 2.x</span>
        <span class="tech-pill">Keras</span>
        <span class="tech-pill">VGG16</span>
        <span class="tech-pill">Transfer Learning</span>
        <span class="tech-pill">NumPy</span>
        <span class="tech-pill">Pillow</span>
        <span class="tech-pill">scikit-learn</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer-wrap">
    <div class="footer-brand">DermAI</div>
    <div>Final Year Project · CNN-based Skin Cancer Detection</div>
    <div style="margin-top:6px; opacity:0.5;">
        Built with Streamlit & TensorFlow · For Academic Use Only
    </div>
</div>
""", unsafe_allow_html=True)

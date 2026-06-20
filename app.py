import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgriScan · Crop & Weed Classifier",
    page_icon="🌿",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── App background ── */
.stApp {
    background-color: #0D1F14;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(52,211,99,0.12) 0%, transparent 70%);
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 760px;
}

/* ── Hero wordmark ── */
.hero {
    text-align: center;
    padding: 3rem 0 1.5rem;
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.22em;
    color: #34D363;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2.4rem, 6vw, 3.6rem);
    line-height: 1.08;
    color: #F0F7F1;
    margin: 0 0 0.5rem;
}
.hero-title em {
    font-style: italic;
    color: #34D363;
}
.hero-sub {
    font-size: 0.95rem;
    color: #7A9E84;
    max-width: 420px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Divider ── */
.rule {
    border: none;
    border-top: 1px solid rgba(52,211,99,0.18);
    margin: 2rem 0;
}

/* ── Upload zone override ── */
[data-testid="stFileUploadDropzone"] {
    background: rgba(52,211,99,0.04) !important;
    border: 1.5px dashed rgba(52,211,99,0.35) !important;
    border-radius: 14px !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: rgba(52,211,99,0.7) !important;
}
[data-testid="stFileUploadDropzone"] p,
[data-testid="stFileUploadDropzone"] span,
[data-testid="stFileUploadDropzone"] div {
    color: #7A9E84 !important;
    font-size: 0.9rem !important;
}

/* ── Upload label ── */
.upload-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    color: #34D363;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    display: block;
}

/* ── Result card ── */
.result-card {
    background: rgba(16, 36, 20, 0.85);
    border: 1px solid rgba(52, 211, 99, 0.22);
    border-radius: 16px;
    padding: 1.6rem 2rem;
    margin-top: 1.5rem;
    backdrop-filter: blur(8px);
}
.result-card.weed {
    border-color: rgba(251, 146, 60, 0.35);
    background: rgba(30, 18, 10, 0.85);
}
.result-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.2em;
    color: #7A9E84;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.result-label.weed { color: #b07040; }

.result-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    line-height: 1;
    color: #34D363;
    margin-bottom: 0.25rem;
}
.result-value.weed { color: #fb923c; }

.result-conf {
    font-size: 0.85rem;
    color: #7A9E84;
}
.result-conf span {
    font-family: 'JetBrains Mono', monospace;
    color: #34D363;
    font-weight: 600;
}
.result-conf.weed span { color: #fb923c; }

/* ── Confidence bar ── */
.conf-bar-track {
    background: rgba(255,255,255,0.07);
    border-radius: 999px;
    height: 6px;
    margin-top: 1rem;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #22c55e, #34D363);
    transition: width 0.8s ease;
}
.conf-bar-fill.weed {
    background: linear-gradient(90deg, #ea580c, #fb923c);
}

/* ── Warning card ── */
.warn-card {
    background: rgba(22,22,18,0.7);
    border: 1px solid rgba(255,255,100,0.18);
    border-radius: 12px;
    padding: 1.2rem 1.6rem;
    margin-top: 1.4rem;
    color: #c9c870;
    font-size: 0.9rem;
}

/* ── Image caption ── */
[data-testid="stImage"] > div > div {
    color: #4a6b52 !important;
    font-size: 0.78rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    text-align: center;
}

/* ── Spinner ── */
[data-testid="stSpinner"] > div {
    color: #34D363 !important;
}

/* ── Footer ── */
.footer {
    text-align: center;
    margin-top: 4rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    color: #2d4a35;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Agricultural AI · YOLOv8</div>
    <h1 class="hero-title">Agri<em>Scan</em></h1>
    <p class="hero-sub">
        Upload a plant image. The model tells you whether it's a
        crop worth protecting or a weed to eliminate — in seconds.
    </p>
</div>
<hr class="rule">
""", unsafe_allow_html=True)

# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return YOLO("agriscan_yolov8.pt")

with st.spinner("Loading model…"):
    model = load_model()

# ── Upload ────────────────────────────────────────────────────────────────────
st.markdown('<span class="upload-label">Plant image</span>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    label="plant_image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

# ── Inference ─────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="uploaded image", use_container_width=True)
    st.markdown('<hr class="rule">', unsafe_allow_html=True)

    with st.spinner("Analysing…"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            results = model.predict(source=tmp.name, conf=0.25, verbose=False)

    if len(results[0].boxes) > 0:
        box        = results[0].boxes[0]
        class_id   = int(box.cls[0])
        prediction = model.names[class_id]
        confidence = float(box.conf[0]) * 100

        is_weed    = "weed" in prediction.lower()
        card_cls   = "weed" if is_weed else ""
        emoji      = "⚠️" if is_weed else "✅"

        st.markdown(f"""
        <div class="result-card {card_cls}">
            <div class="result-label {card_cls}">Classification result</div>
            <div class="result-value {card_cls}">{emoji} {prediction.upper()}</div>
            <div class="result-conf {card_cls}">
                Confidence &nbsp;·&nbsp; <span>{confidence:.1f}%</span>
            </div>
            <div class="conf-bar-track">
                <div class="conf-bar-fill {card_cls}" style="width:{confidence:.1f}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        plotted = results[0].plot()
        st.image(plotted, caption="detection overlay", use_container_width=True)

    else:
        st.markdown("""
        <div class="warn-card">
            ⚡ No crop or weed detected in this image.
            Try a clearer photo with the plant centred and well-lit.
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">AGRISCAN &nbsp;·&nbsp; POWERED BY YOLOV8 &nbsp;·&nbsp; BUILT WITH STREAMLIT</div>
""", unsafe_allow_html=True)
"""
Digit AI - Handwritten Digit Recognition (Streamlit)
Model: K-Nearest Neighbors (k=3, distance weighting) trained on MNIST
Pages: Home, Draw, Upload, History, About
"""

import base64
import csv
import io
import os
from datetime import datetime

import joblib
import numpy as np
import streamlit as st
from PIL import Image, ImageFilter
from streamlit_drawable_canvas import st_canvas

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Digit AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Contact links shown at the bottom of the About page. Leave a value as "" to hide it.
CONTACT = {
    "GitHub": "https://github.com/sindhus56",
    "LinkedIn": "https://www.linkedin.com/in/sindhu-s56/",
    "Email": "sindhus5656@gmail.com",
}

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_knn_model.pkl")

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Caveat:wght@600;700&display=swap');

.stApp{
    background: linear-gradient(135deg,#f6f2ff 0%,#ebe4ff 50%,#ddd2fb 100%);
    background-attachment: fixed;
    font-family:'Poppins',sans-serif;
}
[data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="collapsedControl"],
[data-testid="stToolbar"], #MainMenu, footer{display:none !important;}
.block-container{max-width:1180px; padding:1.4rem 2rem 2rem;}

/* ---------- navbar ---------- */
.logo{color:#2b2160; font-size:1.6rem; font-weight:700; display:flex; align-items:center; gap:.6rem; padding-top:.2rem;}
.st-key-nav [data-testid="stBaseButton-secondary"]{
    background:transparent; color:#2b2160; border:0; font-weight:600; box-shadow:none;}
.st-key-nav [data-testid="stBaseButton-secondary"]:hover{background:rgba(91,75,219,.10); color:#2b2160;}
.st-key-nav [data-testid="stBaseButton-primary"]{
    background:#5b4bdb; color:#fff; border:0; border-radius:999px; font-weight:600;}

/* ---------- buttons ---------- */
.stButton button p{color:inherit !important; font-weight:inherit;}
[data-testid="stBaseButton-primary"]{
    background:linear-gradient(90deg,#4f46e5,#7c3aed); color:#fff; border:0;
    border-radius:12px; font-weight:700; padding:.55rem 1.4rem;}
[data-testid="stBaseButton-primary"]:hover{filter:brightness(1.1); color:#fff;}
[data-testid="stBaseButton-secondary"]{
    background:#fff; color:#2b2160; border:1px solid #d9d1f7; border-radius:12px; font-weight:600;}
.st-key-try [data-testid="stBaseButton-primary"]{
    font-size:1.05rem; padding:.7rem 1.8rem;}
.st-key-back [data-testid="stBaseButton-secondary"],
.st-key-clearhist [data-testid="stBaseButton-secondary"]{
    background:#fff; color:#2b2160; border:1px solid #cfc5f5;}

/* ---------- white cards ---------- */
.st-key-card, .st-key-card2, .st-key-card3{
    background:#fff; border-radius:22px; padding:1.4rem 1.6rem;
    box-shadow:0 18px 40px rgba(91,75,219,.18);}
.card-title{color:#2b2160; font-size:1.35rem; font-weight:700; margin:0;}
.card-sub{color:#5b5390; font-size:.92rem; margin:.2rem 0 .8rem;}

/* ---------- hero ---------- */
.hero h1{color:#2b2160; font-size:3.5rem; line-height:1.05; font-weight:800; margin:0 0 1rem;}
.hero h1 span{color:#6d3df0;}
.hero p{color:#4a4380; font-size:1.25rem; line-height:1.5; margin:0 0 1rem;}
.chk{color:#2b2160; font-size:1rem; display:flex; align-items:center; gap:.7rem; margin:.35rem 0;}
.chk b{background:#22a06b; color:#fff; width:22px; height:22px; border-radius:50%;
       display:inline-flex; align-items:center; justify-content:center; font-size:.8rem;}
.pv-wrap{display:flex; gap:1rem; align-items:center; justify-content:center; padding-top:.2rem; flex-wrap:nowrap;}
.pv-card{background:#fff; border-radius:22px; padding:16px; width:285px; transform:rotate(-4deg);
         box-shadow:0 18px 40px rgba(91,75,219,.22);}
.pv-tabs{display:flex; gap:8px; margin-bottom:12px; font-size:.9rem; font-weight:600;}
.pv-tabs span{padding:6px 14px; border-radius:8px; color:#5b5390;}
.pv-tabs span.on{background:#5b4bdb; color:#fff;}
.pv-pad{border:1px solid #e3dcf8; border-radius:14px; height:255px; display:flex;
        align-items:center; justify-content:center; font-family:'Caveat',cursive;
        font-size:12rem; color:#111; line-height:1;}
.pv-res{background:#fff; border-radius:22px; padding:20px 14px; width:185px; text-align:center;
        transform:rotate(3deg); box-shadow:0 18px 40px rgba(91,75,219,.22);}
.script{font-family:'Caveat',cursive; color:#6d3df0; font-size:2.6rem; text-align:right;
        transform:rotate(-5deg); margin-top:.2rem;}
.chips{display:flex; gap:2.4rem; color:#4a4380; font-size:.9rem; font-weight:500;
       margin-top:.6rem; padding-top:.8rem; border-top:1px solid rgba(91,75,219,.20);}

/* ---------- prediction result ---------- */
.res-title{color:#2b2160; font-weight:700; font-size:1.1rem;}
.res-digit{color:#5b4bdb; font-size:4.2rem; font-weight:800; line-height:1.1;}
.res-conf{color:#2b2160; font-size:.95rem;}
.res-conf b{font-size:1.3rem;}
.bar{height:9px; background:#e7e2fb; border-radius:6px; margin-top:.6rem; overflow:hidden;}
.bar div{height:9px; background:linear-gradient(90deg,#4f46e5,#8b5cf6); border-radius:6px;}
.ring{width:62px; height:62px; border-radius:50%; display:flex; align-items:center; justify-content:center;}
.ring div{width:44px; height:44px; background:#fff; border-radius:50%;}
.tip{background:rgba(255,255,255,.75); border:1px solid #ddd4f8; border-radius:16px; padding:.9rem 1.1rem;
     color:#2b2160; font-size:.9rem; margin-top:.9rem;}
.page-title{color:#2b2160; font-size:1.6rem; font-weight:700; margin:0;}
.page-sub{color:#5b5390; font-size:.95rem; margin:.1rem 0 1rem;}

/* ---------- history ---------- */
.hist{background:#fff; border-radius:16px; overflow:hidden; box-shadow:0 18px 40px rgba(91,75,219,.18);}
.hist table{width:100%; border-collapse:collapse; color:#2b2160; font-size:.95rem;}
.hist th{background:#f1edff; padding:.8rem; text-align:center; font-weight:700;}
.hist td{padding:.6rem; text-align:center; border-top:1px solid #eee9fb; font-weight:600;}
.hist img{width:40px; height:40px; border-radius:6px; image-rendering:pixelated; display:block; margin:auto;}
.empty{background:rgba(255,255,255,.7); color:#2b2160; border:1px solid #ddd4f8; border-radius:16px;
       padding:2rem; text-align:center;}

/* ---------- about ---------- */
.glass{background:#fff; border:1px solid #e3dcf8; border-radius:16px;
       padding:1.2rem 1.4rem; color:#2b2160; height:100%; box-shadow:0 12px 30px rgba(91,75,219,.12);}
.glass h4{margin:0 0 .8rem; font-size:1.1rem; font-weight:700; color:#2b2160;}
.row{display:flex; gap:1rem; font-size:.92rem; margin:.4rem 0; color:#2b2160;}
.row span:first-child{width:90px; color:#6a62a0;}
.about-p{color:#4a4380; line-height:1.8; font-size:1.12rem; margin:.4rem 0 1.1rem;}
.chip-ai{width:150px; height:150px; margin:auto; border-radius:26px; display:flex; align-items:center;
         justify-content:center; font-size:3rem; font-weight:800; color:#5b4bdb;
         border:2px solid #8b5cf6; background:rgba(139,92,246,.10);
         box-shadow:0 0 40px rgba(139,92,246,.30); transform:rotate(-8deg);}

/* ---------- prediction details ---------- */
.sub-h{color:#5b5390; font-size:.85rem; font-weight:600; margin:0 0 .5rem;}
.seen{width:112px; height:112px; background:#000; border-radius:10px; image-rendering:pixelated; display:block;}
.t3{display:flex; align-items:center; gap:.6rem; margin:.5rem 0; color:#2b2160; font-size:.95rem; font-weight:600;}
.t3 .bar{flex:1; margin:0;}
.t3 span:first-child{width:14px;}
.t3 span:last-child{width:52px; text-align:right;}
.nbrs{display:flex; gap:.9rem; flex-wrap:wrap;}
.nb{text-align:center; color:#2b2160; font-size:.8rem; font-weight:600;}
.nb img{width:84px; height:84px; border-radius:8px; image-rendering:pixelated; background:#000; display:block; margin-bottom:.25rem;}
.nb small{color:#6a62a0; font-weight:500;}
.how{background:rgba(255,255,255,.75); border:1px solid #ddd4f8; border-radius:16px; padding:1rem 1.1rem; margin-top:.9rem;}
.how .res-title{font-size:1rem; margin-bottom:.2rem;}
.step{display:flex; gap:.7rem; align-items:flex-start; color:#2b2160; font-size:.88rem; line-height:1.45; margin:.6rem 0 0;}
.step b{background:#5b4bdb; color:#fff; min-width:22px; height:22px; border-radius:50%;
        display:inline-flex; align-items:center; justify-content:center; font-size:.75rem;}
.note{color:#5b5390; font-size:.8rem; margin-top:.6rem;}
.st-key-dl button{background:#fff; color:#2b2160; border:1px solid #cfc5f5; border-radius:12px; font-weight:600;}
.st-key-dl button p{color:#2b2160 !important;}

/* ---------- model comparison ---------- */
.mrow{display:flex; align-items:center; gap:.8rem; margin:.55rem 0; color:#2b2160; font-size:.92rem;}
.mname{width:190px; font-weight:600;}
.mval{width:62px; text-align:right; font-weight:700;}
.mrow .bar{flex:1; margin:0; height:12px;}
.mrow .bar div{height:12px;}
.hist td.l, .hist th.l{text-align:left; padding-left:1.1rem;}
.hist tr.best td{background:#f4f0ff;}
.cm{width:100%; border-collapse:collapse; font-size:.78rem; text-align:center; margin-top:.4rem;}
.cm th{color:#5b5390; font-weight:600; padding:.3rem;}
.cm td{padding:.4rem .15rem; border:1px solid #fff; font-weight:600;}
.err{display:flex; justify-content:space-between; color:#2b2160; font-size:.92rem; padding:.4rem 0; border-bottom:1px solid #eee9fb;}
.err span{color:#6a62a0;}

/* ---------- insights ---------- */
.kpis{display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin-bottom:1rem;}
.kpi{background:#fff; border-radius:16px; padding:1rem 1.2rem; box-shadow:0 12px 30px rgba(91,75,219,.12);}
.kpi small{color:#6a62a0; font-size:.8rem; font-weight:600;}
.kpi b{display:block; color:#2b2160; font-size:1.7rem; line-height:1.3;}
.kpi span{color:#5b5390; font-size:.8rem;}
.mini{display:flex; align-items:center; gap:.5rem; min-width:150px;}
.mini .bar{flex:1; margin:0; height:8px;}
.mini .bar div{height:8px;}
.mini span{width:38px; text-align:right; font-size:.85rem;}
.avgs{display:grid; grid-template-columns:repeat(5,1fr); gap:.9rem; margin-top:.5rem;}
.av{text-align:center; color:#2b2160; font-size:.85rem; font-weight:600;}
.av img{width:100%; max-width:90px; aspect-ratio:1; border-radius:10px; image-rendering:pixelated; background:#000; display:block; margin:0 auto .25rem;}
.av small{color:#6a62a0; font-weight:500;}
.takeaway{color:#2b2160; font-size:.92rem; line-height:1.6; margin:.5rem 0 0;}

/* ---------- about: equal-height row ---------- */
.about-grid{display:grid; grid-template-columns:1.5fr 1.1fr .9fr; gap:1.2rem; align-items:stretch;}
.about-text{display:flex; flex-direction:column; justify-content:center; align-items:flex-start;}
.about-grid .glass{display:flex; flex-direction:column; height:auto;}
.glass-body{flex:1; display:flex; flex-direction:column; justify-content:space-evenly;}
.glass-body .row, .glass-body .chk{margin:.35rem 0;}
.glass-body .row span:first-child{flex:none; width:84px;}
.glass-body .row span:last-child{flex:1; white-space:nowrap;}
a.btn-about, a.btn-about:visited{display:inline-block; background:linear-gradient(90deg,#4f46e5,#7c3aed);
    color:#fff !important; padding:.7rem 1.5rem; border-radius:12px; font-weight:700; font-size:.95rem;
    text-decoration:none !important;}
a.btn-about:hover{filter:brightness(1.1);}
@media (max-width:800px){.about-grid{grid-template-columns:1fr;}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the model...")
def load_model():
    """Load final_knn_model.pkl, or train the same KNN on MNIST if it is missing."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    from sklearn.neighbors import KNeighborsClassifier
    from tensorflow.keras.datasets import mnist

    (x_train, y_train), _ = mnist.load_data()
    x_train = x_train.reshape(len(x_train), -1).astype("float32") / 255.0
    model = KNeighborsClassifier(n_neighbors=3, weights="distance", n_jobs=-1)
    model.fit(x_train, y_train)
    joblib.dump(model, MODEL_PATH)
    return model


# ----------------------------------------------------------------------------
# Image preprocessing (makes drawings/photos look like MNIST digits)
# ----------------------------------------------------------------------------
def preprocess(img: Image.Image):
    """Return a 28x28 float array (0-255, white digit on black) or None if blank.

    Steps: flatten transparency -> grayscale -> make digit white on black ->
    stretch contrast -> thicken thin strokes to MNIST style -> crop ->
    resize to 20 px -> centre by centre of mass in 28x28.
    """
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        base = Image.new("RGBA", img.size, (255, 255, 255, 255))
        base.alpha_composite(img)
        img = base
    arr = np.array(img.convert("L"), dtype=np.float32)

    if arr.mean() > 127:        # light background -> invert so the digit is white
        arr = 255.0 - arr

    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 30:            # (almost) blank image
        return None
    arr = (arr - lo) / (hi - lo) * 255.0     # stretch contrast to the full 0-255 range
    arr[arr < 50] = 0                         # remove faint noise / JPEG artefacts

    ink = arr > 128
    ys, xs = np.where(ink)
    if len(ys) == 0:
        return None
    size = max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)

    # Estimate stroke width (area / half the outline length) and thicken thin pen
    # lines so that, after shrinking to 20 px, the stroke is about as bold as MNIST.
    padded = np.pad(ink, 1)
    inner = (padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1]
             & padded[1:-1, :-2] & padded[1:-1, 2:])
    outline = int((ink & ~inner).sum())
    width = 2.0 * ink.sum() / max(outline, 1)
    target = size * 0.11
    if width < target:
        k = int(round(target - width)) + 1
        k = min(k + (1 - k % 2), 51)          # MaxFilter needs an odd size
        arr = np.array(
            Image.fromarray(arr.astype("uint8")).filter(ImageFilter.MaxFilter(k)),
            dtype=np.float32,
        )

    ys, xs = np.where(arr > 40)
    crop = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    h, w = crop.shape
    scale = 20.0 / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    small = np.array(
        Image.fromarray(crop.astype("uint8")).resize((nw, nh), Image.LANCZOS),
        dtype=np.float32,
    )
    if small.max() > 0:
        small *= 255.0 / small.max()

    total = small.sum()
    cy = (np.arange(nh)[:, None] * small).sum() / total
    cx = (np.arange(nw)[None, :] * small).sum() / total
    top, left = int(round(14 - cy)), int(round(14 - cx))

    out = np.zeros((28, 28), dtype=np.float32)
    y0, x0 = max(top, 0), max(left, 0)
    y1, x1 = min(top + nh, 28), min(left + nw, 28)
    out[y0:y1, x0:x1] = small[y0 - top:y1 - top, x0 - left:x1 - left]
    return out


def thumb_b64(arr: np.ndarray, size: int = 56) -> str:
    im = Image.fromarray(np.clip(arr, 0, 255).astype("uint8")).resize((size, size), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def get_neighbors(model, x, k: int = 3):
    """The k closest MNIST training images the KNN used for this prediction."""
    try:
        dist, idx = model.kneighbors(x, n_neighbors=k)
        fit_x = np.asarray(model._fit_X)
        y_enc = np.asarray(model._y)
        out = []
        for d, i in zip(dist[0], idx[0]):
            img = fit_x[i].reshape(28, 28).astype("float32")
            if img.max() <= 1.5:                      # stored as 0-1 -> back to 0-255
                img = img * 255.0
            out.append({
                "img": thumb_b64(img, 84),
                "label": int(model.classes_[y_enc[i]]),
                "dist": float(d),
            })
        return out
    except Exception:
        return []                                      # feature simply hidden if unavailable


def predict(img: Image.Image):
    """Return a result dict (digit, conf, top3, images, neighbors) or None."""
    arr = preprocess(img)
    if arr is None:
        return None
    model = load_model()
    x = (arr / 255.0).reshape(1, -1).astype("float32")
    proba = model.predict_proba(x)[0]
    order = np.argsort(proba)[::-1][:3]
    top3 = [(int(model.classes_[i]), float(proba[i] * 100)) for i in order]
    return {
        "digit": top3[0][0],
        "conf": top3[0][1],
        "top3": top3,
        "seen": thumb_b64(arr, 112),
        "thumb": thumb_b64(arr, 56),
        "neighbors": get_neighbors(model, x),
    }


def add_history(res):
    now = datetime.now()
    stamp = f"{now.strftime('%b')} {now.day}, {now.year} – {now.strftime('%I:%M %p')}"
    st.session_state.history.insert(
        0, {"img": res["thumb"], "digit": res["digit"], "conf": res["conf"], "time": stamp}
    )
    del st.session_state.history[50:]


def history_csv() -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["#", "Prediction", "Confidence (%)", "Date & Time"])
    for i, r in enumerate(st.session_state.history, start=1):
        writer.writerow([i, r["digit"], f'{r["conf"]:.1f}', r["time"]])
    return buf.getvalue()


# ----------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------
def result_card(res, ring=False):
    if res is None:
        digit, conf = "–", None
    else:
        digit, conf = res["digit"], res["conf"]
    conf_txt = "--" if conf is None else f"{conf:.1f}%"
    width = 0 if conf is None else conf
    ring_html = ""
    if ring:
        ring_html = (
            f'<div class="ring" style="background:conic-gradient(#6d4df2 {width}%, #e7e2fb 0)">'
            f"<div></div></div>"
        )
    st.markdown(
        f"""
        <div style="background:#fff;border-radius:20px;padding:1.2rem;text-align:center;
                    box-shadow:0 18px 40px rgba(91,75,219,.18);">
          <div class="res-title">Prediction Result</div>
          <div class="res-digit">{digit}</div>
          <div style="display:flex;align-items:center;justify-content:center;gap:1rem;">
            <div class="res-conf">Confidence<br><b>{conf_txt}</b></div>{ring_html}
          </div>
          <div class="bar"><div style="width:{width}%"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def details_section(res):
    """Top-3 predictions, what the model sees, and the nearest training images."""
    if not res:
        return
    st.write("")
    with st.container(key="card2"):
        st.markdown('<p class="card-title">🔍 Prediction details</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1.3, 1.9])
        with c1:
            st.markdown(
                '<p class="sub-h">What the model sees (28 × 28)</p>'
                f'<img class="seen" src="data:image/png;base64,{res["seen"]}">',
                unsafe_allow_html=True,
            )
        with c2:
            rows = ""
            for rank, (d, p) in enumerate(res["top3"]):
                fill = "" if rank == 0 else "background:#b9b0ec;"
                rows += (f'<div class="t3"><span>{d}</span><div class="bar">'
                         f'<div style="width:{max(p, 1):.1f}%;{fill}"></div></div>'
                         f'<span>{p:.1f}%</span></div>')
            st.markdown(f'<p class="sub-h">Top 3 predictions</p>{rows}', unsafe_allow_html=True)
        with c3:
            if res["neighbors"]:
                items = "".join(
                    f'<div class="nb"><img src="data:image/png;base64,{n["img"]}">'
                    f'Digit {n["label"]}<br><small>distance {n["dist"]:.2f}</small></div>'
                    for n in res["neighbors"]
                )
                st.markdown(
                    '<p class="sub-h">3 closest training images (the KNN neighbors)</p>'
                    f'<div class="nbrs">{items}</div>'
                    '<p class="note">KNN predicts by finding the most similar images '
                    'in the MNIST training set. Smaller distance means more similar.</p>',
                    unsafe_allow_html=True,
                )


def go(page):
    st.session_state.page = page


def navbar():
    page = st.session_state.page
    # which top-bar button is highlighted for the current page
    active = {"home": "home", "draw": "predict", "upload": "predict", "insights": "insights",
              "models": "models", "about": "about", "history": "history"}[page]
    items = [  # (button key, label, page it opens)
        ("home", "Home", "home"),
        ("predict", "Predict", "draw"),
        ("insights", "Insights", "insights"),
        ("models", "Models", "models"),
        ("about", "About", "about"),
        ("history", "History", "history"),
    ]
    with st.container(key="nav"):
        cols = st.columns([2.2, 1, 1, 1, 1, 1, 1])
        cols[0].markdown('<div class="logo">🧠 Digit AI</div>', unsafe_allow_html=True)
        for col, (key, label, target) in zip(cols[1:], items):
            col.button(label, key=f"nav_{key}", use_container_width=True,
                       type="primary" if active == key else "secondary",
                       on_click=go, args=(target,))


def back_button():
    current = st.session_state.page
    with st.container(key="back"):
        c0, _, c1, c2 = st.columns([2, 3, 1, 1])
        c0.button("← Back to Home", on_click=go, args=("home",))
        c1.button("✏️ Draw", key="tab_draw", use_container_width=True,
                  type="primary" if current == "draw" else "secondary",
                  on_click=go, args=("draw",))
        c2.button("☁️ Upload", key="tab_upload", use_container_width=True,
                  type="primary" if current == "upload" else "secondary",
                  on_click=go, args=("upload",))


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
def page_home():
    left, right = st.columns([1.05, 1])
    with left:
        st.markdown(
            """
            <div class="hero" style="padding-top:.6rem;">
              <h1>Handwritten<br>Digit <span>Recognition</span></h1>
              <p>Turn your handwritten digit<br>into a smart prediction!</p>
              <div class="chk"><b>✓</b>Draw or Upload</div>
              <div class="chk"><b>✓</b>Get Instant Result</div>
              <div class="chk"><b>✓</b>Powered by Machine Learning</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.container(key="try"):
            st.button("Try Now →", type="primary", on_click=go, args=("draw",))
    with right:
        st.markdown(
            """
            <div class="pv-wrap">
              <div class="pv-card">
                <div class="pv-tabs"><span class="on">Draw</span><span>Upload</span></div>
                <div class="pv-pad">9</div>
              </div>
              <div class="pv-res">
                <div class="res-title">Prediction Result</div>
                <div class="res-digit">9</div>
                <div class="res-conf">Confidence<br><b>99.3%</b></div>
                <div class="bar"><div style="width:99.3%"></div></div>
              </div>
            </div>
            <div class="script">Let's Recognize!</div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div class="chips"><span>🏆 Model: KNN</span><span>🗄️ Dataset: MNIST</span>'
        "<span>🎯 Accuracy: 97%</span></div>",
        unsafe_allow_html=True,
    )


def page_draw():
    back_button()
    st.write("")
    left, right = st.columns([1.7, 1])
    with left:
        with st.container(key="card"):
            st.markdown(
                '<p class="card-title">✏️ Draw a Digit</p>'
                '<p class="card-sub">Use your mouse to draw a digit (0-9) in the box below:</p>',
                unsafe_allow_html=True,
            )
            canvas_args = dict(
                fill_color="rgba(0,0,0,0)",
                stroke_width=16,
                stroke_color="#000000",
                background_color="#FFFFFF",
                height=300,
                width=460,
                drawing_mode="freedraw",
                update_streamlit=True,
                key=f"canvas_{st.session_state.canvas_n}",
            )
            try:
                # newer versions of streamlit-drawable-canvas need this flag
                canvas = st_canvas(return_image_data=True, **canvas_args)
            except TypeError:
                # older versions (0.9.x) do not know the flag
                canvas = st_canvas(**canvas_args)
            b1, b2, _ = st.columns([1, 1, 2])
            clear = b1.button("🗑 Clear", use_container_width=True)
            do_predict = b2.button("▶ Predict", type="primary", use_container_width=True)

    if clear:
        st.session_state.canvas_n += 1
        st.session_state.draw_result = None
        st.rerun()

    if do_predict:
        out = None
        if canvas.image_data is not None:
            out = predict(Image.fromarray(canvas.image_data.astype("uint8")))
        if out is None:
            st.session_state.draw_result = None
            with left:
                st.warning("Draw a digit first, then press Predict.")
        else:
            st.session_state.draw_result = out
            add_history(out)

    with right:
        result_card(st.session_state.draw_result, ring=True)
        st.markdown(
            '<div class="tip">💡 <b>Tip:</b> Draw clearly for better accuracy! '
            "Use one thick stroke and fill most of the box.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="how"><div class="res-title">How it works</div>
              <div class="step"><b>1</b>Your drawing is cleaned and resized to 28 × 28.</div>
              <div class="step"><b>2</b>KNN finds the 3 most similar MNIST images.</div>
              <div class="step"><b>3</b>Those neighbors vote, and the winner is shown.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    details_section(st.session_state.draw_result)


def page_upload():
    back_button()
    st.write("")
    left, right = st.columns([1.6, 1])
    with left:
        with st.container(key="card"):
            st.markdown(
                '<p class="card-title">☁️ Upload an Image</p>'
                '<p class="card-sub">Upload a clear image of a handwritten digit (0-9).</p>',
                unsafe_allow_html=True,
            )
            file = st.file_uploader(
                "Drag & drop your image here",
                type=["png", "jpg", "jpeg"],
                label_visibility="collapsed",
            )
            st.caption("Supported formats: PNG, JPG, JPEG")

    shown_img = None
    if file is None:
        st.session_state.upload_result = None
        st.session_state.last_upload = None
    else:
        shown_img = Image.open(file)
        file_key = f"{file.name}-{file.size}"
        if st.session_state.last_upload != file_key:
            out = predict(shown_img)
            st.session_state.last_upload = file_key
            if out is None:
                st.session_state.upload_result = None
                with left:
                    st.warning("No digit found in that image. Try a clearer one.")
            else:
                st.session_state.upload_result = out
                add_history(out)

    with right:
        result_card(st.session_state.upload_result)
        if shown_img is not None:
            st.write("")
            st.image(shown_img, caption="Uploaded Image", width=150)
    details_section(st.session_state.upload_result)


def page_history():
    c1, c_dl, c2 = st.columns([3, 1, 1])
    c1.markdown(
        '<p class="page-title">🕘 Prediction History</p>'
        '<p class="page-sub">Your recent predictions in this session</p>',
        unsafe_allow_html=True,
    )
    with c_dl:
        with st.container(key="dl"):
            st.download_button(
                "⬇ Download CSV",
                data=history_csv(),
                file_name="digit_predictions.csv",
                mime="text/csv",
                use_container_width=True,
                disabled=not st.session_state.history,
            )
    with c2:
        with st.container(key="clearhist"):
            if st.button("🗑 Clear History", use_container_width=True):
                st.session_state.history = []
                st.rerun()

    rows = st.session_state.history
    if not rows:
        st.markdown(
            '<div class="empty">No predictions yet. Draw or upload a digit to get started.</div>',
            unsafe_allow_html=True,
        )
        return

    body = "".join(
        f"<tr><td>{i}</td>"
        f'<td><img src="data:image/png;base64,{r["img"]}"></td>'
        f'<td>{r["digit"]}</td><td>{r["conf"]:.1f}%</td><td>{r["time"]}</td></tr>'
        for i, r in enumerate(rows, start=1)
    )
    st.markdown(
        f"""
        <div class="hist"><table>
          <tr><th>#</th><th>Image</th><th>Prediction</th><th>Confidence</th><th>Date &amp; Time</th></tr>
          {body}
        </table></div>
        """,
        unsafe_allow_html=True,
    )


# Results copied from the training notebook (validation set: 10,002 images)
MODEL_RESULTS = [
    ("K-Nearest Neighbors", 0.972006, 0.972200, 0.972006, 0.971971),
    ("Random Forest", 0.966907, 0.966959, 0.966907, 0.966898),
    ("Logistic Regression", 0.921216, 0.921161, 0.921216, 0.921129),
    ("Support Vector Machine", 0.913617, 0.913389, 0.913617, 0.913405),
]
K_RESULTS = [(1, 0.969506), (3, 0.972006), (5, 0.969006), (7, 0.968206)]
# Final KNN on the 10,000 test images: rows = actual digit, columns = predicted digit
FINAL_CM = [
    [973, 1, 1, 0, 0, 0, 3, 1, 1, 0],
    [0, 1132, 2, 0, 0, 0, 1, 0, 0, 0],
    [10, 6, 995, 2, 0, 0, 1, 14, 3, 1],
    [0, 0, 3, 975, 1, 15, 1, 8, 5, 2],
    [0, 7, 0, 0, 942, 0, 5, 2, 1, 25],
    [5, 1, 0, 11, 2, 858, 5, 1, 3, 6],
    [5, 3, 0, 0, 4, 3, 943, 0, 0, 0],
    [0, 22, 4, 0, 2, 0, 0, 990, 0, 10],
    [7, 0, 2, 12, 6, 12, 3, 3, 922, 7],
    [3, 4, 2, 8, 8, 3, 1, 8, 2, 970],
]
TOP_ERRORS = [(4, 9, 25), (7, 1, 22), (3, 5, 15), (2, 7, 14), (8, 3, 12)]


def confusion_html():
    peak = max(max(r) for r in FINAL_CM)
    head = "".join(f"<th>{j}</th>" for j in range(10))
    body = ""
    for i, row in enumerate(FINAL_CM):
        cells = ""
        for v in row:
            a = (v / peak) ** 0.5 if v else 0
            color = "#fff" if a > 0.55 else "#2b2160"
            cells += f'<td style="background:rgba(91,75,219,{a:.2f});color:{color}">{v}</td>'
        body += f"<tr><th>{i}</th>{cells}</tr>"
    return f'<table class="cm"><tr><th></th>{head}</tr>{body}</table>'


def page_models():
    st.markdown(
        '<p class="page-title">📊 Model Comparison</p>'
        '<p class="page-sub">Four models were trained on MNIST and compared on 10,002 '
        "validation images. KNN scored highest and became the final model.</p>",
        unsafe_allow_html=True,
    )

    bars, rows = "", ""
    for i, (name, acc, prec, rec, f1) in enumerate(MODEL_RESULTS):
        best = i == 0
        width = (acc - 0.85) / 0.15 * 100            # bars start at 85% so gaps are visible
        fill = "" if best else "background:#b9b0ec;"
        bars += (f'<div class="mrow"><span class="mname">{name}</span>'
                 f'<div class="bar"><div style="width:{width:.1f}%;{fill}"></div></div>'
                 f'<span class="mval">{acc * 100:.2f}%</span></div>')
        rows += (f'<tr class="{"best" if best else ""}"><td class="l">{name}{" ⭐" if best else ""}</td>'
                 f"<td>{acc * 100:.2f}%</td><td>{prec * 100:.2f}%</td>"
                 f"<td>{rec * 100:.2f}%</td><td>{f1 * 100:.2f}%</td></tr>")

    with st.container(key="card"):
        st.markdown(
            '<p class="card-title">Validation accuracy</p>'
            f'{bars}<p class="note">Bars start at 85% to make the differences easier to see.</p>',
            unsafe_allow_html=True,
        )
    st.write("")
    st.markdown(
        f"""<div class="hist"><table>
          <tr><th class="l">Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1 score</th></tr>
          {rows}</table></div>""",
        unsafe_allow_html=True,
    )
    st.write("")

    left, right = st.columns([1.5, 1])
    with left:
        with st.container(key="card2"):
            st.markdown(
                '<p class="card-title">Final KNN confusion matrix</p>'
                '<p class="card-sub">10,000 test images · 9,700 correct (97.00%) · '
                "rows = actual digit, columns = predicted digit</p>"
                + confusion_html(),
                unsafe_allow_html=True,
            )
    with right:
        errs = "".join(
            f'<div class="err"><b>{a} → {b}</b><span>{n} errors</span></div>'
            for a, b, n in TOP_ERRORS
        )
        ks = "".join(
            f'<div class="err"><b>k = {k}{" ⭐" if k == 3 else ""}</b><span>{acc * 100:.2f}%</span></div>'
            for k, acc in K_RESULTS
        )
        with st.container(key="card3"):
            st.markdown(
                '<p class="card-title">Most common mistakes</p>'
                '<p class="card-sub">Actual → predicted (test set)</p>'
                f"{errs}"
                '<p class="card-title" style="margin-top:1.2rem;">Why k = 3?</p>'
                '<p class="card-sub">Validation accuracy by number of neighbors</p>'
                f"{ks}",
                unsafe_allow_html=True,
            )


# Final KNN, per digit, on the 10,000 test images (from the notebook)
PER_DIGIT = [  # digit, precision, recall, f1, images
    (0, 0.97, 0.99, 0.98, 980), (1, 0.96, 1.00, 0.98, 1135), (2, 0.99, 0.96, 0.98, 1032),
    (3, 0.97, 0.97, 0.97, 1010), (4, 0.98, 0.96, 0.97, 982), (5, 0.96, 0.96, 0.96, 892),
    (6, 0.98, 0.98, 0.98, 958), (7, 0.96, 0.96, 0.96, 1028), (8, 0.98, 0.95, 0.96, 974),
    (9, 0.95, 0.96, 0.96, 1009),
]
ERRORS_10 = [(4, 9, 25), (7, 1, 22), (3, 5, 15), (2, 7, 14), (8, 3, 12),
             (8, 5, 12), (5, 3, 11), (2, 0, 10), (7, 9, 10), (9, 7, 8)]


@st.cache_resource(show_spinner=False)
def training_stats():
    """Average image and image count of every digit, read from the model's training data."""
    try:
        model = load_model()
        X = np.asarray(model._fit_X, dtype="float32")
        y = np.asarray(model.classes_)[np.asarray(model._y)]
        scale = 255.0 if X.max() <= 1.5 else 1.0
        avgs = [thumb_b64(X[y == d].mean(axis=0).reshape(28, 28) * scale, 90) for d in range(10)]
        counts = [int((y == d).sum()) for d in range(10)]
        return avgs, counts
    except Exception:
        return None


def mini_bar(value, low=0.90):
    width = max((value - low) / (1 - low) * 100, 2)      # bars start at 90% so gaps are visible
    return (f'<div class="mini"><div class="bar"><div style="width:{width:.0f}%"></div></div>'
            f"<span>{value * 100:.0f}%</span></div>")


def page_insights():
    st.markdown(
        '<p class="page-title">💡 Insights</p>'
        '<p class="page-sub">What the final KNN model does well, where it struggles, '
        "and what it learned from. Numbers come from the 10,000 MNIST test images.</p>",
        unsafe_allow_html=True,
    )

    best_rec = max(PER_DIGIT, key=lambda r: r[2])
    low_rec = min(PER_DIGIT, key=lambda r: r[2])
    low_pre = min(PER_DIGIT, key=lambda r: r[1])
    total_err = 300
    top10 = sum(n for _, _, n in ERRORS_10)
    st.markdown(
        f"""<div class="kpis">
          <div class="kpi"><small>Test accuracy</small><b>97.00%</b><span>9,700 of 10,000 correct</span></div>
          <div class="kpi"><small>Easiest digit</small><b>Digit {best_rec[0]}</b><span>recall {best_rec[2] * 100:.0f}%, almost never missed</span></div>
          <div class="kpi"><small>Hardest to find</small><b>Digit {low_rec[0]}</b><span>recall {low_rec[2] * 100:.0f}%, most often mistaken</span></div>
          <div class="kpi"><small>Least precise</small><b>Digit {low_pre[0]}</b><span>precision {low_pre[1] * 100:.0f}%, others get called 9</span></div>
        </div>""",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.5, 1])
    with left:
        rows = "".join(
            f"<tr><td><b>{d}</b></td><td>{mini_bar(p)}</td><td>{mini_bar(r)}</td>"
            f"<td>{f * 100:.0f}%</td><td>{n:,}</td></tr>"
            for d, p, r, f, n in PER_DIGIT
        )
        st.markdown(
            f"""<div class="hist"><table>
              <tr><th>Digit</th><th>Precision</th><th>Recall</th><th>F1</th><th>Test images</th></tr>
              {rows}</table></div>
              <p class="note" style="color:#5b5390">Bars start at 90%. Precision: when the model says this digit, how often it is right.
              Recall: how many real examples of the digit it finds.</p>""",
            unsafe_allow_html=True,
        )
    with right:
        errs = "".join(
            f'<div class="err"><b>{a} → {b}</b><span>{n} errors</span></div>'
            for a, b, n in ERRORS_10
        )
        with st.container(key="card"):
            st.markdown(
                '<p class="card-title">Top 10 mistakes</p>'
                '<p class="card-sub">Actual digit → what the model predicted</p>'
                f"{errs}"
                f'<p class="takeaway">These 10 pairs cause {top10} of the {total_err} errors '
                f"({top10 / total_err * 100:.0f}%). Most are look-alike shapes, "
                "such as 4 and 9, 7 and 1, or 3 and 5.</p>",
                unsafe_allow_html=True,
            )

    st.write("")
    stats = training_stats()
    if stats:
        avgs, counts = stats
        items = "".join(
            f'<div class="av"><img src="data:image/png;base64,{img}">Digit {d}'
            f"<br><small>{n:,} images</small></div>"
            for d, (img, n) in enumerate(zip(avgs, counts))
        )
        with st.container(key="card2"):
            st.markdown(
                '<p class="card-title">What the model learned from</p>'
                f'<p class="card-sub">The average of all {sum(counts):,} training images of each digit. '
                "KNN compares every new drawing with images like these.</p>"
                f'<div class="avgs">{items}</div>',
                unsafe_allow_html=True,
            )


def page_about():
    st.write("")
    st.markdown(
        """
        <div class="about-grid">
          <div class="about-text">
            <p class="page-title" style="font-size:2.1rem;">ℹ️ About This Project</p>
            <p class="about-p">This project uses a K-Nearest Neighbors (KNN) machine learning model
            to recognize handwritten digits from the MNIST dataset. It can accurately predict
            digits (0-9) with an impressive 97% accuracy. Draw a digit or upload an image and the
            app shows the prediction with a confidence score, the top 3 guesses, and the
            training images that influenced the result.</p>
            <a class="btn-about" href="https://scikit-learn.org/stable/modules/neighbors.html"
               target="_blank" rel="noopener">Learn More →</a>
          </div>
          <div class="glass"><h4>Project Details</h4>
            <div class="glass-body">
              <div class="row"><span>Dataset</span><span>: MNIST (70,000 images)</span></div>
              <div class="row"><span>Algorithm</span><span>: K-Nearest Neighbors (KNN)</span></div>
              <div class="row"><span>Image Size</span><span>: 28 × 28 pixels</span></div>
              <div class="row"><span>Accuracy</span><span>: 97%</span></div>
            </div>
          </div>
          <div class="glass"><h4>Key Features</h4>
            <div class="glass-body">
              <div class="chk"><b>✓</b>Draw Digits</div>
              <div class="chk"><b>✓</b>Upload Images</div>
              <div class="chk"><b>✓</b>Instant Prediction</div>
              <div class="chk"><b>✓</b>Prediction History</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    with st.container(key="card"):
        st.markdown(
            '<p class="card-title">📬 Contact</p>'
            '<p class="card-sub">Interested in this project, or want to work together? Let us connect.</p>',
            unsafe_allow_html=True,
        )
        links = []
        for label, value in CONTACT.items():
            if value:
                if label == "Email" and not value.startswith("mailto:"):
                    value = "mailto:" + value
                links.append((label, value))
        if links:
            cols = st.columns([1] * len(links) + [max(1, 5 - len(links))])
            for col, (label, url) in zip(cols, links):
                col.link_button(label, url, use_container_width=True)

    st.write("")
    st.markdown('<div class="chip-ai">AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align:center;color:#5b5390;margin-top:1.5rem;">Developed By Sindhu S</p>',
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
defaults = {
    "page": "home",
    "history": [],
    "canvas_n": 0,
    "draw_result": None,
    "upload_result": None,
    "last_upload": None,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

navbar()
st.write("")

{
    "home": page_home,
    "draw": page_draw,
    "upload": page_upload,
    "history": page_history,
    "insights": page_insights,
    "models": page_models,
    "about": page_about,
}[st.session_state.page]()
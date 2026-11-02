"""
Digit AI - Handwritten Digit Recognition (Streamlit)
Model: K-Nearest Neighbors (k=3, distance weighting) trained on MNIST
Pages: Home, Draw, Upload, History, About
"""

import base64
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
.st-key-card, .st-key-card2{
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
.pv-wrap{display:flex; gap:1rem; align-items:center; justify-content:center; padding-top:1rem;}
.pv-card{background:#fff; border-radius:20px; padding:14px; width:230px; transform:rotate(-4deg);
         box-shadow:0 18px 40px rgba(91,75,219,.22);}
.pv-tabs{display:flex; gap:8px; margin-bottom:10px; font-size:.8rem; font-weight:600;}
.pv-tabs span{padding:6px 14px; border-radius:8px; color:#5b5390;}
.pv-tabs span.on{background:#5b4bdb; color:#fff;}
.pv-pad{border:1px solid #e3dcf8; border-radius:12px; height:190px; display:flex;
        align-items:center; justify-content:center; font-family:'Caveat',cursive;
        font-size:9rem; color:#111; line-height:1;}
.pv-res{background:#fff; border-radius:20px; padding:16px; width:170px; text-align:center;
        transform:rotate(3deg); box-shadow:0 18px 40px rgba(91,75,219,.22);}
.script{font-family:'Caveat',cursive; color:#6d3df0; font-size:2.2rem; text-align:right;
        transform:rotate(-5deg); margin-top:.5rem;}
.chips{display:flex; gap:2.4rem; color:#4a4380; font-size:.9rem; font-weight:500;
       margin-top:1.6rem; padding-top:1rem; border-top:1px solid rgba(91,75,219,.20);}

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
.about-p{color:#4a4380; line-height:1.7; font-size:1rem;}
.chip-ai{width:150px; height:150px; margin:auto; border-radius:26px; display:flex; align-items:center;
         justify-content:center; font-size:3rem; font-weight:800; color:#5b4bdb;
         border:2px solid #8b5cf6; background:rgba(139,92,246,.10);
         box-shadow:0 0 40px rgba(139,92,246,.30); transform:rotate(-8deg);}
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


def predict(img: Image.Image):
    """Return (digit, confidence %, 28x28 array) or None."""
    arr = preprocess(img)
    if arr is None:
        return None
    model = load_model()
    x = (arr / 255.0).reshape(1, -1).astype("float32")
    proba = model.predict_proba(x)[0]
    digit = int(model.classes_[proba.argmax()])
    return digit, float(proba.max() * 100), arr


def thumb_b64(arr: np.ndarray) -> str:
    im = Image.fromarray(arr.astype("uint8")).resize((56, 56), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def add_history(arr, digit, conf):
    now = datetime.now()
    stamp = f"{now.strftime('%b')} {now.day}, {now.year} – {now.strftime('%I:%M %p')}"
    st.session_state.history.insert(
        0, {"img": thumb_b64(arr), "digit": digit, "conf": conf, "time": stamp}
    )
    del st.session_state.history[50:]


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


def go(page):
    st.session_state.page = page


def navbar():
    active = {"home": "home", "draw": "home", "upload": "home",
              "about": "about", "history": "history"}[st.session_state.page]
    with st.container(key="nav"):
        c0, c1, c2, c3 = st.columns([6, 1, 1, 1])
        c0.markdown('<div class="logo">🧠 Digit AI</div>', unsafe_allow_html=True)
        for col, (key, label) in zip((c1, c2, c3),
                                     (("home", "Home"), ("about", "About"), ("history", "History"))):
            col.button(label, key=f"nav_{key}", use_container_width=True,
                       type="primary" if active == key else "secondary",
                       on_click=go, args=(key,))


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
            <div class="hero" style="padding-top:2.2rem;">
              <h1>Handwritten<br>Digit <span>Recognition</span></h1>
              <p>Turn your handwritten digit<br>into a smart prediction!</p>
              <div class="chk"><b>✓</b>Draw or Upload</div>
              <div class="chk"><b>✓</b>Get Instant Result</div>
              <div class="chk"><b>✓</b>Powered by Machine Learning</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
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
            digit, conf, arr = out
            st.session_state.draw_result = {"digit": digit, "conf": conf}
            add_history(arr, digit, conf)

    with right:
        result_card(st.session_state.draw_result, ring=True)
        st.markdown(
            '<div class="tip">💡 <b>Tip:</b> Draw clearly for better accuracy! '
            "Use one thick stroke and fill most of the box.</div>",
            unsafe_allow_html=True,
        )


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
                digit, conf, arr = out
                st.session_state.upload_result = {"digit": digit, "conf": conf}
                add_history(arr, digit, conf)

    with right:
        result_card(st.session_state.upload_result)
        if shown_img is not None:
            st.write("")
            st.image(shown_img, caption="Uploaded Image", width=150)


def page_history():
    c1, c2 = st.columns([4, 1])
    c1.markdown(
        '<p class="page-title">🕘 Prediction History</p>'
        '<p class="page-sub">Your recent predictions in this session</p>',
        unsafe_allow_html=True,
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


def page_about():
    st.write("")
    a, b, c = st.columns([1.5, 1.1, 0.9])
    with a:
        st.markdown(
            """
            <p class="page-title">ℹ️ About This Project</p>
            <p class="about-p">This project uses a K-Nearest Neighbors (KNN) machine learning model
            to recognize handwritten digits from the MNIST dataset. It can accurately predict
            digits (0-9) with an impressive 97% accuracy.</p>
            """,
            unsafe_allow_html=True,
        )
        st.link_button("Learn More →", "https://scikit-learn.org/stable/modules/neighbors.html",
                       type="primary")
    with b:
        st.markdown(
            """
            <div class="glass"><h4>Project Details</h4>
              <div class="row"><span>Dataset</span><span>: MNIST (70,000 images)</span></div>
              <div class="row"><span>Algorithm</span><span>: K-Nearest Neighbors (KNN)</span></div>
              <div class="row"><span>Image Size</span><span>: 28 × 28 pixels</span></div>
              <div class="row"><span>Accuracy</span><span>: 97%</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c:
        st.markdown(
            """
            <div class="glass"><h4>Key Features</h4>
              <div class="chk"><b>✓</b>Draw Digits</div>
              <div class="chk"><b>✓</b>Upload Images</div>
              <div class="chk"><b>✓</b>Instant Prediction</div>
              <div class="chk"><b>✓</b>Prediction History</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
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
    "about": page_about,
}[st.session_state.page]()
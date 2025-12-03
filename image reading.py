import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io

# ---------- Page config ----------
st.set_page_config(
    page_title="VisionForge",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Image utilities ----------
def load_image_bgr(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr

def bgr_to_pil(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def to_grayscale(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def rotate_image(img_bgr, angle):
    if angle == 90:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img_bgr, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img_bgr

def mirror_image(img_bgr):
    return cv2.flip(img_bgr, 1)

def make_grid(img_bgr, rows=4, cols=4):
    h, w = img_bgr.shape[:2]
    cell_h = max(1, h // rows)
    cell_w = max(1, w // cols)
    grid = img_bgr.copy()
    for r in range(1, rows):
        y = r * cell_h
        cv2.line(grid, (0, y), (w, y), (34, 197, 94), 1)
    for c in range(1, cols):
        x = c * cell_w
        cv2.line(grid, (x, 0), (x, h), (34, 197, 94), 1)
    return grid

def detect_objects(img_bgr, min_area=500):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = img_bgr.copy()
    count = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > min_area:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(out, (x,y), (x+w, y+h), (34,197,94), 2)
            count += 1
    return out, count

def get_properties(img_bgr):
    h, w = img_bgr.shape[:2]
    ch = img_bgr.shape[2] if len(img_bgr.shape)==3 else 1
    return {
        "Width": w,
        "Height": h,
        "Channels": ch,
        "Shape": str(img_bgr.shape),
        "Dtype": str(img_bgr.dtype)
    }

# ---------- Modern UI Styling ----------
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap" rel="stylesheet">
    <style>
      :root {
        --bg: #0b1221;
        --card: #0f1724;
        --muted: #94a3b8;
        --text: #e6eef8;
        --accent: #60a5fa;
        --accent-2: #22c55e;
        --radius: 12px;
      }
      html, body, [class*="css"]  {
        background-color: var(--bg) !important;
        color: var(--text) !important;
        font-family: "Poppins", sans-serif;
      }
      .title-big { font-size: 38px; font-weight:800; }
      .subtitle { color: var(--muted); font-size:18px; margin-top:4px; }
      .tagline { color: var(--muted); font-size:14px; margin-bottom:18px; }
      .card { background: var(--card); border-radius: var(--radius); padding: 18px; box-shadow: 0 6px 24px rgba(2,6,23,0.6); }
      .kpi { display:inline-block; margin-right:12px; padding:8px 12px; border-radius:10px; background: rgba(255,255,255,0.04); }
      .small-muted { color:var(--muted); font-size:13px; }
      .footer { text-align:center; color:var(--muted); margin-top:20px; font-size:13px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("<div class='card'><h4>Upload Image</h4></div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg","jpeg","png"])

    st.markdown("---")
    st.markdown("<div class='card'><h4>Settings</h4>", unsafe_allow_html=True)
    show_original = st.checkbox("Always show original", value=False)
    min_area = st.slider("Minimum object area", 200, 5000, 500, step=100)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown(f"<div class='title-big'>VisionForge</div>", unsafe_allow_html=True)
st.markdown(f"<div class='subtitle'>Smart image tools for practical projects</div>", unsafe_allow_html=True)
st.markdown(f"<div class='tagline'>Practical image processing — easy and reliable</div>", unsafe_allow_html=True)
st.write("")

# ---------- Main Content ----------
if uploaded_file is None:
    st.markdown("<div class='card' style='padding:42px; text-align:center;'><h3>👋 Welcome</h3><p class='small-muted'>Upload an image from the sidebar to get started.</p></div>", unsafe_allow_html=True)

else:
    img_bgr = load_image_bgr(uploaded_file)

    props = get_properties(img_bgr)

    # KPIs row
    p1,p2,p3 = st.columns([1,1,1.5])
    with p1:
        st.markdown(f"<div class='kpi card'><b>{props['Width']}</b><br><span class='small-muted'>Width</span></div>", unsafe_allow_html=True)
    with p2:
        st.markdown(f"<div class='kpi card'><b>{props['Height']}</b><br><span class='small-muted'>Height</span></div>", unsafe_allow_html=True)
    with p3:
        st.markdown(f"<div class='kpi card'><b>{props['Channels']}</b><br><span class='small-muted'>Channels</span></div>", unsafe_allow_html=True)

    if show_original:
        st.markdown("<div class='card'><b>Original Image</b></div>", unsafe_allow_html=True)
        st.image(bgr_to_pil(img_bgr), use_column_width=True)

    # TABS
    tabs = st.tabs(["Show", "Grayscale", "Properties", "Rotate", "Mirror", "Grid", "Detect Objects", "Cuts"])

    with tabs[0]:
        st.markdown("<div class='card'><b>Original Image</b></div>", unsafe_allow_html=True)
        st.image(bgr_to_pil(img_bgr), use_column_width=True)

    with tabs[1]:
        st.markdown("<div class='card'><b>Black & White</b></div>", unsafe_allow_html=True)
        gray = to_grayscale(img_bgr)
        st.image(gray, use_column_width=True)

    with tabs[2]:
        st.markdown("<div class='card'><b>Image Properties</b></div>", unsafe_allow_html=True)
        for k,v in props.items():
            st.write(f"**{k}:** {v}")

    with tabs[3]:
        st.markdown("<div class='card'><b>Rotate</b></div>", unsafe_allow_html=True)
        angle = st.radio("Choose angle", [90,180,270], horizontal=True)
        rotated = rotate_image(img_bgr, angle)
        st.image(bgr_to_pil(rotated), use_column_width=True)

    with tabs[4]:
        st.markdown("<div class='card'><b>Mirror Image</b></div>", unsafe_allow_html=True)
        mirrored = mirror_image(img_bgr)
        st.image(bgr_to_pil(mirrored), use_column_width=True)

    with tabs[5]:
        st.markdown("<div class='card'><b>Grid (4×4)</b></div>", unsafe_allow_html=True)
        gimg = make_grid(img_bgr)
        st.image(bgr_to_pil(gimg), use_column_width=True)

    with tabs[6]:
        st.markdown("<div class='card'><b>Object Detection</b></div>", unsafe_allow_html=True)
        detected, count = detect_objects(img_bgr, min_area=min_area)
        st.write(f"Objects detected: **{count}**")
        st.image(bgr_to_pil(detected), use_column_width=True)

    with tabs[7]:
        st.markdown("<div class='card'><b>Cuts</b></div>", unsafe_allow_html=True)
        h,w = img_bgr.shape[:2]
        left = img_bgr[:, :w//2]
        right = img_bgr[:, w//2:]
        top = img_bgr[:h//2, :]
        bottom = img_bgr[h//2:, :]
        split = int(w*0.8)
        part80 = img_bgr[:, :split]
        part20 = img_bgr[:, split:]

        c1,c2 = st.columns(2)
        with c1:
            st.image(bgr_to_pil(left), caption="Left 50%", use_column_width=True)
            st.image(bgr_to_pil(top), caption="Top 50%", use_column_width=True)
        with c2:
            st.image(bgr_to_pil(right), caption="Right 50%", use_column_width=True)
            st.image(bgr_to_pil(bottom), caption="Bottom 50%", use_column_width=True)

        st.write("80 / 20 Cut")
        st.image(bgr_to_pil(part80), caption="80%", use_column_width=True)
        st.image(bgr_to_pil(part20), caption="20%", use_column_width=True)

# ---------- Footer ----------
st.markdown("<div class='footer'>Built with ❤️ — VisionForge</div>", unsafe_allow_html=True)

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# ---------- Utility functions (unchanged logic) ----------

def load_image_bgr(uploaded_file):
    """Load uploaded image as OpenCV BGR image."""
    image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr


def show_image_bgr(img_bgr, caption=None):
    """Display a BGR image in Streamlit as RGB."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption=caption, use_column_width=True)


def to_grayscale(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return gray


def get_properties(img_bgr):
    h, w = img_bgr.shape[:2]
    channels = img_bgr.shape[2] if len(img_bgr.shape) == 3 else 1
    props = {
        "Width": w,
        "Height": h,
        "Channels": channels,
        "Shape": img_bgr.shape,
        "Data type": str(img_bgr.dtype),
        "Total pixels": img_bgr.size
    }
    return props


def rotate_image(img_bgr, angle):
    if angle == 90:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img_bgr, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return img_bgr


def mirror_image(img_bgr):
    return cv2.flip(img_bgr, 1)  # horizontal flip


def make_grid(img_bgr, rows=4, cols=4):
    """Make a grid over the image (non-prime number of cells e.g., 4x4=16)."""
    h, w = img_bgr.shape[:2]
    cell_h = h // rows
    cell_w = w // cols

    grid_img = img_bgr.copy()
    # Horizontal lines
    for r in range(1, rows):
        y = r * cell_h
        cv2.line(grid_img, (0, y), (w, y), (0, 255, 0), 1)
    # Vertical lines
    for c in range(1, cols):
        x = c * cell_w
        cv2.line(grid_img, (x, 0), (x, h), (0, 255, 0), 1)

    return grid_img


def detect_objects(img_bgr, min_area=500):
    """Detect objects without deep learning using edges + contours."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    obj_img = img_bgr.copy()
    count = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > min_area:
            x, y, w_box, h_box = cv2.boundingRect(c)
            cv2.rectangle(obj_img, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            count += 1

    return obj_img, count


# ---------- App layout & style controls ----------

st.set_page_config(page_title="Basic Image Processing App", layout="wide")

# Sidebar style controls
with st.sidebar.expander("Appearance / Fonts", expanded=True):
    st.write("Customize UI appearance")
    font_choice = st.selectbox("Font family", ["Inter", "Poppins", "Roboto", "Georgia", "Courier New"], index=0)
    base_size = st.slider("Base font size (px)", 12, 20, 15)
    accent = st.color_picker("Accent color", "#ff4b4b")
    density = st.selectbox("Density", ["Comfortable", "Compact"], index=0)
    rounded = st.checkbox("Rounded cards & buttons", value=True)
    show_original_always = st.checkbox("Always show original", value=False)

# small helper: map family to Google font import name
google_font_map = {
    "Inter": "Inter",
    "Poppins": "Poppins",
    "Roboto": "Roboto",
    "Georgia": None,
    "Courier New": None,
}

# build CSS
font_import = ""
if google_font_map.get(font_choice):
    font_import = f"@import url('https://fonts.googleapis.com/css2?family={google_font_map[font_choice].replace(' ', '+')}:wght@300;400;600;800&display=swap');"

border_radius = "12px" if rounded else "4px"
padding_val = "8px" if density == "Compact" else "14px"
line_height = "1.35" if density == "Compact" else "1.6"

css = f"""
{font_import}
:root {{
  --accent: {accent};
  --bg: #0f1113;
  --card: #131416;
  --muted: #bfc3c9;
  --text: #e6e6e6;
  --radius: {border_radius};
  --pad: {padding_val};
  --base-size: {base_size}px;
  --line-h: {line_height};
}}

html, body, [class*="css"] {{
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: '{font_choice}', system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
  font-size: var(--base-size);
  line-height: var(--line-h);
}}

.stApp {{
  padding-top: 18px;
}}

.block-container {{
  padding-left: 18px;
  padding-right: 18px;
}}

h1, .stTitle {{
  font-weight: 800;
  letter-spacing: -0.5px;
  color: var(--text);
}}

h2 {{
  color: var(--text);
}}

.sidebar .sidebar-content {{
  background: linear-gradient(180deg, rgba(255,255,255,0.02), transparent);
}}

.stSidebar .stButton>button, .stButton>button {{
  border-radius: var(--radius);
}}

.card {{
  background: var(--card);
  padding: 16px;
  border-radius: var(--radius);
  box-shadow: 0 6px 20px rgba(0,0,0,0.45);
}}

.upload-area .upload-box {{
  border: 1px dashed rgba(255,255,255,0.06);
  background: linear-gradient(180deg, rgba(255,255,255,0.01), transparent);
  padding: 18px;
  border-radius: var(--radius);
}}

.hero {{
  padding: 18px 14px;
  border-radius: var(--radius);
  margin-bottom: 12px;
}}

.title-big {{
  font-size: calc(var(--base-size) * 2.6);
  font-weight: 800;
  margin: 0;
  color: var(--text);
}}

.subtitle {{
  font-size: calc(var(--base-size) * 1.1);
  color: var(--muted);
  margin-top: 6px;
  margin-bottom: 8px;
}}

.accent {{
  color: var(--accent);
}}

.small-muted {{
  color: var(--muted);
  font-size: calc(var(--base-size) * 0.92);
}}

.tab-card {{
  background: linear-gradient(180deg, rgba(255,255,255,0.01), transparent);
  padding: 12px;
  border-radius: calc(var(--radius) - 2px);
  margin-bottom: 12px;
}}

.streamlit-expanderHeader {{
  color: var(--muted);
}}
"""

st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# ---------- Streamlit App content (same functionality, updated presentation) ----------

st.markdown("""
<div class='hero card'>
  <div style='display:flex; justify-content:space-between; align-items:center; gap:12px;'>
    <div>
      <div class='title-big'>Pro Image Studio <span class='accent'>∘</span></div>
      <div class='subtitle'>Professional image processing & analysis tool</div>
    </div>
    <div style='text-align:right'>
      <div class='small-muted'>Practical 1 – Deep Learning (Basic Image Processing)</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Upload area in sidebar (styled)
with st.sidebar:
    st.markdown("<div class='upload-area card'><b>1. Upload Image</b></div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    st.markdown("<hr />", unsafe_allow_html=True)

# Main content area
if uploaded_file is None:
    st.markdown("<div class='card tab-card' style='text-align:center; padding:48px;'><h2>👋 Welcome to Pro Image Studio</h2><p class='small-muted'>Upload an image from the left sidebar to get started.</p></div>", unsafe_allow_html=True)
else:
    img_bgr = load_image_bgr(uploaded_file)

    # show original optionally
    if show_original_always:
        st.markdown("<div class='card tab-card'><b>Original</b></div>", unsafe_allow_html=True)
        show_image_bgr(img_bgr, caption="Original Image")

    tabs = st.tabs([
        "1) Show Image",
        "2) Color to Black & White",
        "3) Properties",
        "4) Rotate Image",
        "5) Mirror Image",
        "6) Make Small Grid",
        "7) Find Objects (No DL)",
        "8) Select All Options"
    ])

    with tabs[0]:
        st.markdown("<div class='card tab-card'><b>Original Image</b></div>", unsafe_allow_html=True)
        show_image_bgr(img_bgr, caption="Original Image")

    with tabs[1]:
        st.markdown("<div class='card tab-card'><b>Black & White (Grayscale)</b></div>", unsafe_allow_html=True)
        gray = to_grayscale(img_bgr)
        st.image(gray, caption="Grayscale Image", use_column_width=True, clamp=True)

    with tabs[2]:
        st.markdown("<div class='card tab-card'><b>Image Properties</b></div>", unsafe_allow_html=True)
        props = get_properties(img_bgr)
        for k, v in props.items():
            st.write(f"**{k}:** {v}")

    with tabs[3]:
        st.markdown("<div class='card tab-card'><b>Rotate the Image</b></div>", unsafe_allow_html=True)
        angle = st.radio("Select rotation angle:", [90, 180, 270], index=0, horizontal=True)
        rotated = rotate_image(img_bgr, angle)
        show_image_bgr(rotated, caption=f"Rotated {angle} degrees")

    with tabs[4]:
        st.markdown("<div class='card tab-card'><b>Mirror Image (Horizontal Flip)</b></div>", unsafe_allow_html=True)
        mirrored = mirror_image(img_bgr)
        show_image_bgr(mirrored, caption="Mirror Image")

    with tabs[5]:
        st.markdown("<div class='card tab-card'><b>Grid on the Image (4x4 = 16 cells)</b></div>", unsafe_allow_html=True)
        grid_img = make_grid(img_bgr, rows=4, cols=4)
        show_image_bgr(grid_img, caption="Image with 4x4 Grid")

    with tabs[6]:
        st.markdown("<div class='card tab-card'><b>Object Detection (Edge + Contours)</b></div>", unsafe_allow_html=True)
        obj_img, count = detect_objects(img_bgr)
        st.write(f"**Approximate number of detected objects:** {count}")
        show_image_bgr(obj_img, caption="Detected Objects (Bounding Boxes)")

    with tabs[7]:
        st.markdown("<div class='card tab-card'><b>All Operations Combined</b></div>", unsafe_allow_html=True)
        st.write("### Original")
        show_image_bgr(img_bgr)

        st.write("### Black & White")
        st.image(to_grayscale(img_bgr), use_column_width=True, clamp=True)

        st.write("### Properties")
        props = get_properties(img_bgr)
        for k, v in props.items():
            st.write(f"**{k}:** {v}")

        st.write("### Rotated 90°, 180°, 270°")
        col1, col2, col3 = st.columns(3)
        with col1:
            show_image_bgr(rotate_image(img_bgr, 90), caption="90°")
        with col2:
            show_image_bgr(rotate_image(img_bgr, 180), caption="180°")
        with col3:
            show_image_bgr(rotate_image(img_bgr, 270), caption="270°")

        st.write("### Mirror Image")
        show_image_bgr(mirror_image(img_bgr), caption="Mirror Image")

        st.write("### Grid (4x4)")
        show_image_bgr(make_grid(img_bgr, 4, 4), caption="4x4 Grid")

        st.write("### Object Detection")
        obj_img_all, count_all = detect_objects(img_bgr)
        st.write(f"**Objects detected:** {count_all}")
        show_image_bgr(obj_img_all, caption="Detected Objects")

# ---------- Footer small note ----------
st.markdown("<div style='padding:10px; color:var(--muted); text-align:center; font-size:12px;'>Built with Streamlit • Pro Image Studio look & feel</div>", unsafe_allow_html=True)

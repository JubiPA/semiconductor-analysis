import streamlit as st
import pandas as pd
import joblib
import os
import time

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wafer Defect Predictor",
    page_icon="🔬",
    layout="centered",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  — animations + styling
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Animated gradient banner ─────────────────────────────────────── */
.banner {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    background-size: 300% 300%;
    animation: gradientShift 6s ease infinite;
    border-radius: 16px;
    padding: 2.2rem 2rem 1.6rem;
    margin-bottom: 1.8rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
@keyframes gradientShift {
    0%   { background-position: 0%   50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0%   50%; }
}
.banner-title {
    font-size: 2.4rem; font-weight: 900; letter-spacing: 1px;
    background: linear-gradient(90deg, #00c6ff, #0072ff, #7f00ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: shimmer 3s linear infinite;
    background-size: 200%;
}
@keyframes shimmer {
    0%   { background-position: 0%   50%; }
    100% { background-position: 200% 50%; }
}
.banner-sub {
    color: #a8d8ea; font-size: 0.95rem; margin-top: 0.4rem; font-weight: 400;
}

/* ── Section headings ─────────────────────────────────────────────── */
.section-heading {
    font-size: 1.25rem; font-weight: 800;
    background: linear-gradient(90deg, #ff6a00, #ee0979);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    border-left: 5px solid #ff6a00;
    padding-left: 12px;
    margin: 1.4rem 0 0.8rem;
    letter-spacing: 0.3px;
}

/* ── Result badges ────────────────────────────────────────────────── */
.result-box {
    border-radius: 14px; padding: 1.4rem 1.8rem;
    text-align: center; font-size: 1.5rem; font-weight: 900;
    letter-spacing: 0.5px; margin: 1rem 0;
    animation: popIn 0.45s cubic-bezier(0.22,1,0.36,1);
}
@keyframes popIn {
    0%   { transform: scale(0.7); opacity: 0; }
    100% { transform: scale(1);   opacity: 1; }
}
.result-good   { background: linear-gradient(135deg,#d4edda,#a8f0c2); color:#145a32; border:2px solid #28a745; }
.result-defect { background: linear-gradient(135deg,#f8d7da,#f5a8ae); color:#7b1c24; border:2px solid #dc3545; }

/* ── Metric chips ─────────────────────────────────────────────────── */
.chip-row { display:flex; gap:14px; justify-content:center; margin:1rem 0; }
.chip {
    border-radius: 50px; padding: 10px 22px;
    font-weight: 700; font-size: 1rem; min-width: 140px;
    text-align: center;
    animation: fadeUp 0.5s ease both;
}
@keyframes fadeUp {
    from { transform:translateY(14px); opacity:0; }
    to   { transform:translateY(0);    opacity:1; }
}
.chip-good   { background:#d4edda; color:#145a32; border:1.5px solid #28a745; }
.chip-defect { background:#f8d7da; color:#7b1c24; border:1.5px solid #dc3545; }

/* ── Divider ──────────────────────────────────────────────────────── */
.fancy-divider {
    height: 3px;
    background: linear-gradient(90deg, #0072ff, #7f00ff, #00c6ff);
    border-radius: 99px;
    margin: 1.4rem 0;
}

/* ── Footer ───────────────────────────────────────────────────────── */
.footer {
    text-align:center; color:#888; font-size:0.8rem;
    margin-top: 2rem; padding-top: 1rem;
    border-top: 1px solid #eee;
}

/* Hide Streamlit default header ──────────────────────────────────── */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    files = ["best_model.pkl", "scaler.pkl", "label_encoder.pkl", "features.pkl"]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        return None, None, None, None
    return (
        joblib.load("best_model.pkl"),
        joblib.load("scaler.pkl"),
        joblib.load("label_encoder.pkl"),
        joblib.load("features.pkl"),
    )

model, scaler, le, features = load_artifacts()

# ─────────────────────────────────────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="banner">
    <div class="banner-title">🔬 Wafer Defect Predictor</div>
    <div class="banner-sub">
        Semiconductor Quality Control · Fabrication Monitoring<br>
        Enter 11 process parameters → Instant defect prediction
    </div>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error(
        "⚠️ **Model artifacts not found.**  \n"
        "Place `best_model.pkl`, `scaler.pkl`, `label_encoder.pkl`, and `features.pkl` "
        "in the same folder as `app.py`, then re-run."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# EXACT 11 FEATURES  (ranges calibrated to real dataset statistics)
#
# Dataset summary (n=4,219 wafers):
#   Chamber_Temperature    : min=55.7,   max=97.4,    mean=75.1
#   Gas_Flow_Rate          : min=5.3,    max=86.0,    mean=49.9
#   RF_Power               : min=127.6,  max=476.8,   mean=301.4
#   Etch_Depth             : min=157.9,  max=874.5,   mean=498.7
#   Rotation_Speed         : min=859.6,  max=2199.0,  mean=1504.7
#   Vacuum_Pressure        : min=0.346,  max=0.688,   mean=0.501
#   Stage_Alignment_Error  : min=-1.57,  max=4.54,    mean=2.00
#   Vibration_Level        : min=-0.005, max=0.029,   mean=0.010
#   UV_Exposure_Intensity  : min=69.0,   max=171.0,   mean=119.9
#   Particle_Count         : min=100,    max=999,     mean=556
# ─────────────────────────────────────────────────────────────────────────────
TOOL_TYPES = list(le.classes_)   # ['Deposition', 'Etching', 'Lithography']

st.markdown('<div class="section-heading">⚙️ Process Parameters</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    tool_type = st.selectbox(
        "**🛠 Tool Type**",
        TOOL_TYPES,
    )
    chamber_temp = st.number_input(
        "**🌡 Chamber Temperature (°C)**",
        min_value=50.0, max_value=100.0,
        value=75.0, step=0.5,
        help="Dataset range: 55.7 – 97.4 °C",
    )
    gas_flow = st.number_input(
        "**💨 Gas Flow Rate (sccm)**",
        min_value=5.0, max_value=90.0,
        value=50.0, step=0.5,
        help="Dataset range: 5.3 – 86.0 sccm",
    )
    rf_power = st.number_input(
        "**⚡ RF Power (W)**",
        min_value=100.0, max_value=500.0,
        value=300.0, step=1.0,
        help="Dataset range: 127.6 – 476.8 W",
    )
    etch_depth = st.number_input(
        "**🔩 Etch Depth (μm)**",
        min_value=100.0, max_value=900.0,
        value=500.0, step=1.0,
        help="Dataset range: 157.9 – 874.5 μm",
    )
    rotation_spd = st.number_input(
        "**🔄 Rotation Speed (rpm)**",
        min_value=800.0, max_value=2300.0,
        value=1500.0, step=1.0,
        help="Dataset range: 859.6 – 2199.0 rpm",
    )

with col2:
    vacuum_pres = st.number_input(
        "**🌀 Vacuum Pressure (Torr)**",
        min_value=0.300, max_value=0.750,
        value=0.500, step=0.001,
        format="%.3f",
        help="Dataset range: 0.346 – 0.688 Torr",
    )
    stage_err = st.number_input(
        "**📐 Stage Alignment Error (μm)**",
        min_value=-2.0, max_value=5.0,
        value=2.0, step=0.01,
        format="%.3f",
        help="Dataset range: -1.57 – 4.54 μm",
    )
    vibration = st.number_input(
        "**📳 Vibration Level (g)**",
        min_value=-0.010, max_value=0.035,
        value=0.010, step=0.001,
        format="%.4f",
        help="Dataset range: -0.005 – 0.029 g",
    )
    uv_intensity = st.number_input(
        "**🔆 UV Exposure Intensity (mJ/cm²)**",
        min_value=60.0, max_value=180.0,
        value=120.0, step=0.5,
        help="Dataset range: 69.0 – 171.0 mJ/cm²",
    )
    particle_cnt = st.number_input(
        "**🔴 Particle Count**",
        min_value=100, max_value=999,
        value=560, step=1,
        help="Dataset range: 100 – 999",
    )

# ─────────────────────────────────────────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

predict_clicked = st.button("🚀 **Predict Wafer Quality**", type="primary", use_container_width=True)

if predict_clicked:

    # ── Animated spinner ─────────────────────────────────────────────────────
    with st.spinner("🔄 Analysing sensor data..."):
        time.sleep(0.6)

    # ── Build input DataFrame with EXACT column names ─────────────────────────
    raw = pd.DataFrame([{
        "Tool_Type"            : tool_type,
        "Chamber_Temperature"  : chamber_temp,
        "Gas_Flow_Rate"        : gas_flow,
        "RF_Power"             : rf_power,
        "Etch_Depth"           : etch_depth,
        "Rotation_Speed"       : rotation_spd,
        "Vacuum_Pressure"      : vacuum_pres,
        "Stage_Alignment_Error": stage_err,
        "Vibration_Level"      : vibration,
        "UV_Exposure_Intensity": uv_intensity,
        "Particle_Count"       : float(particle_cnt),
    }])

    # Encode Tool_Type → integer
    raw["Tool_Type"] = le.transform(raw["Tool_Type"])

    # Align to training feature order (uses features.pkl)
    raw = raw[features]

    # Scale
    X_scaled = scaler.transform(raw)

    # Predict
    pred = int(model.predict(X_scaled)[0])

    # ── Result badge ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-heading">📊 Prediction Result</div>', unsafe_allow_html=True)

    if pred == 1:
        st.markdown(
            '<div class="result-box result-defect">⚠️ DEFECTIVE WAFER DETECTED</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="result-box result-good">✅ WAFER IS NON-DEFECTIVE</div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    🔬 Semiconductor QC &nbsp;·&nbsp; Model: Decision Tree (SMOTE + GridSearchCV)
    &nbsp;·&nbsp; Key Metric: <b>Recall</b> &nbsp;·&nbsp; Dataset: 4,219 wafers
</div>
""", unsafe_allow_html=True)
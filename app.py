from pathlib import Path
import json
import pandas as pd
import streamlit as st

from src.preprocessing import FEATURES, TARGET
from src.train_model import train_and_save, MODEL_PATH, METRICS_PATH, DATA_PATH
from src.predict import load_model, predict_sample
from src.recommendations import get_screening_notes

st.set_page_config(
    page_title="AquaGuard AI",
    page_icon="💧",
    layout="wide",
    menu_items={"Get help": None, "Report a bug": None, "About": None},
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:#f4f8fd}
.block-container{max-width:1400px;padding-top:1.5rem}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#082f49,#075985)}
[data-testid="stSidebar"] *{color:#f0f9ff!important}
.brand{display:flex;align-items:center;gap:14px;margin:.4rem 0 1.3rem}
.brand-icon{width:54px;height:54px;display:flex;align-items:center;justify-content:center;border-radius:17px;background:linear-gradient(135deg,#38bdf8,#2563eb);font-size:28px}
.brand-title{font-family:'Manrope',sans-serif;font-weight:800;color:#102f49;font-size:30px;letter-spacing:-1px}
.brand-subtitle{color:#647b90;font-size:14px;margin-top:4px}
.hero{padding:28px 30px;border-radius:24px;color:white;background:linear-gradient(120deg,#082f49,#075985 55%,#0284c7);margin-bottom:22px;box-shadow:0 14px 32px #0369a11c}
.hero h2{color:white;font-family:'Manrope',sans-serif;font-size:27px;margin:0 0 8px}
.hero p{color:#dbeafe;font-size:15px;max-width:720px;line-height:1.6;margin:0}
.pill{display:inline-block;margin-top:16px;padding:7px 12px;border:1px solid #ffffff55;border-radius:30px;background:#ffffff12;font-size:12px}
.section-title{font-family:'Manrope',sans-serif;color:#12314b;font-size:20px;font-weight:800;margin:10px 0 4px}
.section-help{color:#6b8196;font-size:13px;margin-bottom:14px}
.metric-card,.panel{background:white;border:1px solid #e1edf8;border-radius:18px;padding:19px 20px;box-shadow:0 5px 16px #0f355309}
.metric-card{min-height:120px}
.metric-label{color:#6b8196;font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.8px}
.metric-value{color:#12314b;font-family:'Manrope',sans-serif;font-size:27px;font-weight:800;margin-top:8px}
.metric-note{color:#7890a5;font-size:12px;margin-top:3px}
.panel{padding:22px 24px}
.result-good,.result-warn{border-radius:18px;padding:22px}
.result-good{background:#ecfdf5;border:1px solid #a7f3d0}
.result-warn{background:#fff7ed;border:1px solid #fed7aa}
.result-title{font-family:'Manrope',sans-serif;font-size:22px;font-weight:800;margin-bottom:5px}
div.stButton>button{background:linear-gradient(100deg,#0284c7,#2563eb);color:white;border:0;border-radius:12px;padding:.72rem 1.15rem;font-weight:700}
div.stButton>button:hover{color:white;border:0}
div[data-testid="stForm"]{border:1px solid #e1edf8;border-radius:18px;padding:18px 20px;background:white}
</style>
""", unsafe_allow_html=True)

FEATURES = list(FEATURES)
LABELS = {
    "ph": ("pH level", "Acidity or alkalinity"),
    "Hardness": ("Hardness", "Dissolved calcium and magnesium"),
    "Solids": ("Total dissolved solids", "Approximate dissolved material"),
    "Chloramines": ("Chloramines", "Disinfectant residual"),
    "Sulfate": ("Sulfate", "Sulfate concentration"),
    "Conductivity": ("Conductivity", "Ability to conduct electricity"),
    "Organic_carbon": ("Organic carbon", "Organic material indicator"),
    "Trihalomethanes": ("Trihalomethanes", "Disinfection by-product indicator"),
    "Turbidity": ("Turbidity", "Water cloudiness"),
}
DEFAULTS = {"ph":7.0,"Hardness":200.0,"Solids":20000.0,"Chloramines":7.0,"Sulfate":330.0,"Conductivity":400.0,"Organic_carbon":14.0,"Trihalomethanes":65.0,"Turbidity":4.0}

def metrics_data():
    try:
        with open(METRICS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError, TypeError):
        return {}

def get_model():
    try:
        return load_model(), True
    except (FileNotFoundError, OSError, ValueError, ImportError):
        return None, False

def card(label, value, note):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 💧 AquaGuard AI")
    st.caption("Water quality monitoring")
    st.markdown("---")
    page = st.radio("NAVIGATION", ["Home / Dashboard", "Predict Water Quality", "About the Project"])
    st.markdown("---")
    _, ready = get_model()
    st.markdown("**Project status**")
    if ready:
        st.success("Prediction model loaded")
    else:
        st.warning("Model not trained yet")
    st.caption("Educational prototype • Not a certified testing tool")

st.markdown('<div class="brand"><div class="brand-icon">💧</div><div><div class="brand-title">AquaGuard AI</div><div class="brand-subtitle">Smarter Water Quality Insights</div></div></div>', unsafe_allow_html=True)
st.markdown('<div class="hero"><h2>Understand your water, one reading at a time.</h2><p>An approachable machine-learning demo that screens water-quality measurements and turns them into an easy-to-understand result.</p><span class="pill">✦ ML-powered screening &nbsp; · &nbsp; 9 water-quality parameters</span></div>', unsafe_allow_html=True)

if page == "Home / Dashboard":
    st.markdown('<div class="section-title">Project at a glance</div><div class="section-help">A quick overview of the model and the information it uses.</div>', unsafe_allow_html=True)
    met = metrics_data()
    rows = None
    try:
        if Path(DATA_PATH).exists():
            rows = len(pd.read_csv(DATA_PATH))
    except Exception:
        pass
    acc = met.get("accuracy")
    acc_text = f"{acc*100:.1f}%" if isinstance(acc,(int,float)) else "—"
    c1,c2,c3 = st.columns(3, gap="medium")
    with c1: card("Model accuracy", acc_text, "Actual held-out test score, if trained")
    with c2: card("Dataset records", f"{rows:,}" if rows is not None else "Not loaded", "Rows in the configured CSV")
    with c3: card("Input parameters", str(len(FEATURES)), "Measurements used by the model")
    st.write("")
    left,right = st.columns([1.15,.85], gap="large")
    with left:
        st.markdown('<div class="panel"><div class="section-title">How it works</div><p style="color:#60778d;line-height:1.8">1. Enter water-quality measurements.<br>2. The trained classifier evaluates the values.<br>3. AquaGuard displays the model output and screening notes.</p><p style="color:#71869a;font-size:12px">The result is a model prediction, not a laboratory water-safety certificate.</p></div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel"><div class="section-title">What you can explore</div><p style="color:#60778d;line-height:2">💧 Nine water-quality features<br>📊 Model performance summary<br>🧪 Instant sample prediction<br>📝 Screening notes</p></div>', unsafe_allow_html=True)
    if not ready:
        st.info("To enable predictions, open Predict Water Quality, upload the dataset, and train the model.")
    st.caption("AquaGuard AI • Student machine-learning project • Not a substitute for certified water testing.")

elif page == "Predict Water Quality":
    st.markdown('<div class="section-title">Predict water quality</div><div class="section-help">Enter measurements from a water sample using the same units as the training dataset.</div>', unsafe_allow_html=True)
    _, ready = get_model()
    with st.expander("Upload dataset or train the model", expanded=not ready):
        st.write("Upload the Kaggle Water Potability CSV (`water_potability.csv`). It must contain the nine feature columns and `Potability`.")
        upload = st.file_uploader("Water quality dataset (CSV)", type=["csv"])
        if upload is not None:
            try:
                df = pd.read_csv(upload)
                required = set(FEATURES + [TARGET])
                missing = required - set(df.columns)
                if missing:
                    st.error("Missing columns: " + ", ".join(sorted(missing)))
                else:
                    Path(DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(DATA_PATH, index=False)
                    st.success(f"Dataset saved ({len(df):,} rows).")
            except Exception as exc:
                st.error(f"Could not read CSV: {exc}")
        if st.button("Train / Retrain model"):
            try:
                with st.spinner("Training model and calculating evaluation metrics..."):
                    result = train_and_save()
                st.success(f"Training complete. Test accuracy: {result['accuracy']*100:.1f}%")
                st.rerun()
            except Exception as exc:
                st.error(f"Training failed: {exc}")
                st.caption("Check the CSV file and required column names.")
    model, ready = get_model()
    if not ready:
        st.warning("A trained model was not found. Upload the dataset above and select Train / Retrain model.")
    else:
        with st.form("prediction_form"):
            st.markdown("#### Sample measurements")
            st.caption("The values below are examples. Replace them with measured readings.")
            values = {}
            cols = st.columns(3, gap="medium")
            for i, feature in enumerate(FEATURES):
                label, help_text = LABELS.get(feature, (feature, "Enter measured value"))
                with cols[i % 3]:
                    values[feature] = st.number_input(label, value=float(DEFAULTS.get(feature,0)), step=0.1, format="%.3f", help=help_text)
            submitted = st.form_submit_button("💧 Analyze Water Quality", use_container_width=True)
        if submitted:
            try:
                prediction = int(predict_sample(model, values))
                notes = get_screening_notes(bool(prediction))
                if prediction:
                    st.markdown('<div class="result-good"><div class="result-title">🟢 Model prediction: Potable class</div><div style="color:#166534">The model assigned this sample to the potable category. This does not confirm that it is safe to drink.</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-warn"><div class="result-title">🟠 Model prediction: Non-potable class</div><div style="color:#9a3412">The model assigned this sample to the non-potable category. Consider professional laboratory testing.</div></div>', unsafe_allow_html=True)
                st.markdown("#### Screening notes")
                for note in notes:
                    st.write("• " + note)
                with st.expander("Review submitted values"):
                    st.dataframe(pd.DataFrame([values]), use_container_width=True, hide_index=True)
            except Exception as exc:
                st.error(f"Could not generate prediction: {exc}")

else:
    st.markdown('<div class="section-title">About AquaGuard AI</div><div class="section-help">A beginner-friendly machine-learning project for exploring water-quality data.</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel"><h3 style="color:#12314b;margin-top:0">Project objective</h3><p style="color:#60778d;line-height:1.8">AquaGuard AI demonstrates how a supervised machine-learning classifier can use water-quality measurements to predict a dataset label. It combines data preprocessing, model training, inference and an interactive Streamlit interface.</p><h3 style="color:#12314b">Model workflow</h3><p style="color:#60778d;line-height:1.8">CSV dataset → missing-value handling → train/test split → Random Forest classifier → evaluation metrics → sample prediction.</p><h3 style="color:#12314b">Important limitation</h3><p style="color:#9a3412;line-height:1.8">This is an educational screening tool. Water safety depends on contaminants, sampling conditions and local standards. Do not use this model as a substitute for accredited laboratory testing.</p></div>', unsafe_allow_html=True)
    st.write("")
    rows = [{"Feature":f, "Display name":LABELS.get(f,(f,""))[0], "What it indicates":LABELS.get(f,(f,""))[1]} for f in FEATURES]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Built with Python, Streamlit, Pandas and Scikit-learn.")

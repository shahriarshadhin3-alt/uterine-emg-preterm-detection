import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from ehg_core import bandpass, find_bursts, extract_features, demo_signal
from scipy import signal as sp

st.set_page_config(page_title="Uterine EMG Monitoring", page_icon="🩺", layout="wide")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("1. Input")
    up = st.file_uploader("Uterine EMG / EHG recording (.csv)", type=["csv"])
    fs = st.number_input("Sampling rate (Hz)", min_value=1.0, max_value=5000.0, value=20.0, step=1.0)
    st.header("2. Detection method")
    model_up = st.file_uploader("Trained model (.joblib) – optional", type=["joblib"])
    threshold = st.slider("Preterm decision threshold", 0.05, 0.95, 0.50, 0.05)

# ---------------- Load data ----------------
if up is not None:
    raw = pd.read_csv(up)
    num = raw.select_dtypes("number")
    skip = [c for c in num.columns if c.lower() in ("time", "t", "sample", "index", "unnamed: 0")]
    data = num.drop(columns=skip).dropna()
    source = up.name
else:
    data = pd.DataFrame(demo_signal(fs))
    source = "demo signal"

with st.sidebar:
    chans = st.multiselect("Channels to analyse", list(data.columns), default=list(data.columns)[:3])

st.markdown("# 🩺 Automated Uterine EMG Monitoring")
st.caption("Analyses abdominal uterine EMG (EHG) recordings and estimates preterm-birth risk.")
st.warning("Educational prototype. It is **not** a medical device and must not be used for diagnosis or clinical decisions.")
if up is None:
    st.info("No file uploaded – showing a synthetic demo signal. Upload a CSV (one column per channel) in the sidebar.")

if not chans or len(data) < 200:
    st.error("Need at least one numeric channel and 200+ samples.")
    st.stop()

# ---------------- Analysis ----------------
with st.spinner("Analysing…"):
    feats = pd.DataFrame({c: extract_features(data[c].values, fs) for c in chans}).T
    mean_f = feats.mean()

bundle = None
if model_up is not None:
    bundle = joblib.load(model_up)
elif os.path.exists("model.joblib"):
    bundle = joblib.load("model.joblib")

proba, pred_msg = None, "No model"
if bundle is not None:
    missing = [f for f in bundle["features"] if f not in mean_f.index]
    if missing:
        pred_msg = "Feature mismatch"
        st.error(f"Model needs features this app does not compute: {missing}")
    else:
        X = mean_f[bundle["features"]].fillna(0).values.reshape(1, -1)
        classes = list(bundle["model"].classes_)
        proba = float(bundle["model"].predict_proba(X)[0][classes.index(1)])
        pred_msg = "PRETERM risk" if proba >= threshold else "TERM-like"

dur = len(data) / fs
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Duration", f"{dur:.0f} s")
c2.metric("Channels", len(chans))
c3.metric("Median frequency", f"{mean_f['median_freq']:.2f} Hz")
c4.metric("Contraction bursts", int(round(mean_f["n_bursts"])))
c5.metric("Result", pred_msg, f"{proba:.0%} preterm prob." if proba is not None else None, delta_color="off")
st.caption("EHG energy of interest is roughly 0.3–3 Hz. Bursts are contraction-like envelope rises (≥5 s). "
           "Load a trained model to get a preterm probability.")

# ---------------- Tabs ----------------
t1, t2, t3, t4, t5 = st.tabs(["Signal and bursts", "Spectrum", "Features", "Prediction", "How it works"])

with t1:
    a, b, c = st.columns([2, 1, 2])
    ch = a.selectbox("Channel", chans)
    show_raw = b.checkbox("Show raw signal", value=False)
    max_start = max(1, int(dur) - 10)
    win_len = c.slider("Window length (s)", 10, max(11, int(dur)), min(int(dur), 300))
    start = st.slider("Window start (s)", 0, max(0, int(dur) - win_len), 0)
    x = data[ch].values
    xf = bandpass(x, fs)
    bursts, env, thr = find_bursts(xf, fs)
    t = np.arange(len(x)) / fs
    m = (t >= start) & (t <= start + win_len)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    if show_raw:
        ax.plot(t[m], x[m], color="#bbbbbb", lw=0.7, label="raw")
    ax.plot(t[m], xf[m], color="#ff4b4b", lw=0.8, label="filtered 0.3–3 Hz")
    for s, e in bursts:
        ax.axvspan(s / fs, e / fs, color="#ff4b4b", alpha=0.12)
    ax.set_xlim(start, start + win_len); ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right"); st.pyplot(fig); plt.close(fig)
    st.caption(f"{len(bursts)} burst(s) detected in this channel (shaded).")

with t2:
    ch2 = st.selectbox("Channel ", chans, key="spec_ch")
    xf2 = bandpass(data[ch2].values, fs)
    fr, psd = sp.welch(xf2, fs, nperseg=min(len(xf2), int(fs * 60)))
    fig, ax = plt.subplots(figsize=(10, 3.2))
    k = fr <= min(5, fs / 2)
    ax.semilogy(fr[k], psd[k], color="#ff4b4b")
    ax.axvline(mean_f["peak_freq"], ls="--", color="gray")
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD")
    st.pyplot(fig); plt.close(fig)
    st.caption(f"Dashed line = mean peak frequency ({mean_f['peak_freq']:.2f} Hz).")

with t3:
    st.dataframe(feats.style.format("{:.4f}"), use_container_width=True)
    st.download_button("Download features (CSV)", feats.to_csv().encode(), "features.csv", "text/csv")
    st.write("**Mean across channels** (this is what the model sees):")
    st.dataframe(mean_f.to_frame("value").style.format("{:.4f}"), use_container_width=True)

with t4:
    if proba is None:
        st.info("No trained model loaded. Run `python train_model.py tpehg_features_raw.csv`, "
                "then put `model.joblib` next to `app.py` (or upload it in the sidebar).")
    else:
        st.progress(min(max(proba, 0.0), 1.0), text=f"Preterm probability: {proba:.1%} (threshold {threshold:.0%})")
        if bundle["model"].__class__.__name__ == "RandomForestClassifier":
            imp = pd.Series(bundle["model"].feature_importances_, index=bundle["features"]).sort_values().tail(10)
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.barh(imp.index, imp.values, color="#ff4b4b"); ax.set_title("Top features in the model")
            st.pyplot(fig); plt.close(fig)

with t5:
    st.markdown("""
1. **Input** – a CSV with one numeric column per EHG channel (time column optional).
2. **Filtering** – band-pass 0.3–3 Hz (4th-order Butterworth, zero-phase).
3. **Bursts** – smoothed Hilbert envelope above mean + 0.5 SD for ≥5 s counts as a contraction-like burst.
4. **Features** – 21 time, frequency and burst features per channel, averaged across channels.
5. **Prediction** – a Random Forest trained on term/preterm recordings returns a preterm probability.

Limitations: small dataset, research use only, no clinical validation.
""")

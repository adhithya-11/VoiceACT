import streamlit as st
import pandas as pd
import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import os
import time
from datetime import datetime
from thefuzz import process
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Voice ERP (Whisper)", layout="wide")

DB_FILE = 'inventory_v2.csv'
LOG_FILE = 'history.log'

# Load Whisper model
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# Init log
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w') as f:
        f.write("Timestamp,Action,Item,Quantity,Status\n")

# Session
for key in ['step', 'action', 'item', 'transcript']:
    if key not in st.session_state:
        st.session_state[key] = 1 if key == 'step' else ""

# ─────────────────────────────────────────────
# AUDIO RECORD
# ─────────────────────────────────────────────
def record_audio(duration=5, fs=16000):
    st.warning("🎤 Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp.name, fs, recording)

    # waveform
    data = recording.flatten()
    fig, ax = plt.subplots()
    ax.plot(data)
    ax.set_title("Audio Waveform")
    st.pyplot(fig)

    return temp.name

# ─────────────────────────────────────────────
# WHISPER LISTEN
# ─────────────────────────────────────────────
def listen_whisper():
    try:
        path = record_audio()
        st.info("📡 Processing with Whisper...")

        result = model.transcribe(path)
        text = result["text"].strip()

        if text:
            st.success("✅ Recognized")
            st.markdown(f"### 📝 `{text}`")
            return text
        else:
            st.error("❌ No speech detected")
            return None

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
        return None

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.title("Voice ERP (Whisper)")

        page = st.selectbox("Navigation", [
            "Voice Control",
            "Dashboard",
            "Logs"
        ])

        if st.button("Reset"):
            st.session_state.step = 1
            st.session_state.action = ""
            st.session_state.item = ""
            st.rerun()

    return page

# ─────────────────────────────────────────────
# PAGE: VOICE FLOW
# ─────────────────────────────────────────────
def page_voice(df):
    st.header("🎙️ Step-by-Step Voice ERP")

    col1, col2 = st.columns([3,2])

    with col1:
        df['Status'] = df.apply(
            lambda x: "CRITICAL" if x['stock_quantity'] <= x['reorder_level'] else "OK",
            axis=1
        )
        st.dataframe(df, use_container_width=True)

    with col2:
        st.subheader("Voice Steps")

        # STEP 1
        if st.session_state.step == 1:
            st.info("Step 1: Say ADD or REMOVE")

            if st.button("Record Action"):
                res = listen_whisper()

                if not res:
                    return

                res = res.lower()
                st.session_state.transcript = res

                if "add" in res:
                    st.session_state.action = "Add"
                    st.session_state.step = 2
                    st.success("Action: ADD")
                    st.rerun()

                elif "remove" in res:
                    st.session_state.action = "Remove"
                    st.session_state.step = 2
                    st.success("Action: REMOVE")
                    st.rerun()

                else:
                    st.error("Invalid action")

        # STEP 2
        elif st.session_state.step == 2:
            st.info(f"Step 2: Say Item for {st.session_state.action}")

            if st.button("Record Item"):
                res = listen_whisper()

                if not res:
                    return

                st.session_state.transcript = res

                choices = df['item_name'].tolist()
                match, score = process.extractOne(res, choices)

                if score > 60:
                    st.session_state.item = match
                    st.session_state.step = 3
                    st.success(f"Matched: {match}")
                    st.rerun()
                else:
                    st.error("Item not found")

        # STEP 3
        elif st.session_state.step == 3:
            st.info(f"Step 3: Say Quantity for {st.session_state.item}")

            if st.button("Record Quantity"):
                res = listen_whisper()

                if not res:
                    return

                st.session_state.transcript = res

                nums = [int(s) for s in res.split() if s.isdigit()]

                if not nums:
                    st.error("No number detected")
                    return

                qty = nums[0]

                if st.session_state.action == "Add":
                    df.loc[df['item_name'] == st.session_state.item, 'stock_quantity'] += qty
                else:
                    df.loc[df['item_name'] == st.session_state.item, 'stock_quantity'] -= qty

                df.drop(columns=['Status']).to_csv(DB_FILE, index=False)

                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(LOG_FILE, 'a') as f:
                    f.write(f"{now},{st.session_state.action},{st.session_state.item},{qty},Success\n")

                st.success(f"✅ DONE: {st.session_state.action} {qty} {st.session_state.item}")

                st.session_state.step = 1
                st.session_state.action = ""
                st.session_state.item = ""
                time.sleep(1)
                st.rerun()

    if st.session_state.transcript:
        st.markdown("---")
        st.code(st.session_state.transcript)

# ─────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────
def page_dashboard(df):
    st.header("Dashboard")

    st.bar_chart(df.set_index('item_name')['stock_quantity'])

# ─────────────────────────────────────────────
# LOGS
# ─────────────────────────────────────────────
def page_logs():
    st.header("Logs")

    if os.path.exists(LOG_FILE):
        st.dataframe(pd.read_csv(LOG_FILE))

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    if not os.path.exists(DB_FILE):
        st.error("Missing inventory file")
        return

    df = pd.read_csv(DB_FILE)

    page = sidebar()

    if page == "Voice Control":
        page_voice(df)
    elif page == "Dashboard":
        page_dashboard(df)
    elif page == "Logs":
        page_logs()

if __name__ == "__main__":
    main()
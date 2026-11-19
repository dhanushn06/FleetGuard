
import streamlit as st
import textwrap
import cv2
import av
import time

from pathlib import Path
from camera.face_tracking import FaceTracker

from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration
)

from database.db import initialize_db, save_checkin
from dashboard.fleet import show_fleet_dashboard


# ==================================================
# CAMERA PROCESSOR
# ==================================================

class CameraProcessor:

    def __init__(self):

        model_path = (
            Path(__file__).resolve().parent
            / "models"
            / "face_landmarker.task"
        )

        self.face_tracker = FaceTracker(
            model_path=str(model_path)
        )

        self.face_detected = False
        self.error_message = None

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        try:

            result = self.face_tracker.detect(
                image,
                time.monotonic()
            )

            self.face_detected = result["tracking_status"]

            if self.face_detected:

                landmarks = result["landmarks"]

                x_values = [point[0] for point in landmarks]
                y_values = [point[1] for point in landmarks]

                x1 = int(max(0, min(x_values)))
                y1 = int(max(0, min(y_values)))

                x2 = int(min(image.shape[1] - 1, max(x_values)))
                y2 = int(min(image.shape[0] - 1, max(y_values)))

                # Face boundary
                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (80, 230, 190),
                    2
                )

                # Detection label
                cv2.putText(
                    image,
                    "FACE DETECTED",
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (80, 230, 190),
                    2
                )

            else:

                cv2.putText(
                    image,
                    "NO FACE DETECTED",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 80, 255),
                    2
                )

        except Exception as error:

            self.error_message = str(error)
            self.face_detected = False

            cv2.putText(
                image,
                "FACE TRACKING ERROR",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )

    def on_ended(self):

        self.face_tracker.close()


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="PulseCam FleetGuard",
    page_icon="💚",
    layout="wide",
    initial_sidebar_state="expanded"
)

initialize_db()


# ==================================================
# HTML HELPER
# ==================================================

def render_html(content):

    st.html(
        textwrap.dedent(content).strip()
    )


# ==================================================
# CUSTOM DESIGN
# ==================================================

render_html("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #0B1120;
    color: #F1F5F9;
    font-family: 'Inter', sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding: 2rem 3rem 3rem 3rem;
    max-width: 1500px;
}

[data-testid="stSidebar"] {
    background: #111B2E;
    border-right: 1px solid #26344B;
}

[data-testid="stSidebar"] * {
    color: #E2E8F0;
}

h1, h2, h3 {
    color: #F8FAFC !important;
}

p, label {
    color: #B7C4D8;
}

.stTextInput input {
    background: #17243A;
    color: white;
    border: 1px solid #34445F;
    border-radius: 10px;
}

.stButton button {
    background: linear-gradient(90deg, #16B99A, #43D9B6);
    color: #071522;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    width: 100%;
}

div[data-testid="stMetric"] {
    background: #17243A;
    border: 1px solid #2B3B55;
    padding: 20px;
    border-radius: 14px;
}

div[data-testid="stMetricValue"] {
    color: #55E6C1;
}

</style>
""")


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    render_html("""
    <div style="padding:15px 5px 25px 5px;">

        <div style="font-size:38px;">💚</div>

        <div style="font-size:22px;font-weight:800;color:white;">
            PulseCam
        </div>

        <div style="font-size:22px;font-weight:800;color:#55E6C1;">
            FleetGuard
        </div>

        <div style="color:#91A3BB;font-size:12px;margin-top:8px;">
            AI-POWERED DRIVER WELLNESS
        </div>

    </div>
    """)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Driver Check-in", "Fleet Dashboard"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.caption("RESEARCH PROTOTYPE · 2026")


# ==================================================
# DRIVER CHECK-IN
# ==================================================

if page == "Driver Check-in":

    # ----------------------------------------------
    # WELCOME HEADER
    # ----------------------------------------------

    render_html("""
    <div style="
        background:linear-gradient(110deg,#172B42,#123C42);
        padding:30px;
        border-radius:18px;
        border:1px solid #28545A;
        margin-bottom:25px;
    ">

        <div style="color:#55E6C1;font-size:12px;font-weight:700;">
            DRIVER WELLNESS PLATFORM
        </div>

        <div style="font-size:34px;font-weight:800;color:white;margin-top:10px;">
            Your wellness.
            <span style="color:#55E6C1;">Our priority.</span>
        </div>

        <div style="color:#C1D2DF;margin-top:10px;">
            A privacy-first camera-based check-in before every journey.
        </div>

    </div>
    """)

    # ----------------------------------------------
    # DRIVER DETAILS
    # ----------------------------------------------

    st.markdown("### 🚘 Driver check-in")

    col1, col2 = st.columns(2)

    with col1:

        driver_id = st.text_input(
            "Driver identification",
            placeholder="Enter your Driver ID"
        )

    with col2:

        st.text_input(
            "Check-in type",
            value="Pre-trip wellness check-in",
            disabled=True
        )

    consent = st.checkbox(
        "I understand and consent to this camera-based check-in."
    )

    st.markdown("---")

    # ----------------------------------------------
    # LIVE CAMERA
    # ----------------------------------------------

    st.markdown("### 📷 Live camera monitoring")

    if not consent:

        st.info(
            "Please provide consent before enabling the camera."
        )

    elif not driver_id.strip():

        st.warning(
            "Please enter your Driver ID before starting the check-in."
        )

    else:

        ctx = webrtc_streamer(
            key="pulsecam-camera",
            mode=WebRtcMode.SENDRECV,

            rtc_configuration=RTCConfiguration({
                "iceServers": [
                    {
                        "urls": [
                            "stun:stun.l.google.com:19302"
                        ]
                    }
                ]
            }),

            video_processor_factory=CameraProcessor,

            media_stream_constraints={
                "video": True,
                "audio": False
            },

            async_processing=True
        )

        if ctx.state.playing:

            st.success("Camera connected successfully!")

            if ctx.video_processor:

                if ctx.video_processor.face_detected:

                    st.success(
                        "Face detected! Please remain stationary."
                    )

                else:

                    st.warning(
                        "Face not detected. Position your face in the camera."
                    )

                if ctx.video_processor.error_message:

                    st.error(
                        ctx.video_processor.error_message
                    )

        else:

            st.info(
                "Click START above the camera preview to enable your webcam."
            )

    # ==================================================
    # EIGHT-REGION MONITORING
    # ==================================================

    st.markdown("---")

    st.markdown("### 🧬 Facial Region Monitoring")

    st.caption(
        "Eight facial regions will be monitored for RGB signal extraction. "
        "Actual readings will appear after camera ROI integration."
    )

    regions = [
        "Forehead Center",
        "Forehead Left",
        "Forehead Right",
        "Left Cheek Upper",
        "Left Cheek Lower",
        "Right Cheek Upper",
        "Right Cheek Lower",
        "Nose Bridge"
    ]

    for start in range(0, len(regions), 4):

        cols = st.columns(4)

        for col, region in zip(
            cols,
            regions[start:start + 4]
        ):

            with col:

                with st.container(border=True):

                    st.markdown(f"**{region}**")

                    st.metric(
                        "RGB Signal",
                        "Awaiting data"
                    )

                    st.caption("Brightness: Awaiting data")

                    st.caption("Region status: Not connected")

    # ==================================================
    # RECORDING SESSION INTERFACE
    # ==================================================

    st.markdown("---")

    st.markdown("### ⏱️ Measurement Session")

    st.caption(
        "Prepare a stationary camera measurement session. "
        "The recording engine will be connected after module integration."
    )

    duration = st.select_slider(
        "Select measurement duration",
        options=[30, 35, 40, 45],
        value=45,
        format_func=lambda x: f"{x} seconds"
    )

    session_col1, session_col2 = st.columns(2)

    with session_col1:

        st.markdown("#### Selected duration")

        st.metric(
            "Recording duration",
            f"{duration} sec"
        )

    with session_col2:

        st.markdown("#### Session status")

        st.info("Awaiting recording integration")

    st.markdown("#### Recording progress")

    st.progress(0)

    st.caption(
        f"0% completed · {duration} seconds selected"
    )

    status_col1, status_col2 = st.columns(2)

    with status_col1:

        st.markdown("**Face tracking**")

        st.warning(
            "Waiting for live camera data"
        )

    with status_col2:

        st.markdown("**Lighting quality**")

        st.warning(
            "Waiting for brightness analysis"
        )

    st.info(
        "Recording controls will become active after the camera "
        "and measurement pipeline are connected. No measurement "
        "is currently being recorded."
    )

    # ==================================================
    # EXPERIMENTAL ALERTNESS MONITORING
    # ==================================================

    st.markdown("---")

    st.markdown(
        "### 👁️ Experimental Visual Alertness Monitoring"
    )

    st.caption(
        "Visual indicators based on facial landmarks. "
        "Actual readings will be connected after Member 1's integration."
    )

    alert1, alert2 = st.columns(2)

    with alert1:

        with st.container(border=True):

            st.markdown("**👁️ Eye Monitoring**")

            st.metric(
                "Eye state",
                "Awaiting data"
            )

            st.metric(
                "Blink events",
                "--"
            )

            st.caption(
                "EAR and eye-closure analysis"
            )

    with alert2:

        with st.container(border=True):

            st.markdown("**😮 Mouth Monitoring**")

            st.metric(
                "Possible yawning events",
                "--"
            )

            st.caption(
                "Mouth Aspect Ratio (MAR)"
            )

    pose1, pose2, pose3 = st.columns(3)

    with pose1:

        st.metric("Head yaw", "--")

    with pose2:

        st.metric("Head pitch", "--")

    with pose3:

        st.metric("Head roll", "--")

    st.info(
        "These are experimental visual observations, not a medical "
        "diagnosis or a determination of driver fitness. "
        "No actual alertness readings are available yet."
    )

    # ==================================================
    # MEASUREMENT OVERVIEW
    # ==================================================

    st.markdown("---")

    st.markdown("### 📊 Measurement overview")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Heart rate",
        "-- BPM"
    )

    c2.metric(
        "Signal quality",
        "--"
    )

    c3.metric(
        "Measurement",
        "Pending"
    )

    st.info(
        "Live face detection is enabled. Actual heart-rate measurement "
        "will be integrated after the signal-processing modules are ready."
    )

    # ==================================================
    # DEMO CHECK-IN
    # ==================================================

    st.markdown("---")

    st.markdown("### 📝 Demo Check-in")

    confirm_demo = st.checkbox(
        "I understand that this saves a demo record only, "
        "not an actual heart-rate measurement."
    )

    if st.button(
        "▶ SAVE DEMO RECORD",
        disabled=not (
            consent
            and driver_id.strip()
            and confirm_demo
        )
    ):

        save_checkin({
            "driver_id": driver_id.strip(),
            "bpm": None,
            "quality_score": None,
            "status": "Test",
            "reasons": (
                "Demo record only. "
                "No actual measurement was performed."
            )
        })

        st.success(
            "Demo record saved successfully!"
        )

        st.rerun()

    # ----------------------------------------------
    # FOOTER
    # ----------------------------------------------

    st.caption(
        "Research prototype only. Face detection does not measure heart rate "
        "or determine fitness for duty."
    )


# ==================================================
# FLEET DASHBOARD
# ==================================================

elif page == "Fleet Dashboard":

    show_fleet_dashboard()

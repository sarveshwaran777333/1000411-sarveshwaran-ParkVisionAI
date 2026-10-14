
from pathlib import Path
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ParkVision AI",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# PATHS AND MODEL LOADING
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "best.pt"


@st.cache_resource
def load_model(model_path):
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Make sure best.pt is inside the model folder."
        )

    return YOLO(str(model_path))


# --------------------------------------------------
# STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background: #0b1220;
        color: #f1f5f9;
    }

    [data-testid="stSidebar"] {
        background: #111b2e;
    }

    .hero {
        padding: 28px;
        border-radius: 18px;
        background: linear-gradient(120deg, #12352f, #13223e);
        border: 1px solid #285b4e;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: #a7f3d0;
        font-size: 38px;
        margin-bottom: 8px;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 16px;
    }

    .metric-card {
        background: #131e31;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #293850;
        margin-bottom: 10px;
    }

    .metric-label {
        color: #9fb0c7;
        font-size: 14px;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 29px;
        font-weight: 700;
        margin-top: 4px;
    }

    .section-title {
        color: #a7f3d0;
        font-size: 22px;
        font-weight: 700;
        margin-top: 14px;
    }

    div[data-testid="stMetric"] {
        background: #131e31;
        border: 1px solid #293850;
        padding: 15px;
        border-radius: 12px;
    }

    .stButton button,
    .stDownloadButton button {
        border-radius: 9px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def get_class_names(yolo_model):
    names = yolo_model.names

    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}

    return {index: str(name) for index, name in enumerate(names)}


def render_metric(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def analyze_image(image_rgb, yolo_model, confidence, image_size):
    results = yolo_model.predict(
        source=image_rgb,
        conf=confidence,
        imgsz=image_size,
        verbose=False,
    )

    result = results[0]
    annotated_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    names = get_class_names(yolo_model)
    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            score = float(box.conf[0].item())
            coordinates = box.xyxy[0].tolist()

            detections.append(
                {
                    "Class ID": class_id,
                    "Class": names.get(class_id, str(class_id)),
                    "Confidence": round(score, 4),
                    "X1": round(coordinates[0], 2),
                    "Y1": round(coordinates[1], 2),
                    "X2": round(coordinates[2], 2),
                    "Y2": round(coordinates[3], 2),
                }
            )

    return annotated_rgb, pd.DataFrame(detections), names


def calculate_summary(
    detections,
    vacant_class,
    occupied_class,
):
    if detections.empty:
        vacant_count = 0
        occupied_count = 0
        total_count = 0
    else:
        vacant_count = int(
            (detections["Class"] == vacant_class).sum()
        )
        occupied_count = int(
            (detections["Class"] == occupied_class).sum()
        )
        total_count = len(detections)

    classified_spaces = vacant_count + occupied_count

    if classified_spaces:
        occupancy = occupied_count / classified_spaces * 100
        availability = vacant_count / classified_spaces * 100
    else:
        occupancy = 0.0
        availability = 0.0

    return {
        "vacant": vacant_count,
        "occupied": occupied_count,
        "total": total_count,
        "classified": classified_spaces,
        "occupancy": occupancy,
        "availability": availability,
    }


def show_summary(summary):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Vacant spaces", summary["vacant"])

    with col2:
        st.metric("Occupied spaces", summary["occupied"])

    with col3:
        st.metric("Total detections", summary["total"])

    with col4:
        st.metric(
            "Occupancy rate",
            f'{summary["occupancy"]:.1f}%',
        )

    st.progress(
        int(round(summary["occupancy"])),
        text=f'Estimated occupancy: {summary["occupancy"]:.1f}%',
    )


def show_detection_table(detections):
    st.markdown(
        '<div class="section-title">Detection details</div>',
        unsafe_allow_html=True,
    )

    if detections.empty:
        st.info("No objects were detected at the selected confidence level.")
        return

    display_df = detections.copy()
    display_df["Confidence"] = (
        display_df["Confidence"] * 100
    ).round(2).astype(str) + "%"

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    csv_data = detections.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download detection report (CSV)",
        data=csv_data,
        file_name="parkvision_detections.csv",
        mime="text/csv",
        use_container_width=False,
    )


def show_image_download(image_rgb, filename="parkvision_annotated.png"):
    image = Image.fromarray(image_rgb)

    with tempfile.NamedTemporaryFile(
        suffix=".png",
        delete=False,
    ) as temp:
        temp_path = Path(temp.name)

    try:
        image.save(temp_path)
        image_bytes = temp_path.read_bytes()
    finally:
        temp_path.unlink(missing_ok=True)

    st.download_button(
        "Download annotated image",
        data=image_bytes,
        file_name=filename,
        mime="image/png",
    )


def run_image_analysis(
    image_rgb,
    yolo_model,
    confidence,
    image_size,
    vacant_class,
    occupied_class,
):
    annotated, detections, names = analyze_image(
        image_rgb,
        yolo_model,
        confidence,
        image_size,
    )

    summary = calculate_summary(
        detections,
        vacant_class,
        occupied_class,
    )

    original_col, result_col = st.columns(2)

    with original_col:
        st.markdown("**Original image**")
        st.image(image_rgb, use_container_width=True)

    with result_col:
        st.markdown("**AI detection result**")
        st.image(annotated, use_container_width=True)

    show_summary(summary)
    show_detection_table(detections)
    show_image_download(annotated)

    return annotated, detections, summary


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

try:
    model = load_model(str(MODEL_PATH))
    CLASS_NAMES = get_class_names(model)
except Exception as error:
    st.error(f"Could not load the YOLO model: {error}")
    st.info(
        "Check that the file exists at model/best.pt "
        "and that the installed Ultralytics version supports it."
    )
    st.stop()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("## ParkVision AI")
    st.caption("AI-powered parking space analysis")

    st.divider()

    st.markdown("### Detection settings")

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Higher values reduce low-confidence detections.",
    )

    image_size = st.select_slider(
        "Inference image size",
        options=[320, 416, 512, 640, 768, 960],
        value=640,
    )

    st.divider()

    st.markdown("### Model information")
    st.write(f"**Model:** `{MODEL_PATH.name}`")
    st.write(f"**Location:** `model/{MODEL_PATH.name}`")
    st.write(f"**Classes:** {len(CLASS_NAMES)}")

    for class_id, class_name in CLASS_NAMES.items():
        st.write(f"{class_id}: {class_name}")

    st.divider()
    st.caption("ParkVision AI | Parking analysis dashboard")


# --------------------------------------------------
# CLASS MAPPING
# --------------------------------------------------

class_options = list(CLASS_NAMES.values())

if not class_options:
    st.error("The model does not contain any class names.")
    st.stop()

vacant_default = next(
    (
        name for name in class_options
        if any(word in name.lower() for word in ["empty", "vacant", "free"])
    ),
    class_options[0],
)

occupied_default = next(
    (
        name for name in class_options
        if name != vacant_default
        and any(word in name.lower() for word in ["occupied", "full", "used"])
    ),
    next(
        (name for name in class_options if name != vacant_default),
        class_options[0],
    ),
)


# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>ParkVision AI</h1>
        <p>
            Intelligent parking space detection and occupancy analysis
            powered by computer vision.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Upload an image, capture a camera image, or analyse a video "
    "to run your trained YOLO model."
)


# --------------------------------------------------
# CLASS CONFIGURATION
# --------------------------------------------------

with st.expander("Configure parking class labels", expanded=False):
    st.caption(
        "Confirm which model class means vacant and which means occupied. "
        "These selections affect the occupancy calculations."
    )

    vacant_class = st.selectbox(
        "Vacant-space class",
        options=class_options,
        index=class_options.index(vacant_default),
    )

    occupied_options = [
        name for name in class_options if name != vacant_class
    ]

    if not occupied_options:
        occupied_class = vacant_class
        st.warning(
            "The model has only one class. Occupancy and availability "
            "percentages cannot be separated reliably."
        )
    else:
        occupied_index = (
            occupied_options.index(occupied_default)
            if occupied_default in occupied_options
            else 0
        )

        occupied_class = st.selectbox(
            "Occupied-space class",
            options=occupied_options,
            index=occupied_index,
        )


# --------------------------------------------------
# TABS
# --------------------------------------------------

tab_image, tab_camera, tab_video, tab_about = st.tabs(
    [
        "Image Dashboard",
        "Camera",
        "Video Analysis",
        "About",
    ]
)


# --------------------------------------------------
# IMAGE DASHBOARD
# --------------------------------------------------

with tab_image:
    st.subheader("Image analysis")

    uploaded_image = st.file_uploader(
        "Upload a parking-lot image",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_upload",
    )

    if uploaded_image is not None:
        try:
            pil_image = Image.open(uploaded_image).convert("RGB")
            image_rgb = np.array(pil_image)

            if st.button(
                "Analyse image",
                type="primary",
                key="analyse_image",
            ):
                with st.spinner("Analysing image..."):
                    annotated, detections, summary = run_image_analysis(
                        image_rgb,
                        model,
                        confidence,
                        image_size,
                        vacant_class,
                        occupied_class,
                    )

                st.success("Image analysis completed.")

        except Exception as error:
            st.error(f"Could not process the image: {error}")
    else:
        st.info("Upload an image to start parking-space detection.")


# --------------------------------------------------
# CAMERA
# --------------------------------------------------

with tab_camera:
    st.subheader("Camera capture")

    st.write(
        "Take a picture of a parking area and analyse it with the model."
    )

    camera_image = st.camera_input(
        "Capture parking area",
        key="camera_capture",
    )

    if camera_image is not None:
        try:
            camera_pil = Image.open(camera_image).convert("RGB")
            camera_rgb = np.array(camera_pil)

            if st.button(
                "Analyse captured image",
                type="primary",
                key="analyse_camera",
            ):
                with st.spinner("Analysing camera image..."):
                    run_image_analysis(
                        camera_rgb,
                        model,
                        confidence,
                        image_size,
                        vacant_class,
                        occupied_class,
                    )

                st.success("Camera image analysis completed.")

        except Exception as error:
            st.error(f"Could not analyse the camera image: {error}")


# --------------------------------------------------
# VIDEO ANALYSIS
# --------------------------------------------------

with tab_video:
    st.subheader("Video analysis")

    st.write(
        "Upload a parking video to detect spaces frame by frame. "
        "Long videos may take some time to process."
    )

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload",
    )

    max_frames = st.slider(
        "Maximum frames to analyse",
        min_value=30,
        max_value=600,
        value=180,
        step=30,
    )

    frame_interval = st.slider(
        "Analyse every Nth frame",
        min_value=1,
        max_value=10,
        value=2,
        help="Increase this value to process videos faster.",
    )

    if uploaded_video is not None:
        if st.button(
            "Analyse video",
            type="primary",
            key="analyse_video",
        ):
            input_path = None
            output_path = None
            capture = None
            writer = None

            try:
                with tempfile.NamedTemporaryFile(
                    suffix=Path(uploaded_video.name).suffix or ".mp4",
                    delete=False,
                ) as temp_input:
                    temp_input.write(uploaded_video.getvalue())
                    input_path = Path(temp_input.name)

                capture = cv2.VideoCapture(str(input_path))

                if not capture.isOpened():
                    st.error("Could not open this video file.")
                else:
                    fps = capture.get(cv2.CAP_PROP_FPS)
                    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

                    if not fps or fps <= 0:
                        fps = 25.0

                    if width <= 0 or height <= 0:
                        st.error("The video has invalid dimensions.")
                    else:
                        output_path = Path(tempfile.mktemp(suffix=".mp4"))

                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(
                            str(output_path),
                            fourcc,
                            fps,
                            (width, height),
                        )

                        if not writer.isOpened():
                            st.error("Could not create the annotated video.")
                        else:
                            total_to_read = min(
                                frame_count if frame_count > 0 else max_frames,
                                max_frames * frame_interval,
                            )

                            progress = st.progress(0)
                            status = st.empty()
                            preview = st.empty()

                            vacant_frames = []
                            occupied_frames = []
                            detection_rows = []
                            processed = 0
                            frame_index = 0
                            last_annotated = None

                            while processed < max_frames:
                                ok, frame_bgr = capture.read()

                                if not ok:
                                    break

                                should_analyse = (
                                    frame_index % frame_interval == 0
                                )

                                if should_analyse:
                                    frame_rgb = cv2.cvtColor(
                                        frame_bgr,
                                        cv2.COLOR_BGR2RGB,
                                    )

                                    results = model.predict(
                                        source=frame_rgb,
                                        conf=confidence,
                                        imgsz=image_size,
                                        verbose=False,
                                    )

                                    result = results[0]
                                    annotated_bgr = result.plot()
                                    last_annotated = cv2.cvtColor(
                                        annotated_bgr,
                                        cv2.COLOR_BGR2RGB,
                                    )

                                    frame_detections = []

                                    if result.boxes is not None:
                                        for box in result.boxes:
                                            class_id = int(box.cls[0].item())
                                            class_name = CLASS_NAMES.get(
                                                class_id,
                                                str(class_id),
                                            )
                                            score = float(box.conf[0].item())

                                            row = {
                                                "Frame": frame_index,
                                                "Class ID": class_id,
                                                "Class": class_name,
                                                "Confidence": round(score, 4),
                                            }

                                            detection_rows.append(row)
                                            frame_detections.append(row)

                                    frame_df = pd.DataFrame(frame_detections)

                                    frame_summary = calculate_summary(
                                        frame_df,
                                        vacant_class,
                                        occupied_class,
                                    )

                                    vacant_frames.append(
                                        frame_summary["vacant"]
                                    )
                                    occupied_frames.append(
                                        frame_summary["occupied"]
                                    )

                                    preview.image(
                                        last_annotated,
                                        caption=f"Processed frame {frame_index}",
                                        use_container_width=True,
                                    )

                                    processed += 1

                                else:
                                    annotated_bgr = frame_bgr

                                writer.write(annotated_bgr)

                                frame_index += 1

                                if frame_count > 0:
                                    percent = min(
                                        100,
                                        int(frame_index / frame_count * 100),
                                    )
                                else:
                                    percent = min(
                                        100,
                                        int(processed / max_frames * 100),
                                    )

                                progress.progress(percent)
                                status.write(
                                    f"Frames processed by AI: {processed} "
                                    f"| Video frame: {frame_index}"
                                )

                            capture.release()
                            capture = None
                            writer.release()
                            writer = None

                            if processed == 0:
                                st.warning(
                                    "No frames could be processed from this video."
                                )
                            else:
                                video_bytes = output_path.read_bytes()
                                st.session_state["parkvision_video_result"] = (
                                    video_bytes
                                )
                                st.session_state["parkvision_video_name"] = (
                                    Path(uploaded_video.name).stem
                                    + "_annotated.mp4"
                                )

                                st.success(
                                    f"Video analysis completed. "
                                    f"{processed} frames analysed."
                                )

                                total_vacant = int(sum(vacant_frames))
                                total_occupied = int(sum(occupied_frames))
                                classified_total = (
                                    total_vacant + total_occupied
                                )

                                mean_vacant = (
                                    total_vacant / processed
                                    if processed else 0
                                )
                                mean_occupied = (
                                    total_occupied / processed
                                    if processed else 0
                                )

                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    st.metric(
                                        "Frames analysed",
                                        processed,
                                    )

                                with col2:
                                    st.metric(
                                        "Average vacant detections/frame",
                                        f"{mean_vacant:.2f}",
                                    )

                                with col3:
                                    st.metric(
                                        "Average occupied detections/frame",
                                        f"{mean_occupied:.2f}",
                                    )

                                st.caption(
                                    "Frame detections are counted separately. "
                                    "These averages are not unique-space counts "
                                    "for the whole video."
                                )

                                if detection_rows:
                                    video_df = pd.DataFrame(detection_rows)

                                    st.dataframe(
                                        video_df,
                                        use_container_width=True,
                                        hide_index=True,
                                    )

                                    st.download_button(
                                        "Download video detection report (CSV)",
                                        data=video_df.to_csv(
                                            index=False
                                        ).encode("utf-8"),
                                        file_name="parkvision_video_report.csv",
                                        mime="text/csv",
                                    )

                                st.download_button(
                                    "Download annotated video",
                                    data=video_bytes,
                                    file_name=st.session_state[
                                        "parkvision_video_name"
                                    ],
                                    mime="video/mp4",
                                )

            except Exception as error:
                st.error(f"Video analysis failed: {error}")

            finally:
                if capture is not None:
                    capture.release()

                if writer is not None:
                    writer.release()

                if input_path is not None:
                    input_path.unlink(missing_ok=True)

                if output_path is not None:
                    output_path.unlink(missing_ok=True)

    else:
        st.info("Upload a video to start video analysis.")

    if st.session_state.get("parkvision_video_result"):
        st.markdown("### Latest processed video")

        st.download_button(
            "Download latest annotated video",
            data=st.session_state["parkvision_video_result"],
            file_name=st.session_state.get(
                "parkvision_video_name",
                "parkvision_annotated.mp4",
            ),
            mime="video/mp4",
            key="download_latest_video",
        )


# --------------------------------------------------
# ABOUT
# --------------------------------------------------

with tab_about:
    st.subheader("About ParkVision AI")

    st.write(
        """
        ParkVision AI is a computer-vision dashboard for analysing
        parking-area images and videos using a trained YOLO model.
        """
    )

    st.markdown("### Features")

    st.markdown(
        """
        - Parking-space object detection
        - Image upload and analysis
        - Camera capture and analysis
        - Video frame analysis
        - Confidence threshold adjustment
        - Annotated image and video exports
        - Detection details in tables
        - CSV reports for further analysis
        """
    )

    st.markdown("### Model details")

    st.write(f"**Checkpoint:** `{MODEL_PATH}`")
    st.write("**Class labels:**")

    for class_id, class_name in CLASS_NAMES.items():
        st.write(f"- `{class_id}` — {class_name}")

    st.warning(
        "Occupancy percentages depend on the trained model's class meanings "
        "and detection quality. Validate the class labels and test the model "
        "on representative parking images before relying on the results."
    )

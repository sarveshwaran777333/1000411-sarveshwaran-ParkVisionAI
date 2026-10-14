## 1000411-sarveshwaran-ParkVisionAI


# ParkVision AI

ParkVision AI is a computer vision application that uses a trained YOLO model to detect parking spaces in images and videos. It provides an interactive Streamlit dashboard for visualising detections, reviewing confidence scores, and exporting results.

## Features

- Parking-space detection using YOLO
- Image upload and analysis
- Camera capture and analysis
- Video frame-by-frame analysis
- Adjustable detection confidence threshold
- Adjustable inference image size
- Annotated image and video downloads
- Detection tables with class names and confidence scores
- CSV reports for image and video detections
- Occupancy and availability estimates based on configured class labels

## Project Structure

```text
ParkVision/
├── app.py
├── README.md
├── requirements.txt
└── model/
    ├── best.pt
    └── last.pt
```

## Requirements

- Python 3.10 or later
- pip
- A supported operating system
- The trained YOLO checkpoint at `model/best.pt`

## Installation

### 1. Clone or download the project

Make sure the project files and the `model` folder are in the same directory.

### 2. Create a virtual environment (recommended)

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

From the project directory, run:

```bash
streamlit run app.py
```

Streamlit will provide a local URL to open the dashboard in your browser.

## Model Configuration

The application loads the model from:

```text
model/best.pt
```

The `last.pt` checkpoint is included as an alternative training checkpoint. To use it, change the model path in `app.py` to:

```python
MODEL_PATH = BASE_DIR / "model" / "last.pt"
```

The application reads the class names from the selected model. Check the **Configure parking class labels** section in the dashboard to ensure the vacant-space and occupied-space classes are mapped correctly.

## How to Use

### Image Dashboard
1. Open the Image Dashboard tab.
2. Upload a parking-area image.
3. Adjust the confidence threshold if required.
4. Click **Analyse image**.
5. Review the annotated image, metrics, and detection table.
6. Download the annotated image or CSV report.

### Camera
1. Open the Camera tab.
2. Capture an image using the camera control.
3. Click **Analyse captured image**.
4. Review the detection results.

Camera access depends on browser permissions and device availability.

### Video Analysis
1. Open the Video Analysis tab.
2. Upload a supported video.
3. Choose the maximum number of frames and frame interval.
4. Click **Analyse video**.
5. Review the processed frames and detection report.
6. Download the annotated video or CSV report.

Long videos may take additional time to process.

## Understanding the Results

- **Vacant spaces:** Detections assigned to the configured vacant-space class.
- **Occupied spaces:** Detections assigned to the configured occupied-space class.
- **Total detections:** All detected objects returned by the model.
- **Occupancy rate:** Occupied detections divided by the total classified vacant and occupied detections.

For video analysis, detections are counted frame by frame. They do not necessarily represent unique parking spaces across the entire video.

## Troubleshooting

### Model file not found

Check that the checkpoint exists at:

```text
model/best.pt
```

### Module not found

Install the project dependencies:

```bash
pip install -r requirements.txt
```

### Camera not working

Allow camera access in the browser and ensure a camera is connected.

### Video cannot be processed

Try a supported video format such as MP4. Very large videos may require more processing time or memory.

### Incorrect occupancy results

Verify the class names and class mapping. Model accuracy depends on the training data, camera angle, lighting, and image quality.

## Limitations

- Predictions depend on the quality and training of the YOLO model.
- Occupancy estimates are only meaningful when the model's class labels correctly represent vacant and occupied spaces.
- Video results are frame-based and may count the same space repeatedly.
- This application is intended for analysis and demonstration; validate its performance before using it for operational parking decisions.

## Technologies Used

- Python
- Streamlit
- Ultralytics YOLO
- OpenCV
- NumPy
- Pandas
- Pillow

## License

Add the applicable project license here if you plan to distribute the application.

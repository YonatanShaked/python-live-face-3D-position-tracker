# Webcam AR Application

This repository contains a Python application for augmented reality (AR) using a webcam. The application overlays 3D objects (e.g., glasses, hats) on detected faces in real-time.

## Prerequisites

- OpenCV (`cv2`)
- NumPy
- Pillow (`PIL`)
- Tkinter

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Make sure your webcam is connected to your system.

2. Run the application:

    ```bash
    python tracker.py
    ```

3. Interact with the application:
    - Press 'q' to quit the application.
    - Press 'd' to toggle debug mode (showing face detection and landmarks).
    - Press 'l' to toggle low-pass filtering for smoother object tracking.
    - Press '1', '2', or '3' to switch between different 3D objects (glasses, hat, head).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project utilizes the following resources:
- Haar Cascade classifier for face detection
- Dlib's facial landmark estimation
- OpenCV for computer vision tasks
- Tkinter for the graphical user interface

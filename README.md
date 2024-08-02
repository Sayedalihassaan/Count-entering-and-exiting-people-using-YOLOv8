
# People Counting System Using YOLOv8

This project uses YOLOv8 to track and count people entering and exiting a defined area. The project includes the following components:

- **YOLOv8 Model**: Used for object detection.
- **Tracker**: Tracks the detected objects across frames.
- **Video Processing**: Reads video frames and processes them to count people.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Download the YOLOv8 model weights and place them in the project directory:

    ```plaintext
    yolov8s.pt
    ```

## Usage

1. Define the areas for entering and exiting in the `area1` and `area2` variables.

    ```python
    area1 = [(312, 388), (289, 390), (474, 469), (497, 462)]
    area2 = [(279, 392), (250, 397), (423, 477), (454, 469)]
    ```

2. Run the script to process a video:

    ```bash
    python main.py
    ```

3. Press `q` to quit the video window.

## Code Overview

- **main.py**: The main script that processes the video and counts people.
- **tracker.py**: Contains the `Tracker` class for tracking objects across frames.
- **coco.txt**: Contains class labels for COCO dataset.

### main.py

The main script contains the following functions:

- `points(events, x, y, flags, param)`: Handles mouse events to display coordinates.
- `process_video(video_path)`: Reads and processes video frames.
- `update_tracking_info(frame, bbox_id)`: Updates tracking information for each frame.
- `update_area_info(frame, id, x3, y3, x4, y4)`: Updates the entering and exiting areas for tracked objects.
- `display_info(frame)`: Displays information about the number of people entering and exiting.

### tracker.py

The `Tracker` class contains methods to update the tracked objects and assign IDs to them.

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License.

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/yolov8)
- [COCO Dataset](https://cocodataset.org/)

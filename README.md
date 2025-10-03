# Django-OpenCV Face Detection

This project demonstrates face detection using OpenCV in a Django environment. It includes a simple Python script that detects faces in an image using a pre-trained Caffe model.

## Features

- Face detection using OpenCV's DNN module
- Pre-trained SSD MobileNet model for face detection
- Simple image processing and visualization

## Requirements

- Python 3.x
- OpenCV
- NumPy

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd Django-OpenCV
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure the model files are in the `models/` directory and the test image is in `images/`.

## Usage

Run the face detection script:
```
python main.py
```

This will load the image `images/big_team.jpg`, detect faces, draw bounding boxes, and display the result.

## Project Structure

- `main.py`: Main script for face detection
- `requirements.txt`: Python dependencies
- `models/`: Pre-trained model files
- `images/`: Sample images for testing

## License

This project is open-source. Please check individual model licenses for usage restrictions.
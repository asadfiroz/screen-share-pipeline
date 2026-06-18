# Screen Share via Phone Camera

A browser based tool that points your phone camera at a laptop or computer screen,
detects the screen, flattens it to a straight on view, and sharpens it. The goal is
to let you show your screen to someone (a mentor, a teammate, an advisor) in
situations where normal screen sharing tools are blocked or not available.

Live demo: https://asadfiroz.github.io/screen-share-pipeline/

(Open it on a phone, allow the camera, and point the back camera at a screen.)

## The idea

Sometimes you cannot share your screen the normal way. A simple alternative is to
point a phone camera at the screen. The problem is that a phone photo of a screen is
tilted, warped, and hard to read. This project fixes that automatically: it finds the
screen in the camera view and corrects it to a clean, head on, readable image in real
time.

## How it works (the pipeline)

The app runs the same five steps on every camera frame:

1. Capture a live frame from the phone camera.
2. Detect the four corners of the screen.
3. Flatten the screen using a perspective transform so it looks straight on.
4. Sharpen the flattened image.
5. Show the result.

Only step 2 (detecting the screen) changed as the project grew. Everything around it
stayed the same.

## Two ways to detect the screen

### Version 1: Classical computer vision

The first version used classical image processing: find edges, find shapes, and keep
the largest four sided shape. This works well when the screen is bright and fills the
frame, but it fails on busy backgrounds, on a plain desktop wallpaper, and when other
rectangles (windows, picture frames) are in view, because it only looks at edges and
has no real idea of what a screen is.

### Version 2: Machine learning model (current)

The current version uses a small neural network that looks at the image and predicts
the four screen corners directly. It learns the shape of a screen rather than its
content, so it handles many more situations than the classical method.

## The model

- Input: a 256 x 256 image.
- Output: 8 numbers, which are the x and y of the four corners.
- A small convolutional neural network (CNN) built in PyTorch.
- Exported to the ONNX format and run directly in the browser using ONNX Runtime Web,
  so no server is needed.

## Training data

The model was trained on two kinds of data:

1. Synthetic data: a Python script builds thousands of fake laptops (a screen with a
   bezel and a keyboard) placed on real room photos at different sizes and angles.
   Because the script creates the screen, it knows the exact corner positions, so the
   labels are free and accurate.
2. Real data: a separate capture page uses the older classical detector to auto label
   real photos of an actual screen. When the classical detector locks on, it saves the
   photo and the corners it found. This gave a few hundred real, correctly labeled
   images with almost no manual work.

The model was trained on both sets combined, which made it work much better on real
screens than synthetic data alone.

## Tech stack

- App and inference: HTML, JavaScript, OpenCV.js (for the warp and sharpen),
  ONNX Runtime Web (to run the model).
- Training: Python, PyTorch, OpenCV, NumPy, on a free Google Colab GPU.
- Hosting: GitHub Pages (free static hosting with HTTPS, which the camera needs).

## How to use it

1. Open the demo link above on your phone.
2. Allow camera access.
3. Point the back camera at a screen.
4. A green outline appears on the detected screen, and the lower view shows the
   flattened, straightened version.

## Files in this repository

- `index.html` : the live app (camera, model, flatten, sharpen).
- `capture.html` : tool used to collect and auto label real training images.
- `generate_data.py` : creates the synthetic training data.
- `train.py`, `train_v2.py`, `train_v3.py` : model training scripts.
- `screen_corners.onnx` : the trained model used by the app.

## Screenshots

Screen detected and outlined:

![Screen detected 1](detection-1.png)

![Screen detected 2](detection-2.png)

## Current status

What works:
- Detects and flattens a screen when it fills the frame fairly head on.
- Runs fully in the browser on a phone, no app install and no server.
- Trained on a mix of synthetic and real data.

Known limitations:
- The corners can drift on some angles, sometimes including part of the keyboard.
- It can still draw a box on a busy non screen scene, because the current model does
  not yet output a "screen present" confidence.

## Future work

- Add a "screen present" confidence output so the box hides when there is no screen.
- Collect more real data at harder angles and distances to tighten the corners.
- Move the model to a background worker so the live video is smoother.
- Try a pretrained backbone (such as MobileNet) for better accuracy.

## Note

This is a work in progress prototype built as a learning and portfolio project. It is
actively being improved.

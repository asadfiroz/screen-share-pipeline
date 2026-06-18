"""
Step 1: Video input loop.

Goal of this file: prove we can pull frames from a video source one at a
time and display them. This is the skeleton every later stage plugs into -
detection, warping and enhancement all happen *inside* this loop, once per
frame.
"""

import cv2

# The video source.
#   0            -> your laptop's built-in webcam (quick sanity check)
#   "clip.mp4"   -> a recorded video file (use this for real phone footage)
# To test with your phone: film your laptop screen, copy the file next to
# this script, and set SOURCE = "your_filename.mp4".
SOURCE = 0

cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    raise SystemExit(f"Could not open video source: {SOURCE}")

while True:
    ok, frame = cap.read()      # grab the next frame; ok is False when the video ends
    if not ok:
        break

    cv2.imshow("Step 1 - raw video (press q to quit)", frame)

    # waitKey(1) waits ~1ms for a keypress and is also what lets the window
    # actually refresh. Quit when the user presses q.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

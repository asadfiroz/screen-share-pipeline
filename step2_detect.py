"""
Step 2: Find the laptop screen in each frame (classical computer vision).

We don't train anything yet. We use the same trick a document scanner uses:
the screen is a big four-sided shape, so we look for the largest convex
quadrilateral in the frame and assume that's the screen.

This is fast, needs no data, and lets us build and test the rest of the
pipeline immediately. It is also fragile in messy lighting - which is exactly
why we replace this step with a trained YOLO model later. For now it teaches
the core ideas and gives us a working scaffold.
"""

import cv2
import numpy as np

SOURCE = "Sample_Clip.mov" # see step1_capture.py for how to point this at phone footage


def find_screen(frame):
    """Return the screen's 4 corners as a (4,2) float32 array, or None."""

    # 1. Grayscale. We care about *brightness edges*, not color, and working
    #    on one channel instead of three is faster.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Blur. Smooths away fine texture and noise so the next step latches
    #    onto big boundaries (the screen edge) instead of every speck.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Canny edge detection. Marks pixels where brightness changes sharply.
    #    The screen's border against its surroundings becomes a bright outline.
    edges = cv2.Canny(blurred, 50, 150)

    # 4. Dilate. Thickens those edge pixels slightly to close small gaps, so
    #    the outline forms one continuous loop we can trace.
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    # 5. Find contours: continuous edge loops. RETR_EXTERNAL keeps only the
    #    outermost loops, so we get the screen's border, not the content edges
    #    inside it.
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # 6. Biggest shapes first - the screen is one of the largest things in view.
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    frame_area = frame.shape[0] * frame.shape[1]
    for c in contours[:5]:
        # 7. Simplify the wobbly contour into a clean polygon. approxPolyDP
        #    drops points that don't add much shape; a rectangle collapses to
        #    its 4 corners. epsilon scales with the contour's perimeter.
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)

        # 8. Accept it only if it's a 4-cornered, convex shape covering a real
        #    chunk of the frame. That rules out triangles, dented blobs and
        #    tiny stray quads.
        if len(approx) == 4 and cv2.isContourConvex(approx):
            if cv2.contourArea(approx) > 0.10 * frame_area:
                return approx.reshape(4, 2).astype("float32")

    return None


def main():
    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {SOURCE}")

    # Make the display window resizable and scale the video to fit it, instead
    # of opening at the video's full pixel size. A phone clip is often larger
    # than your monitor, so the default window would only show the top-left
    # corner of each frame. WINDOW_NORMAL lets the whole frame scale down to
    # fit, and you can drag the window edges to resize it however you like.
    window = "Step 2 - screen detection (press q to quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        corners = find_screen(frame)
        if corners is not None:
            pts = corners.astype(int)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 3)      # green outline
            for (x, y) in pts:
                cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)      # red corner dots

        cv2.imshow(window, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

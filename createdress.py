import cv2
import numpy as np
import os

# Create folder if not exists
out_dir = os.path.join(os.path.dirname(__file__), "assets", "dresses")
os.makedirs(out_dir, exist_ok=True)

def make_dress(filename, color):
    # Transparent canvas
    dress = np.zeros((600, 300, 4), dtype=np.uint8)

    # Neckline
    cv2.ellipse(dress, (150, 80), (80, 40), 0, 0, 360, color, -1)

    # Body polygon
    pts = np.array([[70, 80], [230, 80], [200, 500], [100, 500]], np.int32)
    cv2.fillPoly(dress, [pts], color)

    # Add some details
    white = (255, 255, 255, 255)
    for y in range(120, 500, 40):
        cv2.line(dress, (90, y), (210, y), white, 2)

    # Save as PNG
    path = os.path.join(out_dir, filename)
    cv2.imwrite(path, dress)
    print(f"Created {path}")

# Make 2 sample dresses
make_dress("dress1.png", (20, 90, 200, 255))   # Blue
make_dress("dress2.png", (180, 50, 120, 255))  # Pink

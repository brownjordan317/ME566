import cv2


class GridDetector:

    def __init__(self, image):
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.detect_corners()

    def detect_corners(self):
        # Common inner corner sizes to test (Width from 3 to 14, Height from 3 to 14)
        for w in range(3, 15):
            for h in range(3, 15):
                size = (w, h)
                ret, corners = cv2.findChessboardCorners(self.image, size, None)

                if ret:
                    self.corners = corners
                    self.dimensions = size
                    print(f"Detected {len(corners)} inner corners.")
                    print(f"Grid dimensions found: {w}x{h}")
                    return

        print("No chessboard detected.")

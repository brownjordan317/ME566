import cv2


class ChessboardDetector:
    def __init__(self, chessboard_size=(7, 7)):
        self.chessboard_size = chessboard_size

    def detect_chessboard(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

        if ret:
            # Refine corner locations
            corners2 = cv2.cornerSubPix(
                gray,  # The input image (grayscale)
                corners,  # The initial corner locations
                (11, 11),  # The search window size (width, height)
                (-1, -1),  # The zero zone (no zero zone)
                criteria=(
                    cv2.TERM_CRITERIA_EPS
                    + cv2.TERM_CRITERIA_MAX_ITER,  # The type of termination criteria
                    30,  # The maximum number of iterations
                    0.001,  # The desired accuracy
                ),  # The termination criteria for the corner refinement algorithm
            )
            return True, corners2
        else:
            return False, None

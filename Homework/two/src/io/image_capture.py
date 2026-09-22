import os
import cv2


class ImageCapture:
    def __init__(self, camera_path, save_path, extension=".jpg", capture_rate=5):
        # camera path can be an integer (for webcam) or an RTSP stream URL
        self.camera_path = camera_path
        self.save_path = save_path
        # create the save_path directory if it doesn't exist
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.extension = extension.lower()
        # check if extension is valid
        self.check_extension()
        self.capture_rate = capture_rate

    def check_extension(self):
        valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
        if self.extension.lower() not in valid_extensions:
            raise ValueError(
                f"Invalid extension: {self.extension}. Valid extensions are: {valid_extensions}"
            )
        # strip the leading dot if present
        self.extension = self.extension.removeprefix(".")

    def open_camera(self):
        # Open the camera stream
        self.cap = cv2.VideoCapture(self.camera_path)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open camera stream: {self.camera_path}")

    def io(self):
        self.open_camera()
        frame_count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Save the frame at the specified capture rate
            if frame_count % self.capture_rate == 0:
                filename = os.path.join(
                    self.save_path, f"frame_{frame_count}.{self.extension}"
                )
                cv2.imwrite(filename, frame)
                print(f"Saved {filename}")

            frame_count += 1

            # Display the frame (optional)
            cv2.imshow("Camera Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    camera_path = "/dev/video0"
    save_path = "Homework/two/images/hw2_translation_images"
    extension = ".jpg"
    capture_rate = 5  # Capture every 5 frames

    image_capture = ImageCapture(camera_path, save_path, extension, capture_rate)
    image_capture.io()
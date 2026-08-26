import os
import cv2
import numpy as np


class ImageReader:
    def __init__(self, path=None):
        self.path = path
        if self.path is not None:
            self.images = self.read_path()

    def get_path_type(self):
        # Check if the path is a directory or a file
        if os.path.isdir(self.path):
            return "directory"
        elif os.path.isfile(self.path):
            # get the file extension
            _, ext = os.path.splitext(self.path)
            if ext.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]:
                return "image_file"
        else:
            raise ValueError(f"Invalid path: {self.path}")

    def read_path(self, path=None):
        if path is not None:
            self.path = path
        path_type = self.get_path_type()
        images = []

        if path_type == "directory":
            # Read all image files in the directory
            for filename in os.listdir(self.path):
                file_path = os.path.join(self.path, filename)
                file_type = self.get_path_type()
                if file_type == "image_file":
                    image = cv2.imread(file_path)
                    if image is not None:
                        images.append(image)
                elif file_type == "directory":
                    # Recursively read images from subdirectories
                    sub_reader = ImageReader(file_path)
                    images.extend(sub_reader.images)
        elif path_type == "image_file":
            # Read the single image file
            image = cv2.imread(self.path)
            if image is not None:
                images.append(image)

        return np.asarray(images)

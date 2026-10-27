import re
from pathlib import Path

import cv2


class ImageReader:
    EXTENSIONS = frozenset({'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'})

    def __init__(self, path=None):
        self.path = path
        self.image_paths = []
        if path is not None:
            self.images = self.read_path()

    def get_path_type(self, path=None):
        path = Path(self.path if path is None else path)
        if path.is_dir():
            return 'directory'
        if path.is_file():
            return 'image_file' if path.suffix.lower() in self.EXTENSIONS else None
        raise ValueError(f'Invalid path: {path}')

    def read_path(self, path=None):
        """Read images in natural relative-path order, including subfolders.

        Return a list so images with different resolutions are supported.
        image_paths contains the corresponding successfully loaded paths.
        """
        if path is not None:
            self.path = path
        kind = self.get_path_type()
        root = Path(self.path)
        def natural_key(path):
            return tuple((1, int(part)) if part.isdigit() else (0, part.casefold())
                         for part in re.split(r'(\d+)', path.relative_to(root).as_posix()))

        candidates = sorted(root.rglob('*'), key=natural_key) if kind == 'directory' else [root]
        images = []
        self.image_paths = []
        for candidate in candidates:
            if not candidate.is_file() or candidate.suffix.lower() not in self.EXTENSIONS:
                continue
            image = cv2.imread(str(candidate))
            if image is not None:
                images.append(image)
                self.image_paths.append(candidate)
        self.images = images
        return images

import os
import requests
import tarfile
import random
from PIL import Image
from tqdm import tqdm
from typing import Tuple, List, Optional


class BackgroundGenerator:
    """
    Manages and provides background images for dataset generation.

    Downloads, stores, and processes a collection of background images from a specified URL.
    This is particularly useful for generating diverse backgrounds in supervised learning datasets.

    Attributes:
        background_size (tuple): Desired size (width, height) of background images, adjusted within a specified
            range due to original image size constraints. Defaults to (512, 512)
        background_names (list): List of all background image files.
    """

    def __init__(self, background_size: Tuple[int, int] = (512, 512)) -> None:
        """
        Initializes the BackgroundGenerator with an optional specific background size.

        Args:
            background_size (tuple): The width and height in pixels of the background images,
                adjusted if outside the 300x300 to 640x640 range. Defaults to (512, 512).
        """
        self._data_url = (
            "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
        )
        self._data_folder = os.path.join(self.get_data_path(), "background_images")
        self._data_filename = "dtd-r1.0.1.tar.gz"
        self.download(self._data_url, self._data_filename, self._data_folder)
        self.background_size = self._validate_background_size(background_size)
        self.background_names = self.get_background_names()

    def get_data_path(self) -> str:
        """
        Determines and ensures the existence of a data storage path.

        Calculates the path for storing downloaded data based on the script's location.
        Creates the data directory if it does not exist, facilitating consistent data access.

        Returns:
            str: The absolute path to the 'data' directory.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Navigate up the directory hierarchy to define the "data" directory's intended location
        parent_dir = os.path.dirname(script_dir)
        grandparent_dir = os.path.dirname(parent_dir)
        greatgrandparent_dir = os.path.dirname(grandparent_dir)
        data_dir = os.path.join(greatgrandparent_dir, "data")

        # Create the "data" directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"'data' directory created at: {data_dir}")
        return data_dir

    def download(self, url: str, file_name: str, extract_folder: str) -> None:
        """
        Downloads and extracts a dataset archive from a specified URL.

        Handles the download of a compressed file, its storage, and extraction into a target folder.
        Optionally removes the downloaded archive after successful extraction.

        Args:
            url (str): URL of the file to be downloaded.
            file_name (str): Name for saving the downloaded file.
            extract_folder (str): Target folder for extracting the file's contents.
        """
        if not os.path.exists(extract_folder):
            os.makedirs(extract_folder, exist_ok=True)

            print(f"Downloading {file_name}...")

            with requests.get(url, stream=True) as response:
                response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code

                total_size_in_bytes = int(response.headers.get("content-length", 0))
                block_size = 1024  # 1 Kibibyte
                progress_bar = tqdm(
                    total=total_size_in_bytes, unit="iB", unit_scale=True
                )

                with open(file_name, "wb") as file:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        file.write(data)
                progress_bar.close()

            print(f"Extracting {file_name}...")
            with tarfile.open(file_name, "r:*") as tar:
                tar.extractall(path=extract_folder)

            print(f"Extracted to {extract_folder}")
            os.remove(file_name)  # Optionally remove the tar file after extraction

    def _validate_background_size(
        self, background_size: Tuple[int, int]
    ) -> Tuple[int, int]:
        """
        Validates and adjusts the provided background size within acceptable limits.

        Ensures the background size dimensions are within the 300 to 640 pixels range.
        Adjusts dimensions outside this range to the nearest valid value.

        Args:
            background_size (tuple): Desired background size as a (width, height) tuple.

        Returns:
            tuple: Validated and possibly adjusted background size.

        Raises:
            ValueError: If input is not a tuple of two integers.
        """
        if not (
            isinstance(background_size, tuple)
            and len(background_size) == 2
            and isinstance(background_size[0], int)
            and isinstance(background_size[1], int)
        ):
            raise ValueError("background_size should be a tuple of two integers")

        width = background_size[0]
        if width < 300:
            print(f"background_size width {width} below minimum. Defaulting to 300.")
            width = 300
        if width > 640:
            print(f"background_size width {width} above maximum. Defaulting to 640.")
            width = 640

        height = background_size[1]
        if height < 300:
            print(f"background_size height {height} below minimum. Defaulting to 300.")
            height = 300
        if height > 640:
            print(f"background_size height {height} above maximum. Defaulting to 640.")
            height = 640

        background_size = (width, height)
        return background_size

    def get_background_names(self) -> List[str]:
        """
        Retrieves a flattened list of filenames of all background images within the dataset.

        This method scans the dataset's base folder for all subfolders, listing
        the names of files that are recognized as image files (PNG, JPG, JPEG) across
        all these subfolders. The result is a single list that aggregates the filenames
        from all the subfolders, providing a comprehensive view of the available
        background images.

        Returns:
            list: A list containing the filenames of all background images found within
                the dataset's base directory. Filenames are listed as strings and
                include images across all subfolders.
        """
        names = []
        base_folder = os.path.join(self._data_folder, "dtd", "images")
        subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir()]
        for subfolder in subfolders:
            files = os.listdir(subfolder)
            names.extend(
                [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            )
        return names

    def get_background(self, image_file: Optional[str] = None) -> Image.Image:
        """
        Fetches and resizes a background image from the stored dataset to the specified background size.

        This method either selects a specific background image if an image file name is provided,
        or randomly picks one from the dataset. The selected image is then resized to match the
        configured `background_size`. If the specified file does not exist or other errors occur
        during file handling, appropriate exceptions are raised.

        Args:
            image_file (str, optional): The name of the specific background image file to fetch.
                If None, a random image file from the dataset is selected. The file name should
                be relative to the base dataset directory. Defaults to None.

        Returns:
            Image: The selected and resized background image as a PIL Image object.

        Raises:
            FileNotFoundError: Raised if the specified image file does not exist in the dataset directory.
            RuntimeError: Raised if there are issues opening the file, such as corruption or unexpected file format.

        Example:
            >>> bg_generator = BackgroundGenerator(background_size=(512, 512))
            >>> background_image = bg_generator.get_background('example.jpg')
            >>> background_image.show()  # This will display the image.

        Note:
            The images are assumed to be stored in a directory structure within a 'dtd/images' folder. Each
            subfolder in 'dtd/images' represents a different category or type of backgrounds. Image files
            should be in formats recognized by PIL (e.g., PNG, JPG).
        """
        base_folder = os.path.join(self._data_folder, "dtd", "images")
        try:
            if image_file is None:
                subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir()]
                subfolder = random.choice(subfolders)
                files = os.listdir(subfolder)
                image_files = [
                    f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ]
                image_file = random.choice(image_files)
            else:
                subfolder = image_file.split("_")[0]
                subfolder = os.path.join(base_folder, subfolder)
                image_file = os.path.join(
                    base_folder, image_file.split("_")[0], image_file
                )

            image_path = os.path.join(subfolder, image_file)
            image = Image.open(image_path)
            image = image.resize(self.background_size, Image.Resampling.LANCZOS)
            return image
        except FileNotFoundError:
            raise FileNotFoundError(f"The specified file does not exist: {image_path}")
        except Exception as e:
            raise RuntimeError(f"An error occurred while loading the image: {e}")

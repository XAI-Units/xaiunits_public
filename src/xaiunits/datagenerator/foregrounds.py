import math
import random
import os
import urllib.parse
import re
import requests
from tqdm import tqdm
from io import BytesIO
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Union, List, Callable


class ForegroundGenerator(ABC):
    """
    Abstract base class for generating overlay images (e.g., geometric shapes, dinosaurs) with customizable colors.

    This class sets up common attributes and methods for its subclasses to generate images with specific characteristics
    and colors. Subclasses are expected to implement the `get_shape` method, providing a way to create and retrieve images
    with optional color overlays.

    Attributes:
        shape_names (list): Names of available shapes or dinosaurs. Populated by subclasses based on their specific image types.
    """

    def __init__(self) -> None:
        """Initializes a ForegroundGenerator object."""
        self._colors_rgba: Dict[str, Tuple[int, int, int, int]] = {
            "red": (255, 0, 0, 255),
            "green": (0, 255, 0, 255),
            "blue": (0, 0, 255, 255),
            "yellow": (255, 255, 0, 255),
            "black": (0, 0, 0, 255),
            "white": (255, 255, 255, 255),
            "orange": (255, 165, 0, 255),
            "purple": (128, 0, 128, 255),
        }
        self._data_path: str = self.get_data_path()

    def validate_color(
        self, color: Union[str, Tuple[int, int, int, int]]
    ) -> Tuple[int, int, int, int]:
        """
        Validates and returns the RGBA value for a given color name or tuple.

        Args:
            color (str | tuple): The color specified as a name or RGBA tuple.

        Returns:
            tuple: The RGBA tuple corresponding to the specified color.

        Raises:
            ValueError: If input color is a value that is not supported.
        """
        if isinstance(color, str):
            try:
                color = self._colors_rgba[color]  # Check if it's in the dictionary
                return color
            except KeyError:
                raise ValueError(
                    "Color name must be a valid key in the color dictionary self._colors_rgba"
                )
        elif (
            isinstance(color, tuple)
            and len(color) == 4
            and all(isinstance(c, int) and 0 <= c <= 255 for c in color)
        ):
            return color
        else:
            raise ValueError(
                "color must be a lowercase string or 4-d tuple of int between 0 and 255"
            )

    def apply_color_fill(
        self, image: Image.Image, color: Optional[Tuple[int, int, int, int]]
    ) -> Image.Image:
        """
        Applies a color fill to an overlay shape while preserving transparency.

        If a color is provided, apply it to the non-transparent parts of the image,
        effectively changing the shape's color while keeping the background transparent.

        Args:
            image (PIL.Image.Image): Original PIL.Image.Image object of a shape.
            color (tuple | NoneType): RGBA tuple to apply as the new color.

        Returns:
            PIL.Image.Image: A new PIL Image object with the color applied.
        """
        if color is None:
            return image

        if image.mode != "RGBA":
            image = image.convert("RGBA")

        alpha = image.split()[3]
        solid_color = Image.new("RGBA", image.size, color)
        filled_image = Image.composite(solid_color, image, alpha)
        return filled_image

    def get_data_path(self) -> str:
        """
        Determines the path for storing downloaded data, locating the 'data' directory at a fixed level above
        this script's location in the directory hierarchy.

        This function calculates an absolute path to a 'data' directory intended to reside a few levels above the
        directory containing this script. It ensures the 'data' directory exists, creating it if necessary.
        This approach allows for a consistent data storage location relative to the script's position in the project
        structure, facilitating access across different environments and setups.

        Returns:
            str: The absolute path to the 'data' directory, ensuring it is consistently located relative to the
                script's position in the project's directory hierarchy.
        """
        # Get the directory of the current script
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

    @abstractmethod
    def get_shape(
        self,
        name: Optional[str] = None,
        color: Optional[Union[str, Tuple[int, int, int, int]]] = None,
    ) -> Tuple[Image.Image, str]:
        """
        Abstract method to generate and return an image of a specific overlay type (shape or dinosaur) in a specified color.

        Subclasses must implement this method to create images according to their specialization (geometric shapes or dinosaurs)
        and optionally apply a color overlay based on the provided color parameter.

        Args:
            name (str, optional): The name of the specific shape or dinosaur to generate.
                Defaults to a random selection if None.
            color (str | tuple, optional): The color to apply to the image. It can be a color name or an RGBA tuple.
                Defaults to no color overlay if None.

        Returns:
            tuple[PIL.Image.Image, str]: A tuple with the PIL.Image.Image object of the generated foreground with the applied color overlay,
                and the name of the generated foreground.
        """
        pass


class GeometricShapeGenerator(ForegroundGenerator):
    """
    Generates images of geometric shapes with customizable colors.

    This class provides functionality to generate images of various geometric shapes,
    such as circles, ellipses, rectangles, and polygons with a specified number of sides,
    each with a specified or default color. Shapes are drawn on a transparent background.

    Inherits from:
        ForegroundGenerator: The base class for generating foreground images.

    Attributes:
        size (int): The size of the square image in pixels. Defaults to 200.
        shape_names (list): List of all possible shape names to be drawn.
        shape_id_map (dict): Maps an integer to each name in shape_names list
    """

    def __init__(self, size: int = 200) -> None:
        """Initializes a GeometricShapeGenerator object."""
        super().__init__()
        self.size: int = size
        self.shape_names: List[str] = list(self.geometric_shapes().keys())
        self.shape_id_map: Dict[str, int] = {
            shape: index for index, shape in enumerate(self.shape_names)
        }

    def geometric_shapes(self) -> Dict[str, Callable[[], Image.Image]]:
        """
        Provides a dictionary of lambdas for generating geometric shapes.

        Returns:
            dict: A dictionary mapping shape names to lambdas that generate shape images.
        """
        return {
            "circle": self.make_circle,
            "ellipse": self.make_ellipse,
            "triangle": lambda: self.make_ngon(3),
            "square": lambda: self.make_ngon(4),
            "rectangle": self.make_rectangle,
            "pentagon": lambda: self.make_ngon(5),
            "hexagon": lambda: self.make_ngon(6),
            "heptagon": lambda: self.make_ngon(7),
            "octagon": lambda: self.make_ngon(8),
            "nonagon": lambda: self.make_ngon(9),
            "decagon": lambda: self.make_ngon(10),
        }

    def calculate_ngon_vertices(
        self, center_x: int, center_y: int, radius: float, sides: int
    ) -> List[Tuple[float, float]]:
        """
        Calculates the vertices of a regular polygon.

        Given the center coordinates, radius, and number of sides, this method calculates
        the vertices of a regular polygon centered at the given point.

        Args:
            center_x (int): The x-coordinate of the polygon's center.
            center_y (int): The y-coordinate of the polygon's center.
            radius (float): The radius of the circumcircle of the polygon.
            sides (int): The number of sides (and vertices) of the polygon.

        Returns:
            list[tuple]: A list of vertices, where each vertex is a tuple (x, y).
        """
        vertices = []
        for i in range(sides):
            angle = math.radians((360 / sides) * i - 90)
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            vertices.append((x, y))
        return vertices

    def make_ngon(self, sides: int) -> Image.Image:
        """
        Generates an image of a regular polygon with a specified number of sides.

        Args:
            sides (int): The number of sides of the polygon.

        Returns:
            PIL.Image.Image: A PIL.Image.Image object containing the drawn polygon.
        """
        img = Image.new("RGBA", (self.size, self.size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        center_x, center_y = self.size // 2, self.size // 2
        radius = 200 * 0.4
        ngon_vertices = self.calculate_ngon_vertices(center_x, center_y, radius, sides)
        draw.polygon(ngon_vertices, fill=(255, 255, 255, 255))
        return img

    def make_rectangle(self) -> Image.Image:
        """
        Generates an image of a rectangle.

        Returns:
            PIL.Image.Image: A PIL.Image.Image object containing the drawn rectangle.
        """
        img = Image.new("RGBA", (self.size, self.size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rectangle(
            [0.25 * 200, 0.25 * 200, 0.75 * 200, 0.75 * 200],
            fill=(255, 255, 255, 255),
        )
        return img

    def make_circle(self) -> Image.Image:
        """
        Generates an image of a circle.

        Returns:
            PIL.Image.Image: A PIL.Image.Image object containing the drawn circle.
        """
        img = Image.new("RGBA", (self.size, self.size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.ellipse(
            [0.25 * self.size, 0.25 * self.size, 0.75 * self.size, 0.75 * self.size],
            fill=(255, 255, 255, 255),
        )
        return img

    def make_ellipse(self) -> Image.Image:
        """
        Generates an image of an ellipse.

        Returns:
            PIL.Image.Image: An image object containing the drawn ellipse.
        """
        img = Image.new("RGBA", (self.size, self.size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.ellipse(
            [0.125 * self.size, 0.25 * self.size, 0.875 * self.size, 0.75 * self.size],
            fill=(255, 255, 255, 255),
        )
        return img

    def get_shape(
        self,
        name: Optional[str] = None,
        color: Optional[Union[str, Tuple[int, int, int, int]]] = None,
    ) -> Tuple[Image.Image, str]:
        """
        Retrieves and returns an image of a specified geometric shape in a specified color.

        If no shape name is specified, a shape is randomly selected from the available shapes.
        The specified color can be a color name or an RGBA tuple. If no color is specified,
        the shape is generated in black by default.

        Args:
            name (str, optional): The name of the shape to generate. Defaults to a random shape if None.
            color (str | tuple, optional): The color of the shape, specified as a color name or RGBA tuple.
                Defaults to black if None or invalid.

        Returns:
            tuple[PIL.Image.Image, str]: A tuple containing an PIL.Image.Image object containing the drawn shape,
                and the name of the generated shape.
        """
        shapes = self.geometric_shapes()
        if name is None:
            name = random.choice(self.shape_names)
        shape_image = shapes[name]()
        color_rgba = self.validate_color(color) if color else (255, 255, 255, 255)
        shape_image_colored = self.apply_color_fill(shape_image, color_rgba)
        return shape_image_colored, name


class DinosaurShapeGenerator(ForegroundGenerator):
    """
    Generates and manipulates images of dinosaurs with customizable colors.

    This class fetches dinosaur images with transparent backgrounds from Wikimedia Commons,
    enabling the generation of images for supervised learning datasets. It supports
    customizing the color of dinosaurs post-download.

    Attributes:
        dino_meta_data (list): Metadata for the fetched dinosaur images.
        dino_dict (dict): Maps dinosaur names to their Image objects.
        shape_names (list): List of all dinosaur names that we can use for sampling.
        shape_id_map (dict): Maps an integer to each name in shape_names list
    """

    def __init__(self, meta_data_source: str = "local") -> None:
        """
        Initializes a DinosaurShapeGenerator object and downloads dinosaur images for local use.

        Downloads images from Wikimedia Commons and prepares them for generating datasets,
        storing them locally for efficient access.
        """
        super().__init__()
        self._url = "https://commons.wikimedia.org/wiki/Category:Dinosaurs_on_transparent_background"
        self._data_folder = os.path.join(self.get_data_path(), "dinosaur_images")
        if not os.path.exists(self._data_folder):
            os.makedirs(self._data_folder)
            meta_data_source = "url"
        self.meta_data_source = meta_data_source
        self.dino_meta_data = self.dino_image_metadata(meta_data_source)
        self.dino_dict = dict()
        self.load_all_dinos()
        self.shape_names = list(self.dino_dict.keys())
        self.shape_id_map = {
            shape: index for index, shape in enumerate(self.shape_names)
        }

    def clean_dino_name_from_URL(self, url: str) -> str:
        """
        Extracts and cleans the dinosaur name from a given URL.

        Parses the URL to extract the dinosaur name, removing URL encoding
        and invalid filename characters.

        Args:
            url (str): URL containing the dinosaur name.

        Returns:
            str: Cleaned dinosaur name suitable for filenames.
        """
        last_part = url.split("/")[-1]
        decoded_part = urllib.parse.unquote(last_part)
        name_with_prefix = decoded_part.split("-")[-1]
        name = name_with_prefix.rsplit(".", 1)[0]
        name = re.sub(
            r'[<>:"/\\|?*]', "", name
        )  # Remove characters not allowed in filenames
        return name

    def dino_image_metadata(
        self, meta_data_source: str
    ) -> List[Dict[str, Union[str, int]]]:
        """
        Fetches metadata for dinosaur images from the specified URL.

        Returns:
            list[dict]: A list of dictionaries, each containing metadata for a dinosaur image.
        """
        if meta_data_source == "url":
            response = requests.get(self._url)

            if response.status_code == 200:
                content_path = os.path.join(
                    self._data_folder, "content_snapshot" + ".txt"
                )
                with open(content_path, "w") as file:
                    file.write(f"{response.content}")
                soup = BeautifulSoup(response.content, "html.parser")
                image_tags = soup.find_all("img")
                image_info = [
                    {
                        "label": self.clean_dino_name_from_URL(img["src"]),
                        "thumbnail": img["src"],
                        "width": img["width"],
                        "height": img["height"],
                    }
                    for img in image_tags
                    if "src" in img.attrs and "class" not in img.attrs
                ]
                image_info = image_info[:-3]
            return image_info
        else:
            content_path = os.path.join(self._data_folder, "content_snapshot" + ".txt")

            with open(content_path, "r") as file:
                content = file.read()

            soup = BeautifulSoup(content, "html.parser")
            image_tags = soup.find_all("img")
            image_info = [
                {
                    "label": self.clean_dino_name_from_URL(img["src"]),
                    "thumbnail": img["src"],
                    "width": img["width"],
                    "height": img["height"],
                }
                for img in image_tags
                if "src" in img.attrs and "class" not in img.attrs
            ]
            image_info = image_info[:-3]
            return image_info

    def load_dino_image(self, url: str, save_path: str) -> Optional[Image.Image]:
        """
        Downloads a dinosaur image from a URL and saves it locally.

        Args:
            url (str): URL of the image to be downloaded.
            save_path (str): Local file path to save the downloaded image.

        Returns:
            PIL.Image.Image | NoneType: The downloaded PIL.Image.Image object, or None if the download fails.
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            + " AppleWebKit/537.36 (KHTML, like Gecko) "
            + "Chrome/58.0.3029.110 Safari/537.3"
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            image.save(save_path)
            return image
        else:
            print("Failed to download image: ", response.status_code)
            return None

    def load_all_dinos(self) -> None:
        """
        Loads all available dinosaur images into memory from the local storage.

        Ensures that images are downloaded based on metadata if they are not already present
        locally, making them readily available in memory for image generation.
        """
        N = len(self.dino_meta_data)
        download_required = False

        # Check if any download is required
        for meta in self.dino_meta_data:
            dino_lbl = meta["label"]
            img_path = os.path.join(self._data_folder, dino_lbl + ".png")
            if not os.path.exists(img_path):
                download_required = True
                break

        if download_required:
            progress_bar = tqdm(total=N, desc="Downloading Dinosaurs")
            for i, meta in enumerate(self.dino_meta_data):
                dino_lbl = meta["label"]
                img_url = meta["thumbnail"]
                img_path = os.path.join(self._data_folder, dino_lbl + ".png")
                if not os.path.exists(img_path):
                    self.load_dino_image(img_url, img_path)
                if not dino_lbl in self.dino_dict:  # Check if image is already loaded
                    self.dino_dict[dino_lbl] = Image.open(img_path)

                # Update progress bar every 10 dinosaurs
                if (i + 1) % 10 == 0 or i == N - 1:
                    progress_bar.update(10 if i < N - 1 else N % 10)
            progress_bar.close()
        else:
            for meta in self.dino_meta_data:
                dino_lbl = meta["label"]
                img_path = os.path.join(self._data_folder, dino_lbl + ".png")
                self.dino_dict[dino_lbl] = Image.open(img_path)

    def get_shape(
        self,
        name: Optional[str] = None,
        color: Optional[Union[str, Tuple[int, int, int, int]]] = None,
    ) -> Tuple[Image.Image, str]:
        """
        Retrieves a dinosaur image by name with an optional color overlay.

        If no name is specified, selects a dinosaur randomly. Optionally applies a color
        to the dinosaur image before returning.

        Args:
            name (str, optional): Name of the dinosaur. Selects randomly if None.
                Defaults to None.
            color (str | tuple, optional): color to apply. Uses predefined or RGBA tuple.
                Defaults to None.

        Returns:
            tuple[PIL.Image.Image, str]: A tuple containing the colored PIL.Image.Image object of the dinosaur and its name.
        """
        if name is None:
            name = random.choice(list(self.dino_dict.keys()))
        dino_image = self.dino_dict[name]

        if color:
            color = self.validate_color(color)
            dino_image = self.apply_color_fill(dino_image, color)
        return dino_image, name

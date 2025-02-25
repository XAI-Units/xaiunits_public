import pytest
from xaiunits.datagenerator.foregrounds import ForegroundGenerator
from xaiunits.datagenerator.foregrounds import GeometricShapeGenerator
from xaiunits.datagenerator.foregrounds import DinosaurShapeGenerator
from unittest.mock import patch
from PIL import Image, ImageDraw
import math
import bs4
import requests
import os


class TestForegroundGenerator:
    def test_abstract_class(self):
        """Tests that the class is an Abstract base class."""
        with pytest.raises(TypeError):
            fg = ForegroundGenerator()
            # Should raise TypeError because ABC can't be instantiated directly.

    def test_init(self):
        """Tests the __init__ method."""
        with patch.object(
            ForegroundGenerator, "get_data_path", return_value="/fake/path"
        ):
            shapes = GeometricShapeGenerator()
            assert shapes._data_path == "/fake/path"
            assert shapes._colors_rgba["red"] == (255, 0, 0, 255)

    def test_validate_color_rgb(self):
        """Tests the validate_color method given tuple input of RBGA."""
        shapes = GeometricShapeGenerator()
        assert shapes.validate_color((0, 128, 128, 255)) == (0, 128, 128, 255)

    def test_validate_color_str(self):
        """Tests the validate_color method given str input."""
        shapes = GeometricShapeGenerator()
        assert shapes.validate_color("red") == (255, 0, 0, 255)
        assert shapes.validate_color("green") == (0, 255, 0, 255)
        assert shapes.validate_color("blue") == (0, 0, 255, 255)
        assert shapes.validate_color("yellow") == (255, 255, 0, 255)
        assert shapes.validate_color("black") == (0, 0, 0, 255)
        assert shapes.validate_color("white") == (255, 255, 255, 255)
        assert shapes.validate_color("orange") == (255, 165, 0, 255)
        assert shapes.validate_color("purple") == (128, 0, 128, 255)

    def test_validate_color_error(self):
        """Tests whether the validate_color method raises error."""
        shapes = GeometricShapeGenerator()
        with pytest.raises(ValueError):
            shapes.validate_color("not_a_color")
        with pytest.raises(ValueError):
            shapes.validate_color((256, -1, 300, 256))

    def test_apply_color_fill_none_color(self):
        """Tests the apply_color_fill method given NoneType color."""
        shapes = GeometricShapeGenerator()
        image = Image.new("RGBA", (10, 10), "white")
        result_image = shapes.apply_color_fill(image, None)
        assert image == result_image  # Should be the same as no fill applied

    def test_apply_color_fill_not_rbga_image(self):
        """
        Tests the apply_color_fill method given image that is not
        already in rbga.
        """
        shapes = GeometricShapeGenerator()
        image = Image.new("RGB", (10, 10), "white")
        colored_image = shapes.apply_color_fill(image, (255, 0, 0, 255))
        assert colored_image.mode == "RGBA"

    def test_apply_color_fill_rbga_image(self):
        """
        Tests the apply_color_fill method given image that is
        already in rbga.
        """
        shapes = GeometricShapeGenerator()
        image = Image.new("RGBA", (10, 10), "white")
        colored_image = shapes.apply_color_fill(image, (255, 0, 0, 255))
        assert colored_image.mode == "RGBA"

    def test_get_data_path_existing_dir(self):
        """
        Tests the get_data_path method when the 'data' directory
        exists.
        """
        shapes = GeometricShapeGenerator()
        with (
            patch("os.path.exists", return_value=True),
            patch("os.makedirs") as mock_mkdir,
        ):
            path = shapes.get_data_path()
            mock_mkdir.assert_not_called()  # makedirs should not be called if directory exists

    def test_get_data_path_not_existing_dir(self):
        """
        Tests the get_data_path method when the 'data' directory
        does not exist.
        """
        shapes = GeometricShapeGenerator()
        with (
            patch("os.path.exists", return_value=False),
            patch("os.makedirs") as mock_mkdir,
        ):
            path = shapes.get_data_path()
            mock_mkdir.assert_called_once()


# Test the get_shape method can be correctly overwritten
class MockForegroundGenerator(ForegroundGenerator):
    def get_shape(self, name=None, color=None):
        # Create a simple 100x100 black image for testing
        img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rectangle([(10, 10), (90, 90)], fill="white", outline="black")
        return img, "test_shape"


@pytest.fixture
def test_gen():
    return MockForegroundGenerator()


def test_get_shape(test_gen):
    img, shape_name = test_gen.get_shape()
    assert isinstance(img, Image.Image), "The method should return a PIL Image object."
    assert img.size == (100, 100), "The image should be of the expected size."
    assert shape_name == "test_shape", "The shape name should be as expected."


class TestGeometricShapeGenerator:
    def test_subclass(self):
        """Tests that the class is a subclass of ForegroundGenerator."""
        assert issubclass(GeometricShapeGenerator, ForegroundGenerator)

    def test_init(self):
        """Tests the __init__ method."""
        shapes = GeometricShapeGenerator(size=250)
        assert shapes.size == 250
        assert isinstance(shapes.shape_names, list)
        assert "circle" in shapes.shape_names

    def test_default_initialization(self):
        generator = GeometricShapeGenerator()
        assert generator.size == 200, "Default size should be 200 pixels."
        assert "red" in generator._colors_rgba, "Default colors should include 'red'."

    def test_geometric_shapes(self):
        """Tests the correctness of the geometric_shapes method."""
        generator = GeometricShapeGenerator()
        shapes = generator.geometric_shapes()
        assert isinstance(shapes, dict)
        assert "triangle" in shapes
        assert callable(shapes["triangle"])

    def test_calculate_ngon_vertices_correctness_triangle(self):
        """Tests the correctness of the calculate_ngon_vertices for triangle."""
        generator = GeometricShapeGenerator()
        vertices = generator.calculate_ngon_vertices(100, 100, 50, 3)
        expected_vertices = [(100, 50), (143.3, 125), (56.7, 125)]
        for v, ev in zip(vertices, expected_vertices):
            assert math.isclose(v[0], ev[0], abs_tol=0.1) and math.isclose(
                v[1], ev[1], abs_tol=0.1
            )

    def test_calculate_ngon_vertices_num_vertices(self):
        """
        Tests if the calculate_ngon_vertices returns same number of vertices
        as number of sides inputed.
        """
        generator = GeometricShapeGenerator()
        for sides in range(
            3, 11
        ):  # Test for different polygons from triangle to decagon
            vertices = generator.calculate_ngon_vertices(100, 100, 50, sides)
            assert len(vertices) == sides

    def test_make_ngon_image_size(self):
        """
        Tests the make_ngon method returns a Image object with correct
        specifications.
        """
        generator = GeometricShapeGenerator()
        img = generator.make_ngon(6)
        assert img.size == (generator.size, generator.size)
        assert img.mode == "RGBA"

    def test_make_ngon_single_point(self):
        """
        Tests the correctness of the make_ngon method when requiring only a
        single point. This should fail. If a point is needed, use a circle.
        """
        generator = GeometricShapeGenerator()
        with pytest.raises(TypeError):
            generator.make_ngon(1)

    def test_make_rectangle_image_size(self):
        """
        Tests the make_rectangle method returns a Image object with correct
        specifications.
        """
        generator = GeometricShapeGenerator()
        img = generator.make_rectangle()
        assert img.size == (generator.size, generator.size)
        assert img.mode == "RGBA"

    def test_make_circle_image_size(self):
        """
        Tests the make_circle method returns a Image object with correct
        specifications.
        """
        generator = GeometricShapeGenerator()
        img = generator.make_circle()
        assert img.size == (generator.size, generator.size)
        assert img.mode == "RGBA"

    def test_make_ellipse_image_size(self):
        """
        Tests the make_ellipse method returns a Image object with correct
        specifications.
        """
        generator = GeometricShapeGenerator()
        img = generator.make_ellipse()
        assert img.size == (generator.size, generator.size)
        assert img.mode == "RGBA"

    def test_get_shape_name(self):
        """Tests the name output of get_shape method is correct."""
        generator = GeometricShapeGenerator()
        img, name = generator.get_shape(name="circle")
        assert name == "circle"

    def test_get_shape_none_name(self):
        """Tests the name output of get_shape method is from selected pool."""
        generator = GeometricShapeGenerator()
        img, name = generator.get_shape()
        assert name in generator.shape_names
        assert isinstance(img, Image.Image)

    def test_get_shape_img_color(self):
        """
        Tests the color of one of the pixel of Image output from get_shape
        method.
        """
        generator = GeometricShapeGenerator()
        img, name = generator.get_shape(name="circle", color="red")
        img.load()
        color = img.getpixel((generator.size // 2, generator.size // 2))
        assert color == (255, 0, 0, 255)


class TestDinosaurShapeGenerator:
    @pytest.fixture
    def generator(self):
        with patch.object(DinosaurShapeGenerator, "load_all_dinos", return_value=None):
            return DinosaurShapeGenerator()

    def test_subclass(self):
        """Tests that the class is a subclass of ForegroundGenerator."""
        assert issubclass(DinosaurShapeGenerator, ForegroundGenerator)

    def test_init(self, generator):
        """Tests the __init__ method."""
        assert (
            generator._url
            == "https://commons.wikimedia.org/wiki/Category:Dinosaurs_with_transparent_background"
        )
        assert "dinosaur_images" in generator._data_folder

    def test_clean_dino_name_from_URL_no_url_encoding(self, generator):
        """
        Tests the clean_dino_name_from_URL method if its output contains
        url encoding.
        """
        url = "https://example.com/Some%20Encoded%20Name.png"
        cleaned_name = generator.clean_dino_name_from_URL(url)
        assert "Some Encoded Name" == cleaned_name

    def test_clean_dino_name_from_URL_no_illegal_char(self, generator):
        """
        Tests the clean_dino_name_from_URL method if its output contains
        illegal characters.
        """
        url = "https://example.com/Some<Name>.png"
        cleaned_name = generator.clean_dino_name_from_URL(url)
        assert "SomeName" == cleaned_name

    def test_clean_dino_name_from_URL_correctness(self, generator):
        """
        Tests the correctness of the clean_dino_name_from_URL method.
        """
        url = "https://example.com/images/Dino-Rex.png"
        cleaned_name = generator.clean_dino_name_from_URL(url)
        assert cleaned_name == "Rex"

    def test_dino_image_metadata_type(self, generator):
        """
        Verifies that the 'dino_image_metadata' method produces a list of dictionaries, each containing
        metadata for dinosaur images. The method under test fetches and parses HTML content from a
        specified URL to extract image details.

        The test checks:
        1. The method returns a list.
        2. Each item in the list is a dictionary containing the keys 'label' and 'thumbnail'.

        The test uses mock objects to:
        - Simulate HTTP responses for the image metadata fetch.
        - Parse HTML content to find image elements.
        - Ensure that the image elements are processed to extract metadata.

        Steps:
        - A mock for 'requests.get' simulates a successful HTTP response with HTML content.
        - 'BeautifulSoup.find_all' is patched to return a list of mocked img tags, crafted from a minimal
          HTML structure, ensuring that BeautifulSoup can parse and handle it as expected in the real function.
        - The function's output is then asserted to be a list containing dictionaries structured as expected.
        """
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.content = b'<html><body><img src="image1.png" width="100" height="100"/></body></html>'
            with patch(
                "bs4.BeautifulSoup.__init__", return_value=None
            ) as mock_soup_init:
                with patch("bs4.BeautifulSoup.find_all") as mock_find_all:
                    mock_tag = bs4.BeautifulSoup(
                        '<img src="image1.png" width="100" height="100"/>',
                        "html.parser",
                    ).find("img")
                    mock_find_all.return_value = [mock_tag]

                    metadata = generator.dino_image_metadata()
                    assert isinstance(metadata, list)
                    # Ensure that metadata list contains dictionaries with expected keys
                    assert all(isinstance(item, dict) for item in metadata)
                    assert all(
                        "label" in item and "thumbnail" in item for item in metadata
                    )

    def test_load_dino_image_type(self, generator):
        """
        Tests the load_dino_image method by actually downloading an image, ensuring it returns
        a PIL Image object and saves it locally, while gracefully handling HTTP errors.
        """
        test_url = (
            "https://upload.wikimedia.org/wikipedia/commons/8/8f/Aletopelta_UDL.png"
        )
        save_path = os.path.join(generator._data_folder, "test_dino_image.png")
        os.makedirs(generator._data_folder, exist_ok=True)

        try:
            image = generator.load_dino_image(test_url, save_path)
            assert isinstance(
                image, Image.Image
            ), "The returned object should be an instance of PIL.Image.Image"
            assert os.path.exists(
                save_path
            ), "The image file should exist at the specified path"
            with Image.open(save_path) as img:
                assert img.format == "PNG", "The image should be saved in PNG format"

        except requests.exceptions.HTTPError as e:
            assert (
                e.response.status_code == 404
            ), "Should handle not found error gracefully"
        except Exception as e:
            # General exception catch if unexpected errors occur
            assert False, f"An unexpected error occurred: {e}"

        finally:
            if os.path.exists(save_path):
                os.remove(save_path)

    def test_load_dino_image_none(self, generator):
        """
        Tests the load_dino_image method when image cannot be laoded.
        """
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 404
            image = generator.load_dino_image(
                "http://fakeurl.com/image.png", "/fake/path"
            )
            assert image is None

    def test_load_all_dinos(self, generator):
        """
        Tests if the load_all_dinos method updates the corresponding attribute.
        """
        with patch.object(
            DinosaurShapeGenerator,
            "load_dino_image",
            return_value=Image.new("RGBA", (100, 100)),
        ):
            generator.load_all_dinos()
            assert len(generator.dino_dict) > 0

    def test_get_shape_name(self, generator):
        """Tests the name output of get_shape method is correct."""
        generator.dino_dict = {"T-Rex": Image.new("RGBA", (100, 100))}
        img, name = generator.get_shape(name="T-Rex")
        assert name == "T-Rex"

    def test_get_shape_none_name(self, generator):
        """Tests the name output of get_shape method is from selected pool."""
        generator.dino_dict = {"T-Rex": Image.new("RGBA", (100, 100))}
        img, name = generator.get_shape()
        assert name in generator.dino_dict

    def test_get_shape_img_color(self, generator):
        """
        Tests the color of one of the pixel of Image output from get_shape
        method.
        """
        generator.dino_dict = {"T-Rex": Image.new("RGBA", (100, 100), "white")}
        img, name = generator.get_shape(name="T-Rex", color="red")
        assert img.getpixel((10, 10)) == (255, 0, 0, 255)


# this block is only executed when you run the file directly as `python this_script.py`
# but you should be running `pytest` or `pytest this_script.py`
if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))

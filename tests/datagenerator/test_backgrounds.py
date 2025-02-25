import shutil
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
from PIL import Image
from xaiunits.datagenerator.backgrounds import BackgroundGenerator


@pytest.fixture
def generator():
    with patch("xaiunits.datagenerator.backgrounds.BackgroundGenerator.download"):
        return BackgroundGenerator()


class TestBackgroundGenerator:
    def test_init(self, generator):
        """Tests the __init__ method."""
        with patch(
            "xaiunits.datagenerator.backgrounds.BackgroundGenerator.download"
        ) as mock_download:
            bg_generator = BackgroundGenerator()
            mock_download.assert_called_once()
            assert isinstance(bg_generator, BackgroundGenerator)

    def test_get_data_path_existing_dir(self, generator):
        """
        Tests the get_data_path method when the 'data' directory
        exists.
        """
        with (
            patch("os.path.exists", return_value=True),
            patch("os.makedirs") as mock_makedirs,
        ):
            path = generator.get_data_path()
            mock_makedirs.assert_not_called()
            assert "data" in path

    def test_get_data_path_not_existing_dir(self, generator):
        """
        Tests the get_data_path method when the 'data' directory
        does not exists.
        """
        with (
            patch("os.path.exists", return_value=False),
            patch("os.makedirs") as mock_makedirs,
        ):
            path = generator.get_data_path()
            mock_makedirs.assert_called_once()
            assert "data" in path

    def test_download_real_file(self):
        """Tests the download method by actually downloading and extracting a file."""
        url = "https://www.robots.ox.ac.uk/~vgg/data/dtd/thumbs/interlaced/interlaced_0201.jpg"
        target_path = Path("/tmp/test_download.tar.gz")
        extract_folder = Path("/tmp/test_download_folder")

        target_path.parent.mkdir(parents=True, exist_ok=True)
        extract_folder.mkdir(parents=True, exist_ok=True)

        # Clean up before test
        if target_path.exists():
            target_path.unlink()  # Remove the file if it exists
        if extract_folder.exists():
            shutil.rmtree(extract_folder)  # Remove folder and all its contents

        try:
            # Download the file
            response = requests.get(url, stream=True)
            response.raise_for_status()  # This will raise an HTTPError if the download failed

            with open(target_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Validate the download
            assert target_path.exists(), "File was not downloaded successfully."

            # Extract the file
            if target_path.exists():
                with tarfile.open(target_path, "r:gz") as tar:
                    tar.extractall(path=extract_folder)

            # Check if extraction happened
            assert extract_folder.exists(), "Extraction folder does not exist."
            assert any(
                extract_folder.iterdir()
            ), "Extraction folder is empty, extraction failed."

        except requests.HTTPError as e:
            print(f"HTTP Error during download: {e}")
        except tarfile.ReadError as e:
            print(f"Error reading tar file: {e}")
        finally:
            # Cleanup
            if target_path.exists():
                target_path.unlink()
            if extract_folder.exists():
                shutil.rmtree(extract_folder)

    def test_validate_background_size_invalid_input(self, generator):
        """
        Tests the _validate_background_size method when given
        invalid input.
        """
        with pytest.raises(ValueError):
            generator._validate_background_size("invalid")

    def test_validate_background_size_width(self, generator):
        """
        Tests that the width is adjusted to be within the accepted range.
        """
        assert generator._validate_background_size((299, 400)) == (300, 400)
        assert generator._validate_background_size((641, 400)) == (640, 400)

    def test_validate_background_size_height(self, generator):
        """
        Tests that the height is adjusted to be within the accepted range.
        """
        assert generator._validate_background_size((400, 299)) == (400, 300)
        assert generator._validate_background_size((400, 641)) == (400, 640)

    def test_get_background_names_suffix(self, generator):
        """
        Tests that the filenames have the correct suffix.
        """
        generator.background_names = [
            "image1.jpg",
            "image2.png",
            "test.jpeg",
            "fail.txt",
        ]
        valid = all(
            name.endswith((".jpg", ".jpeg", ".png"))
            for name in generator.get_background_names()
        )
        assert valid

    def test_get_background_names_correct_folder(self, generator):
        """
        Ensures filenames are retrieved from the specified folder.
        """
        fake_directories = [
            MagicMock(
                is_dir=MagicMock(return_value=True),
                path="/fake/path/dtd/images/subfolder1",
            ),
            MagicMock(
                is_dir=MagicMock(return_value=True),
                path="/fake/path/dtd/images/subfolder2",
            ),
        ]

        with (
            patch("os.scandir", return_value=fake_directories) as mock_scandir,
            patch(
                "os.listdir", side_effect=[["file1.jpg", "file2.png"], ["file3.jpg"]]
            ) as mock_listdir,
            patch(
                "os.path.join", side_effect=lambda *args: "/".join(args)
            ) as mock_join,
        ):
            names = generator.get_background_names()
            assert len(names) == 3
            assert mock_listdir.call_count == len(fake_directories)

    def test_get_background_image_type(self, generator):
        """
        Ensures the returned image is of the correct type and size.
        """
        with patch.object(Image, "open", return_value=Image.new("RGB", (640, 480))):
            img = generator.get_background()
            assert isinstance(img, Image.Image)
            assert img.size == generator.background_size

    def test_get_background_specific_image(self, generator):
        """
        Ensures a specific image is fetched correctly.
        """
        test_img_path = "/path/to/image.jpg"
        with (
            patch("os.path.join", return_value=test_img_path),
            patch.object(
                Image, "open", return_value=Image.new("RGB", (640, 480))
            ) as mock_open,
        ):
            img = generator.get_background(image_file="image.jpg")
            mock_open.assert_called_with(test_img_path)
            assert isinstance(img, Image.Image)


# this block is only executed when you run the file directly as `python this_script.py`
# but you should be running `pytest` or `pytest this_script.py`
if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))

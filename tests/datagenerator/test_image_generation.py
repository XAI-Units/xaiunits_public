import pytest
from PIL import Image
import torch
from collections import Counter
from xaiunits.datagenerator.foregrounds import GeometricShapeGenerator
from xaiunits.datagenerator.foregrounds import DinosaurShapeGenerator
from xaiunits.datagenerator.backgrounds import BackgroundGenerator
from xaiunits.datagenerator.image_generation import ImageBuilder
from xaiunits.datagenerator.image_generation import ImageDataset
from xaiunits.datagenerator.image_generation import BalancedImageDataset
from xaiunits.datagenerator.image_generation import ImbalancedImageDataset


@pytest.fixture
def image_builder():
    yield ImageBuilder()


@pytest.fixture
def image_builder_with_default_bg():
    return ImageBuilder(default_background_imagefile="cracked_0049.jpg")


@pytest.fixture
def setup_image_dataset():
    return ImageDataset()


@pytest.fixture
def balanced_image_dataset():
    return BalancedImageDataset()


@pytest.fixture
def ordered_dataset():
    # Generate a dataset with a known order
    dataset = BalancedImageDataset(
        shapes=3, backgrounds=3, n_variants=2, shuffled=False
    )
    original_order = [
        (id(img), id(lbl)) for img, lbl in zip(dataset.samples, dataset.labels)
    ]
    return dataset, original_order


@pytest.fixture
def imbalanced_image_dataset():
    return ImbalancedImageDataset(
        backgrounds=5, shapes=5, n_variants=10, shape_colors="red", imbalance=0.5
    )


class TestImageBuilder:
    # Init
    def test_init_geometric(self, image_builder):
        """Tests the __init__ method when the shape_type is geometric."""
        builder = ImageBuilder(shape_type="geometric")
        assert isinstance(builder.shape_gen, GeometricShapeGenerator)
        assert builder.shape_type == "geometric"

    def test_init_dinosaurs(self, image_builder):
        """Tests the __init__ method when the shape_type is dinosaurs."""
        builder = ImageBuilder(shape_type="dinosaurs")
        assert isinstance(builder.shape_gen, DinosaurShapeGenerator)
        assert builder.shape_type == "dinosaurs"

    def test_init_error(self):
        """Tests the __init__ method when an error is expected to be raised"""
        with pytest.raises(ValueError):
            ImageBuilder(shape_type="unknown")

    # Overlay resize
    def test_resize_overlay_to_background_same_aspect_ratio(self, image_builder):
        """
        Tests that the output from the resize_overlay_to_background method
        has the same aspect ratio as the background image.
        """
        background = Image.new("RGBA", (512, 512), "black")
        overlay = Image.new("RGBA", (100, 200), "red")
        resized = image_builder.resize_overlay_to_background(background, overlay)
        expected_ratio = overlay.width / overlay.height
        resulting_ratio = resized.width / resized.height
        assert expected_ratio == pytest.approx(resulting_ratio)

    def test_resize_overlay_to_background_same_image(self, image_builder):
        """
        Tests that the output image can be resized back to the original
        image.
        """
        original_overlay = Image.new("RGBA", (100, 100), "blue")
        background = Image.new("RGBA", (512, 512), "green")
        resized_overlay = image_builder.resize_overlay_to_background(
            background, original_overlay
        )
        resized_back_overlay = resized_overlay.resize(
            original_overlay.size, Image.Resampling.LANCZOS
        )
        assert (
            original_overlay.size == resized_back_overlay.size
        ), "The resized-back image dimensions should match the original."
        # This histogram check is a basic check: there may be slight modifications in the resizing process
        assert (
            original_overlay.histogram() == resized_back_overlay.histogram()
        ), "The histograms should be similar."

    # Default backgrounds
    def test_default_background_used(self, image_builder_with_default_bg):
        """Tests that the specified default background is used for all samples."""
        image_builder_with_default_bg.back_gen = BackgroundGenerator()
        image_builder_with_default_bg.back_gen.background_names = [
            "cracked_0049.jpg",
            "cracked_0073.jpg",
        ]
        img, label, lbl_str, gt = image_builder_with_default_bg.generate_sample()
        # Check if the background used is the default
        assert (
            img.info["background"] == "cracked_0049.jpg"
        ), "Default background should be used"

    def test_random_background_used_when_no_default(self):
        """Tests that a random background is selected when no default is provided."""
        image_with_no_bg_default = ImageBuilder()
        # Setup background generator
        image_with_no_bg_default.back_gen = BackgroundGenerator()
        image_with_no_bg_default.back_gen.background_names = [
            "cracked_0061.jpg",
            "cracked_0062.jpg",
            "cracked_0063.jpg",
        ]
        backgrounds_used = set()
        # Generate multiple samples to check random selection of backgrounds
        for _ in range(10):
            img, label, lbl_str, gt = image_with_no_bg_default.generate_sample()
            backgrounds_used.add(img.info["background"])
        # Assuming random choice is properly random, this should not have the same background every time
        assert len(backgrounds_used) > 1, "Should use different backgrounds randomly"

    # Overlay scaling
    def test_extreme_overlay_scaling(self, image_builder):
        """Tests overlay scaling at extreme values."""
        background = Image.new("RGBA", (512, 512), "black")
        overlay = Image.new("RGBA", (100, 100), "red")
        # Test with very small scale
        image_builder.overlay_scale = 0.01
        resized_small = image_builder.resize_overlay_to_background(background, overlay)
        assert resized_small.size[0] == int(
            512 * 0.01
        ), "Width should match the scaled size"

        # Test with very large scale
        image_builder.overlay_scale = 2.0
        resized_large = image_builder.resize_overlay_to_background(background, overlay)
        assert resized_large.size[0] == int(
            512 * 2.0
        ), "Width should exceed background size for large scale"

    # Rotations
    def test_img_rotation_diff_size(self, image_builder):
        """
        Tests that the img_rotation method changes the size of the image.
        In img_rotation we use expand=True to increase the bounding box and
        avoid clipping the image. We don't want to cut corners off.
        """
        img = Image.new("RGBA", (100, 100), "blue")
        rotated_img = image_builder.img_rotation(img)
        assert img.size != rotated_img.size

    def test_rotation_zero_degrees(self, image_builder):
        """Test that an image rotated by 0 degrees remains unchanged."""
        original_img = Image.new("RGBA", (100, 100), "blue")
        rotated_img = image_builder.img_rotation(original_img, 0)
        assert (
            original_img.tobytes() == rotated_img.tobytes()
        ), "The image should remain unchanged with 0-degree rotation."

    def test_rotation_full_circle(self, image_builder):
        """Test that an image rotated by 360 degrees remains unchanged."""
        original_img = Image.new("RGBA", (100, 100), "blue")
        rotated_img = image_builder.img_rotation(original_img, 360)
        assert (
            original_img.tobytes() == rotated_img.tobytes()
        ), "The image should remain unchanged with 360-degree rotation."

    def test_rotation_ninety_degrees(self, image_builder):
        """Test that an image rotated by 90 degrees changes as expected."""
        original_img = Image.new("RGBA", (100, 200), "green")
        rotated_img = image_builder.img_rotation(original_img, 90)
        assert original_img.size == (
            rotated_img.height,
            rotated_img.width,
        ), "The image dimensions should be swapped with 90-degree rotation."

    def test_image_integrity_post_rotation(self, image_builder):
        """Check that the image does not lose any part post rotation."""
        original_img = Image.new("RGBA", (100, 100), "red")
        rotated_img = image_builder.img_rotation(original_img, 45)  # Arbitrary angle
        assert (
            rotated_img.size >= original_img.size
        ), "The rotated image should accommodate the entire original image."

    # Overlay positions
    def test_overlay_pos_center(self, image_builder):
        """
        Tests the correctness of the overlay_pos method when the overlay position
        is center.
        """
        background = Image.new("RGBA", (500, 500), "black")
        overlay = Image.new("RGBA", (100, 100), "red")
        x, y = image_builder.overlay_pos(background, overlay)
        assert x == (background.width - overlay.width) // 2
        assert y == (background.height - overlay.height) // 2

    def test_overlay_pos_random(self, image_builder):
        """
        Tests the output of the overlay_pos method is within the maximum bounds
        when the overlay position is random.
        """
        background = Image.new("RGBA", (500, 500), "black")
        overlay = Image.new("RGBA", (100, 100), "red")
        positions = [image_builder.overlay_pos(background, overlay) for _ in range(10)]
        for x, y in positions:
            assert 0 <= x <= background.width - overlay.width
            assert 0 <= y <= background.height - overlay.height

    def test_overlay_pos_error(self, image_builder):
        """Tests the overlay_pos method when the overlay position is invalid."""
        background = Image.new("RGBA", (512, 512), "green")
        overlay = Image.new("RGBA", (100, 100), "blue")
        image_builder.position = "left"  # An invalid position value
        with pytest.raises(ValueError) as excinfo:
            image_builder.overlay_pos(background, overlay)
        assert "Position must be either 'center' or 'random'" in str(
            excinfo.value
        ), "Expected ValueError with specific message about position"

    # Generate sample
    def test_generate_sample_type(self, image_builder):
        """Tests the type of the output from the generate_sample method."""
        image, label, lbl_string, gt = image_builder.generate_sample(
            name="circle"
        )  # Assuming 'circle' is a valid shape name
        assert isinstance(image, Image.Image)
        assert isinstance(gt, Image.Image)
        # Adjust assertion based on actual data type of label
        assert isinstance(label, int)
        assert isinstance(lbl_string, str)

    def test_generate_sample_consistency(self, image_builder):
        """
        Tests that the generate_sample method returns consistent results when called multiple times
        with the same parameters.
        """
        name = "circle"
        # Assume this is a valid background file
        background_imagefile = "cracked_0062.jpg"
        color = (255, 0, 0, 255)  # Red in RGBA
        img1, lbl1, lbl_string1, gt1 = image_builder.generate_sample(
            name=name, background_imagefile=background_imagefile, color=color
        )
        img2, lbl2, lbl_string2, gt2 = image_builder.generate_sample(
            name=name, background_imagefile=background_imagefile, color=color
        )
        assert lbl1 == lbl2, "Labels should be consistent across multiple calls"
        assert (
            lbl_string1 == lbl_string2
        ), "Label strings should be consistent across multiple calls"
        assert (
            img1.tobytes() == img2.tobytes()
        ), "Generated images should be consistent across multiple calls"
        assert (
            gt1.tobytes() == gt2.tobytes()
        ), "Generated ground truths should be consistent across multiple calls"

    # Colors
    def test_color_validity(self):
        """Tests that invalid colors are correctly handled by raising errors."""
        with pytest.raises(ValueError) as e_type:
            ImageBuilder(color="invalid_shape_color").generate_sample()
            assert (
                "Color name must be a valid key in the color dictionary self._colors_rgba"
                in str(e_type.value)
            )

        with pytest.raises(ValueError) as e_type:
            ImageBuilder(color=(0, -1, 0, 0)).generate_sample()
            assert (
                "color must be a lowercase string or 4-d tuple of int between 0 and 255"
                in str(e_type.value)
            )

        assert ImageBuilder(color=(0, 0, 0, 0)).color == (0, 0, 0, 0)
        assert ImageBuilder(color=(255, 255, 255, 255)).color == (255, 255, 255, 255)


class TestImageDataset:
    def test_subclass(self):
        """
        Tests that the class is a subclass of torch.utils.data.Dataset.
        """
        assert issubclass(
            ImageDataset, torch.utils.data.Dataset
        ), "ImageDataset should be a subclass of torch.utils.data.Dataset."

    def test_init(self):
        """Tests the __init__ method."""
        dataset = ImageDataset()
        assert isinstance(
            dataset, ImageDataset
        ), "Initialization should create an instance of ImageDataset."
        assert len(dataset.backgrounds) > 0, "Backgrounds should be initialized."
        assert len(dataset.shapes) > 0, "Shapes should be initialized."

    def test_prepare_backgrounds_int(self):
        """
        Tests the output from the _prepare_backgrounds method given
        integer input is generated from the BackgroundGenerator.
        """
        dataset = ImageDataset(backgrounds=5)
        assert len(dataset.backgrounds) == 5, "Should prepare 5 backgrounds."

    def test_prepare_shape_color(self):
        """Tests the _prepare_colors method given list input."""
        dataset = ImageDataset(shape_colors=["red", "blue", "green"])
        assert dataset.shape_colors == [
            (255, 0, 0, 255),
            (0, 0, 255, 255),
            (0, 255, 0, 255),
        ], "Should correctly set the shape colors from a list."

    def test_prepare_shape_color_with_integer(self, setup_image_dataset):
        try:
            colors = setup_image_dataset._prepare_shape_color(
                2
            )  # Expect 2 random colors
            assert len(colors) == 2, "Should return a list of two colors"
            for color in colors:
                assert (
                    isinstance(color, tuple) and len(color) == 4
                ), "Each color must be a valid RGBA tuple"
        except Exception as e:
            pytest.fail(f"An unexpected exception occurred: {str(e)}")

    def test_prepare_shape_color_with_list(self, setup_image_dataset):
        try:
            colors = setup_image_dataset._prepare_shape_color(["red", (0, 0, 255, 255)])
            assert (
                len(colors) == 2
            ), "Should handle a list of color names and RGBA tuples"
            assert isinstance(colors[0], tuple) and colors[1] == (
                0,
                0,
                255,
                255,
            ), "Should validate color names and tuples"
        except Exception as e:
            pytest.fail(f"An unexpected exception occurred: {str(e)}")

    def test_prepare_shape_color_with_invalid_input(self, setup_image_dataset):
        # Test with invalid input types
        with pytest.raises(ValueError):
            # Invalid because a single string is not a list or int
            setup_image_dataset._prepare_shape_color("invalid")

        with pytest.raises(ValueError):
            # Assuming 999 exceeds available colors
            setup_image_dataset._prepare_shape_color(999)

        with pytest.raises(ValueError):
            # Assuming this color name does not exist
            setup_image_dataset._prepare_shape_color(["non_existent_color_name"])

    def test_prepare_shapes_geometric_int(self):
        """
        Tests the _prepare_shapes method given geometric shape type and
        integer shapes inputs.
        """
        dataset = ImageDataset(shape_type="geometric", shapes=2)
        assert len(dataset.shapes) == 2, "Should prepare 2 geometric shapes."

    def test_prepare_shapes_geometric_list(self):
        """
        Tests the _prepare_shapes method given geometric shape type and
        list shapes inputs.
        """
        geometric_shapes = ["circle", "square", "triangle"]
        dataset = ImageDataset(shapes=geometric_shapes, shape_type="geometric")
        prepared_shapes = dataset._prepare_shapes("geometric", dataset.shapes)
        assert set(prepared_shapes) == set(
            geometric_shapes
        ), "The prepared shapes do not match the input list."

    def test_prepare_shapes_dinosaurs_int(self):
        """
        Tests the _prepare_shapes method given dinosaurs shape type and
        integer shapes inputs.
        """
        dataset = ImageDataset(shape_type="dinosaurs", shapes=2)
        assert len(dataset.shapes) == 2, "Should prepare 2 dinosaur shapes."

    def test_prepare_shapes_dinosaurs_list(self):
        """
        Tests the _prepare_shapes method given dinosaurs shape type and
        list shapes inputs.
        """
        dinosaur_shapes = [
            "Diabloceratops_UDL",
            "Sauroposeidon_UDL",
            "Minimocursor_fuzzy",
        ]
        dataset = ImageDataset(shapes=dinosaur_shapes, shape_type="dinosaurs")
        prepared_shapes = dataset._prepare_shapes("dinosaurs", dataset.shapes)
        assert set(prepared_shapes) == set(
            dinosaur_shapes
        ), "The prepared shapes do not match the input list."

    def test_prepare_shapes_error(self):
        """
        Tests the _prepare_shapes method given invalid shape type or
        invalid shapes inputs.
        """
        with pytest.raises(ValueError) as e_type:
            ImageDataset(shapes=10, shape_type="invalid_type")
            assert "shape_type must be either 'geometric' or 'dinosaurs'" in str(
                e_type.value
            )

        # Test with invalid shapes input
        invalid_shapes_input = 1000  # Assuming there aren't 1000 shapes available
        with pytest.raises(ValueError) as e_shapes:
            ImageDataset(shape_type="geometric", shapes=invalid_shapes_input)
            assert "Requested number of shapes" in str(e_shapes.value)

    def test_generate_samples_type(self):
        """
        Tests the samples and labels attribute contain the correct type
        after applying the generate_samples method.
        """
        dataset = ImageDataset()
        assert all(
            isinstance(sample, Image.Image) for sample in dataset.samples
        ), "All samples should be of type PIL.Image.Image."
        assert all(
            isinstance(ground_truth, Image.Image)
            for ground_truth in dataset.ground_truth
        ), "All ground truths should be of type PIL.Image.Image."

    def test_len(self):
        """Tests the correctness of the __len__ method."""
        dataset = ImageDataset()
        expected_len = len(dataset.samples)
        assert (
            len(dataset) == expected_len
        ), "Length should match the number of samples."

    # Error handling
    def test_invalid_shape_type(self):
        """Tests error handling for invalid shape type input."""
        with pytest.raises(ValueError) as excinfo:
            ImageDataset(shape_type="invalid")
        assert (
            "Invalid shape_type provided. Please enter 'geometric' or 'dinosaurs'."
            in str(excinfo.value)
        )

    def test_invalid_background_input(self):
        """Tests error handling for invalid background input types."""
        with pytest.raises(TypeError):
            ImageDataset(backgrounds="five")

    def test_invalid_contour_thickness(self):
        """Tests error handling for invalid contour thickness input."""
        with pytest.raises(AssertionError) as excinfo:
            ImageDataset(contour_thickness=3.5)
        assert "Contour thickness must be an integer value." in str(excinfo.value)

    def test_invalid_shapes_input(self):
        """Tests error handling for invalid shapes input."""
        with pytest.raises(ValueError):
            ImageDataset(shapes="many")
        with pytest.raises(ValueError):
            ImageDataset(shapes=1000)  # Assuming it exceeds the available shapes

    def test_empty_color_list(self):
        """Tests that providing an empty color list correctly triggers the fallback mechanism."""
        dataset = ImageDataset(shape_colors=[])
        assert (
            len(dataset.shape_colors) > 0
        ), "Fallback color should be used when an empty list is provided"

    def test_invalid_color_input(self):
        """Tests error handling for invalid color inputs."""
        with pytest.raises(ValueError):
            ImageDataset(
                shape_colors=["invisible"]
            )  # Assuming 'invisible' is not a valid color
        with pytest.raises(ValueError):
            ImageDataset(shape_colors=9999)  # Assuming it exceeds the available colors


class TestBalancedImageDataset:
    # Initialisation
    def test_initializations(self):
        """
        Tests proper handling of parameters during initialization.
        """
        dataset = BalancedImageDataset(n_variants=1, shuffled=False)
        assert not dataset.shuffled, "Shuffled parameter should be respected"
        assert dataset.n_variants == 1, "n_variants should be correctly set"

    # Shuffle
    def test_shuffle_dataset_same_items(self):
        """
        Tests the samples and labels attribute of the object still contains
        the same items after the shuffle_dataset method.
        """
        dataset = BalancedImageDataset(shuffled=False)
        pre_shuffle = [
            (id(img), label) for img, label in zip(dataset.samples, dataset.labels)
        ]  # Use id of Image to avoid comparison errors
        dataset.shuffle_dataset()
        post_shuffle = [
            (id(img), label) for img, label in zip(dataset.samples, dataset.labels)
        ]
        assert Counter(pre_shuffle) == Counter(
            post_shuffle
        ), "Shuffling should not change dataset contents."

    def test_shuffle_dataset_changes_order(self, ordered_dataset):
        """
        Ensures that the shuffle_dataset method effectively shuffles the order of dataset elements
        without losing or duplicating any items.
        """
        dataset, original_order = ordered_dataset
        dataset.shuffle_dataset()
        shuffled_order = [
            (id(img), id(lbl)) for img, lbl in zip(dataset.samples, dataset.labels)
        ]
        assert Counter(original_order) == Counter(
            shuffled_order
        ), "Shuffling should not change dataset contents"
        # Check if order has potentially changed
        if original_order == shuffled_order:
            pytest.fail(
                "Shuffling did not change the order of elements, which might be okay sometimes but should be investigated."
            )

    # Edge cases for sample generation
    def test_zero_variants(self):
        """
        Tests that initializing the ImageDataset with zero variants raises a ValueError.
        This test checks the constraint that `n_variants` must be a positive integer, as zero variants mean no data
        would be generated, which is impractical for any dataset intended for training or testing purposes.
        """
        with pytest.raises(ValueError) as excinfo:
            BalancedImageDataset(n_variants=0)
        assert "n_variants must be a positive integer greater than 0." in str(
            excinfo.value
        ), "Expected ValueError when n_variants is zero"

    def test_single_variant(self):
        """
        Verifies that the correct number of samples are generated for a single variant per combination.
        """
        dataset = BalancedImageDataset(n_variants=1)
        expected_count = len(dataset.shapes) * len(dataset.backgrounds)
        assert (
            len(dataset.samples) == expected_count
        ), "Should generate one sample per shape-background combination."

    def test_maximum_shapes(self):
        """
        Tests generation with the maximum number of shapes provided by the shape generator.
        """
        max_shapes = len(
            GeometricShapeGenerator().shape_names
        )  # Assuming using GeometricShapeGenerator
        dataset = BalancedImageDataset(shapes=max_shapes)
        assert (
            len(dataset.shapes) == max_shapes
        ), "Should handle maximum number of shapes."

    def test_negative_variants(self):
        """
        Ensures that the system properly handles or rejects negative numbers of variants.
        """
        with pytest.raises(ValueError):
            ImageDataset(n_variants=-1)

    # Sample counts and get item
    def test_sample_count(self):
        shapes = ["circle", "square", "triangle"]
        backgrounds = ["bubbly_0118.jpg", "waffled_0060.jpg", "fibrous_0151.jpg"]
        dataset = BalancedImageDataset(
            shapes=shapes, backgrounds=backgrounds, n_variants=2, shuffled=False
        )
        expected_count = (
            len(dataset.shapes)
            * len(dataset.backgrounds)
            * dataset.n_variants
            * len(dataset.shape_colors)
        )
        assert (
            len(dataset.samples) == expected_count
        ), "Total number of samples does not match expected."

    def test_get_item(self):
        """
        Tests the correctness of the __getitem__ method. It should automatically transform the Image into a torch.Tensor.
        """
        dataset = BalancedImageDataset()
        img, label, extras = dataset[0]
        assert isinstance(img, torch.Tensor), "Item should be a torch.Tensor"
        assert img.dtype == torch.float32, "The tensor should have dtype float32"
        assert isinstance(label, int), "Label should be an integer for the ID."
        assert isinstance(extras, dict), "Extras should be a dictionary."

    def test_balance_of_combinations(self):
        """
        Tests if each combination of background, shape, and color appears exactly the expected number of times.
        """
        dataset = BalancedImageDataset(
            backgrounds=3, shapes=3, n_variants=2, shape_colors=["red", "blue"]
        )
        count_combinations = {}
        for bg_label, fg_shape, fg_color in zip(
            dataset.bg_labels, dataset.fg_shapes, dataset.fg_colors
        ):
            key = (bg_label, fg_shape, fg_color)
            if key in count_combinations:
                count_combinations[key] += 1
            else:
                count_combinations[key] = 1
        expected_count = dataset.n_variants
        for count in count_combinations.values():
            assert (
                count == expected_count
            ), "Each combination should appear exactly n_variants times"

    def test_correct_inheritance_and_overrides(self):
        """
        Ensure that the class correctly inherits from ImageDataset and that the generate_samples method is correctly overridden.
        """
        dataset = BalancedImageDataset(n_variants=1)
        assert isinstance(
            dataset, ImageDataset
        ), "BalancedImageDataset should inherit from ImageDataset"
        assert "generate_samples" in dir(
            dataset
        ), "generate_samples should be overridden in BalancedImageDataset"

    def test_correct_inheritance_and_initialization(self):
        """
        Ensure that the class correctly inherits from ImageDataset and that parameters are properly passed down
        to the base class and used in the initialization process.
        """
        # Initialize with specific values
        backgrounds = 4
        shapes = 5
        n_variants = 2
        shape_colors = ["red", "green"]
        dataset = BalancedImageDataset(
            backgrounds=backgrounds,
            shapes=shapes,
            n_variants=n_variants,
            shape_colors=shape_colors,
            shuffled=True,
            position="center",  # Testing a non-default position
            rotation=True,  # Testing rotation enabled
            contour_thickness=4,
        )

        # Assert inheritance from ImageDataset
        assert isinstance(
            dataset, ImageDataset
        ), "BalancedImageDataset should inherit from ImageDataset"

        # Check if generate_samples and shuffle_dataset are correctly called in __init__
        assert (
            len(dataset.samples)
            == backgrounds * shapes * len(shape_colors) * n_variants
        ), "Sample generation might be incorrect"
        assert (
            dataset.shuffled
        ), "Dataset should be shuffled based on the initialization parameter"

        # Verify that the parameters are correctly set
        assert (
            dataset.image_builder.position == "center"
        ), "The position should be correctly initialized from parameters"
        assert (
            dataset.image_builder.rotation
        ), "The rotation flag should be correctly initialized from parameters"
        assert (
            dataset.contour_thickness == 4
        ), "The contour thickness for the ground truth should be correctly initialized from parameters"
        assert (
            len(dataset.backgrounds) == backgrounds
        ), "Backgrounds should match the provided count"
        assert len(dataset.shapes) == shapes, "Shapes should match the provided count"
        assert len(dataset.shape_colors) == len(
            shape_colors
        ), "Shape colors should match the provided list"

    # Edge cases and errors
    def test_empty_and_edge_cases(self):
        """
        Test the dataset's response to edge cases like zero variants and empty color lists.
        """
        # Testing zero variants
        with pytest.raises(ValueError):
            BalancedImageDataset(n_variants=0)

        # Testing empty color list with fallback mechanism
        dataset = BalancedImageDataset(shape_colors=[])
        assert (
            len(dataset.shape_colors) > 0
        ), "A fallback color should be used when an empty list is provided"

    def test_error_handling(self):
        """
        Tests that the dataset raises errors for incorrect configurations.
        """
        with pytest.raises(ValueError):
            BalancedImageDataset(shapes=1000)

    def test_ground_truth(self):
        """
        Test function to verify properties of generated RGB images for ground truth.
        """
        dataset = BalancedImageDataset(backgrounds=3, shapes=3)

        for ground_truth in dataset.ground_truth:
            assert isinstance(
                ground_truth, Image.Image
            ), "Ground truth is not an instance of PIL.Image.Image"
            pixel_data = ground_truth.load()
            found_255 = False

            for x in range(ground_truth.width):
                for y in range(ground_truth.height):
                    pixel = pixel_data[x, y]
                    assert (
                        isinstance(pixel, tuple) and len(pixel) == 3
                    ), "Pixel format is not RGB"

                    # Check if the pixel has at least one channel with value 255
                    if 255 in pixel:
                        found_255 = True

            assert found_255, "Image does not have any pixel with value 255"


class TestImbalancedDataset:
    def test_initialization(self, imbalanced_image_dataset):
        """
        Tests that the ImbalancedImageDataset initializes correctly with expected properties.
        """
        assert isinstance(imbalanced_image_dataset, ImbalancedImageDataset)
        assert imbalanced_image_dataset.imbalance == 0.5
        assert len(imbalanced_image_dataset.backgrounds) == 5
        assert len(imbalanced_image_dataset.shapes) == 5
        assert imbalanced_image_dataset.shape_colors[0] == (
            255,
            0,
            0,
            255,
        )  # Assuming 'red' translates to this RGBA

    def test_imbalance_validation(self):
        """
        Tests the validation of the imbalance parameter to ensure it accepts correct ranges and defaults.
        """
        with pytest.raises(ValueError):
            ImbalancedImageDataset(imbalance=1.1)
        with pytest.raises(ValueError):
            ImbalancedImageDataset(imbalance=-0.1)
        dataset = ImbalancedImageDataset(imbalance=None)
        assert dataset.imbalance == 0.3  # Default value

    def test_error_handling(self):
        """
        Tests that appropriate errors are thrown for invalid inputs.
        """
        with pytest.raises(ValueError):
            ImbalancedImageDataset(shape_colors="invisible_color")
        with pytest.raises(ValueError):
            ImbalancedImageDataset(shapes="not_a_shape_list")

    def test_sample_count(self):
        """
        Tests the total number of samples generated matches expected counts based on n_variants and the balance setup.
        """
        imb = ImbalancedImageDataset(
            backgrounds=5, shapes=5, n_variants=10, shape_colors="red", imbalance=0.6
        )
        expected_count = len(imb.backgrounds) * imb.n_variants
        actual_count = len(imb.samples)
        assert (
            actual_count == expected_count
        ), f"Expected {expected_count} samples, got {actual_count}"

    def test_color_preparation(self):
        """
        Tests the color preparation handles various scenarios and defaults.
        """
        dataset_with_no_color = ImbalancedImageDataset(shape_colors=None)
        assert (
            len(dataset_with_no_color.shape_colors) == 1
        )  # Default to one random color
        with pytest.raises(ValueError):
            ImbalancedImageDataset(shape_colors=["red", "blue", "green"])

    def test_ground_truth(self):
        """
        Test function to verify properties of generated RGB images for ground truth.
        """
        dataset = ImbalancedImageDataset(backgrounds=3, shapes=3)

        for ground_truth in dataset.ground_truth:
            assert isinstance(
                ground_truth, Image.Image
            ), "Ground truth is not an instance of PIL.Image.Image"
            pixel_data = ground_truth.load()
            found_255 = False

            for x in range(ground_truth.width):
                for y in range(ground_truth.height):
                    pixel = pixel_data[x, y]
                    assert (
                        isinstance(pixel, tuple) and len(pixel) == 3
                    ), "Pixel format is not RGB"

                    # Check if the pixel has at least one channel with value 255
                    if 255 in pixel:
                        found_255 = True

            assert found_255, "Image does not have any pixel with value 255"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))

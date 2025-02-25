import decimal
import random

import cv2
import numpy as np
import torch
from PIL import Image
from typing import List, Tuple, Optional, Union, Dict, Any, Callable

decimal.getcontext().rounding = decimal.ROUND_HALF_UP
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset
from xaiunits.datagenerator.backgrounds import BackgroundGenerator
from xaiunits.datagenerator.foregrounds import (
    DinosaurShapeGenerator,
    GeometricShapeGenerator,
)


class ImageBuilder:
    """
    A class to build images with either geometric shapes or dinosaurs superimposed on various backgrounds,
    including options for color customization, image rotation, and positioning of the overlay image.

    A specific background image can be set as default for all generated samples.

    Attributes:
        shape_type (str): Specifies the type of overlay shape, either 'geometric' or 'dinosaurs'.
        rotation (bool): If True, the overlay image will be randomly rotated.
        color (str | tuple): RGBA range (0-255) for the color of overlay shape. Used as default unless overridden.
        position (str): Specifies the positioning of the overlay image on the background, either 'center' or 'random'.
        overlay_scale (float): The scale of the overlay image relative to the background size.
        background_size (tuple): The size of the background image.
        default_background_imagefile (str): Specific background image filename to use as default for all samples.
        back_gen (BackgroundGenerator): An instance of BackgroundGenerator to fetch background images.
        image_gen (ShapeGenerator | DinoGenerator): The generator for overlay images, either shapes or dinosaurs.
    """

    def __init__(
        self,
        shape_type: str = "geometric",
        rotation: bool = False,
        color: Optional[Union[str, Tuple[int, int, int, int]]] = None,
        position: str = "center",
        overlay_scale: float = 0.3,
        background_size: Tuple[int, int] = (512, 512),
        default_background_imagefile: Optional[str] = None,
        source: str = "local",
    ) -> None:
        """
        Initializes the ImageBuilder object with specified configurations for image generation.

        Args:
            shape_type (str): The type of shapes to overlay ('geometric' or 'dinosaurs'). Defaults to 'geometric'.
            rotation (bool): Whether to apply random rotation to the overlay images. Defaults to False.
            color (str | tuple, optional): The color of the shape, specified by name or RGBA tuple. Defaults to None.
            position (str): The positioning of the overlay image ('center' or 'random'). Defaults to 'center'.
            overlay_scale (float): Scale of the overlay image relative to the background. Defaults to 0.3.
            background_size (tuple): The size (width, height) of the background image. Defaults to (512, 512).
            default_background_imagefile (str, optional): Specific background image filename to use as default for all samples.
                Defaults to None.

        Raises:
            ValueError: If input other than 'geometric' or 'dinosaurs' is passed to shape_type.
        """
        self.shape_type = shape_type
        self.rotation = rotation
        self.color = color
        self.position = position
        self.overlay_scale = overlay_scale
        self.background_size = background_size
        self.back_gen = BackgroundGenerator(background_size)
        self.default_background_imagefile = default_background_imagefile

        if shape_type == "geometric":
            self.shape_gen = GeometricShapeGenerator()
        elif shape_type == "dinosaurs":
            self.shape_gen = DinosaurShapeGenerator(source)
        else:
            raise ValueError(
                "Invalid shape_type provided. Please enter 'geometric' or 'dinosaurs'."
            )

    def resize_overlay_to_background(
        self, background: Image.Image, overlay: Image.Image
    ) -> Image.Image:
        """
        Resizes the overlay image to maintain its aspect ratio and fit within the specified background size,
        calculated based on the overlay_scale.

        This method ensures that the overlay fits aesthetically with the background without distorting its original aspect ratio.

        Args:
            background (PIL.Image.Image): The background image onto which the overlay will be placed.
            overlay (PIL.Image.Image): The overlay image to be resized.

        Returns:
            PIL.Image.Image: The resized overlay image, maintaining its original aspect ratio.
        """
        new_width = int(background.width * self.overlay_scale)

        # Calculate the new height to maintain the overlay's aspect ratio
        aspect_ratio = overlay.height / overlay.width
        new_height = int(new_width * aspect_ratio)

        # Resize the overlay image
        resized_overlay = overlay.resize(
            (new_width, new_height), Image.Resampling.LANCZOS
        )

        return resized_overlay

    def img_rotation(
        self, img: Image.Image, angle: Optional[int] = None
    ) -> Image.Image:
        """
        Rotates the image by a specified angle or a random angle between 0 and 360 degrees if none is provided.

        Args:
            img (PIL.Image.Image): The image to be rotated.
            angle (int, optional): The angle in degrees for rotation. If None, a random angle will be used.
                Defaults to None.

        Returns:
            PIL.Image.Image: The rotated image, resized to accommodate its new dimensions if necessary.

        Raises:
            ValueError: If the specified angle is not within the expected range of 0-360 degrees.
        """
        if angle is not None:
            if not isinstance(angle, int):
                raise TypeError("Angle must be an integer.")
            if angle < 0 or angle > 360:
                raise ValueError("Angle must be between 0 and 360 degrees.")
        else:
            angle = random.randint(0, 360)

        rotated_img = img.rotate(angle, expand=True)
        return rotated_img

    def overlay_pos(
        self, background_image: Image.Image, overlay_image: Image.Image
    ) -> Tuple[int, int]:
        """
        Calculates the position of the overlay image on the background based on the specified position setting.

        Args:
            background_image (PIL.Image.Image): The background image.
            overlay_image (PIL.Image.Image): The overlay image.

        Returns:
            tuple: A tuple (x, y) representing the top-left corner of the overlay image's position on the background.

        Raises:
            ValueError: If the position attribute is not 'center' or 'random'.
        """
        if self.position == "center":
            base_width, base_height = background_image.size
            overlay_width, overlay_height = overlay_image.size
            x = (base_width - overlay_width) // 2
            y = (base_height - overlay_height) // 2
        elif self.position == "random":
            max_x = background_image.width - overlay_image.width
            max_y = background_image.height - overlay_image.height
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
        else:
            raise ValueError("Position must be either 'center' or 'random'")
        return (x, y)

    def generate_sample(
        self,
        name: Optional[str] = None,
        background_imagefile: Optional[str] = None,
        color: Optional[Union[str, Tuple[int, int, int, int]]] = None,
        contour_thickness: int = 3,
    ) -> Tuple[Image.Image, int, str, Image.Image]:
        """
        Generates a single image sample with a specified shape (or randomly selected if none is provided), color, and background.
        This method allows for specific customization of the image sample or can use default settings defined in the class.
        Useful for generating varied datasets for image recognition tasks.

        Args:
            name (str, optional): The name of the shape or dinosaur to use. Randomly selects a shape if None.
                Defaults to None.
            background_imagefile (str, optional): Path to a specific background image file to use. Uses the class-level default if None.
                Defaults to None.
            color (str | tuple, optional): The color to apply to the shape, specified as a name or RGBA tuple. Uses the class-level default if None.
                Defaults to None.
            contour_thickness (int) Thickness of lines the contours are drawn with. If it is negative, the contour interiors are drawn. Defaults to 3.

        Returns:
            tuple: Contains the composite image (PIL.Image.Image), the numeric label representing the shape, a string label describing the shape and the gorund truth.
        """
        effective_color = color if color is not None else self.color
        # Select a default background if no specific file is provided for the sample
        if background_imagefile is not None:
            effective_background_imagefile = background_imagefile
        elif self.default_background_imagefile is not None:
            effective_background_imagefile = self.default_background_imagefile
        else:
            # Randomly choose a background if no default and no specific background is provided
            effective_background_imagefile = random.choice(
                self.back_gen.background_names
            )

        background = self.back_gen.get_background(effective_background_imagefile)

        # Fallback to a random shape if name is None
        if name is None:
            name = random.choice(self.shape_gen.shape_names)

        img, lbl_string = self.shape_gen.get_shape(name, effective_color)
        lbl = self.shape_gen.shape_id_map[name]
        img = self.resize_overlay_to_background(background, img)

        if self.rotation:
            img = self.img_rotation(img)
        pos = self.overlay_pos(background, overlay_image=img)
        combined_img = background.copy()
        combined_img.paste(img, pos, img)
        combined_img.info["background"] = effective_background_imagefile

        # Generate ground truth
        # Get the foreground and black background
        bg_width, bg_height = background.size
        rgba_img = Image.new("RGBA", (bg_width, bg_height), (0, 0, 0, 0))
        image_array = np.array(img)
        alpha_channel = image_array[:, :, 3]
        mask = alpha_channel > 0
        image_array[mask, :3] = 255
        img = Image.fromarray(image_array)
        rgba_img.paste(img, pos, img)
        rgb_img = rgba_img.convert("RGB")

        np_image = np.array(rgb_img)
        bgr_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

        contours, _ = cv2.findContours(
            cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        contour_image = bgr_image.copy()
        cv2.drawContours(
            contour_image, contours, -1, (255, 255, 255), contour_thickness
        )

        # Convert the contour image back to RGB format
        rgb_contour_image = cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)
        ground_truth = Image.fromarray(rgb_contour_image)

        return (combined_img, lbl, lbl_string, ground_truth)


class ImageDataset(Dataset):
    """
    A dataset for images with specified configurations for image generation, supporting both balanced and imbalanced datasets.

    Inherits from:
        torch.utils.data.Dataset: The standard base class for defining a dataset within the PyTorch framework.

    Attributes:
        seed (int): Seed for random number generation to ensure reproducibility.
        backgrounds (list): List of background images to use for dataset generation.
        shapes (list): List of shapes to overlay on background images.
        n_variants (int): Number of variations per shape-background combination, affects dataset size.
        background_size (tuple): Dimensions (width, height) of background images.
        shape_type (str): Type of shapes: 'geometric' for geometric shapes, 'dinosaurs' for dinosaur shapes.
        position (str): Overlay position on the background ('center' or 'random').
        overlay_scale (float): Scale factor for overlay relative to the background size.
        rotation (bool): If True, applies random rotation to overlays.
        shape_colors (list): List of default color(s) for shapes, accepts color names or RGBA tuples.
        shuffled (bool): If True, shuffles the dataset after generation.
        transform (callable): Transformation function to apply to each image, typically converting to tensor.
        contour_thickness (int): Thickness of lines the contours are drawn with.
            If it is negative, the contour interiors are drawn.
        image_builder (ImageBuilder): Instance of ImageBuilder for generating images.
        samples (list): List to store the generated samples.
        labels (list): List to store the labels.
        fg_shapes (list): List to store the foreground shapes.
        bg_labels (list): List to store the background labels.
        fg_colors (list): List to store the foreground colors.
        ground_truth (list): List to store the ground truths.
    """

    def __init__(
        self,
        seed: int = 0,
        backgrounds: Union[int, List[str]] = 5,
        shapes: Union[int, List[str]] = 10,
        n_variants: int = 4,
        background_size: Tuple[int, int] = (512, 512),
        shape_type: str = "geometric",
        position: str = "random",
        overlay_scale: float = 0.3,
        rotation: bool = False,
        shape_colors: Optional[
            Union[
                str,
                Tuple[int, int, int, int],
                List[Union[str, Tuple[int, int, int, int]]],
            ]
        ] = None,
        shuffled: bool = True,
        transform: Optional[Callable] = None,
        contour_thickness: int = 3,
        source: str = "local",
    ) -> None:
        """
        Initializes an ImageDataset object.

        Args:
            seed (int): Seed for random number generation to ensure reproducibility. Defaults to 0.
            backgrounds (int | list): Number or list of specific backgrounds to use. Defaults to 5.
            shapes (int | list): Number or list of specific shapes. Defaults to 10.
            n_variants (int): Number of variations per shape-background combination, affects dataset size.
                Defaults to 4.
            background_size (tuple): Dimensions (width, height) of background images. Defaults to (512, 512).
            shape_type (str): 'geometric' for geometric shapes, 'dinosaurs' for dinosaur shapes.
                Defaults to 'geometric'.
            position (str): Overlay position on the background ('center' or 'random'). Defaults to 'random'.
            overlay_scale (float): Scale factor for overlay relative to the background size. Defaults to 0.3.
            rotation (bool): If True, applies random rotation to overlays. Defaults to False.
            shape_colors (str | tuple, optional): Default color(s) for shapes, accepts color names or RGBA tuples.
                Defaults to None.
            shuffled (bool): If True, shuffles the dataset after generation. Defaults to True.
            transform (callable, optional): Transformation function to apply to each image, typically converting to tensor.
                Defaults to None.
            contour_thickness (int) Thickness of lines the contours are drawn with. If it is negative, the contour interiors are drawn.
                Defaults to 3.
        """
        self.seed = seed
        random.seed(self.seed)
        assert isinstance(
            contour_thickness, int
        ), "Contour thickness must be an integer value."
        self.n_variants = self._validate_n_variants(n_variants)
        self.image_builder = ImageBuilder(
            shape_type=shape_type,
            rotation=rotation,
            position=position,
            overlay_scale=overlay_scale,
            background_size=background_size,
            source=source,
        )
        self.backgrounds = self._prepare_backgrounds(backgrounds)
        self.shapes = self._prepare_shapes(shape_type, shapes, source)
        self.shape_colors = self._prepare_shape_color(shape_colors)
        self.transform = transform or transforms.Compose(
            [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
        )
        self.samples = []
        self.labels = []
        self.fg_shapes = []
        self.bg_labels = []
        self.fg_colors = []
        self.ground_truth = []
        self.shuffled = shuffled
        self.contour_thickness = contour_thickness

    def _validate_n_variants(self, n_variants: int) -> int:
        """
        Validates that the number of variants per shape-background combination is a positive integer.

        The `n_variants` parameter controls how many different versions of each shape-background combination are generated,
        varying elements such as position and possibly color if specified. This allows for diverse training data in image
        recognition tasks, improving the model's ability to generalize from different perspectives and conditions.

        Args:
            n_variants (int): The number of variations per shape-background combination to generate.

        Returns:
            int: The validated number of variants.

        Raises:
            ValueError: If `n_variants` is not an integer or is less than or equal to zero.
        """
        if not isinstance(n_variants, int) or n_variants <= 0:
            raise ValueError("n_variants must be a positive integer greater than 0.")
        return n_variants

    def _prepare_shapes(
        self, shape_type: str, shapes: Union[int, List[str]], source: str
    ) -> List[str]:
        """
        Prepares a list of shapes or dinosaurs based on the input and the specified shape type.

        This method processes the input to generate a list of specific shapes or dinosaur names.
        If a numerical input is provided, it selects that many random shapes/dinosaurs from the
        available names. If a list is provided, it directly uses those specific names.

        Args:
            shape_type (str): Specifies the type of overlay image, either 'geometric' or 'dinosaurs'.
            shapes (int | list): Number or list of specific shape names. If an integer is provided,
                it indicates how many random shapes or dinosaurs to select.

        Returns:
            list: A list of shape names or dinosaur names to be used as overlays.

        Raises:
            ValueError: If the shapes input is neither an integer nor a list, or if the shape_type is
                not recognized as 'geometric' or 'dinosaurs'.
        """
        if shape_type == "geometric":
            shape_generator = GeometricShapeGenerator()
        elif shape_type == "dinosaurs":
            shape_generator = DinosaurShapeGenerator(source)
        else:
            raise ValueError("shape_type must be either 'geometric' or 'dinosaurs'")

        all_shapes = shape_generator.shape_names

        if isinstance(shapes, int):
            if shapes > len(all_shapes):
                raise ValueError(
                    f"Requested number of shapes ({shapes}) exceeds the available shapes ({len(all_shapes)})."
                )
            return random.sample(all_shapes, shapes)
        elif isinstance(shapes, list):
            if not all(name in all_shapes for name in shapes):
                raise ValueError("One or more specified shapes are not valid.")
            return shapes
        else:
            raise ValueError("Shapes must be an integer or a list of shape names.")

    def _prepare_backgrounds(self, backgrounds: Union[int, List[str]]) -> List[str]:
        """
        Prepares background images based on the input.

        This method helps to either randomly select a set number of background images from the available
        pool or validate and use a provided list of specific background filenames.

        If a numerical value is provided, selects that many random backgrounds. If a list is provided,
        validates and uses those specific backgrounds.

        Args:
            backgrounds (int | list): Number of random backgrounds to select or a list of specific background filenames.

        Returns:
            list: A list of background filenames to be used in the dataset.

        Raises:
            ValueError: If the input is neither an integer nor a list, or if any specified background filename
                is not found in the available backgrounds.
        """
        available_backgrounds = BackgroundGenerator().background_names
        if isinstance(backgrounds, int):
            if backgrounds > len(available_backgrounds):
                raise ValueError(
                    f"Requested {backgrounds} backgrounds, but only {len(available_backgrounds)} are available."
                )
            return random.sample(available_backgrounds, backgrounds)
        elif isinstance(backgrounds, list):
            if not all(bg in available_backgrounds for bg in backgrounds):
                missing = [bg for bg in backgrounds if bg not in available_backgrounds]
                raise ValueError(f"Some specified backgrounds not found: {missing}")
            return backgrounds
        else:
            raise TypeError(
                "Backgrounds should be either an int or a list of background filenames."
            )

    def _prepare_shape_color(
        self,
        shape_colors: Optional[
            Union[
                int,
                str,
                Tuple[int, int, int, int],
                List[Union[str, Tuple[int, int, int, int]]],
            ]
        ],
    ) -> List[Tuple[int, int, int, int]]:
        """
        Prepares shape colors by validating input against available colors.

        If no valid colors are provided, a default color is selected. Accepts single or multiple colors.

        Args:
            shape_colors (int | str | tuple | list): Specifies how many random colors to select or provides specific color(s).
                Can be a single color name, RGBA tuple, or list of names/tuples.

        Returns:
            list: A list of validated RGBA tuples representing the colors.

        Raises:
            ValueError: If input is invalid or colors are not found in the available color dictionary.
                Details about the invalid input are provided in the error message.
        """
        available_colors = list(self.image_builder.shape_gen._colors_rgba.values())
        if not shape_colors:
            return [random.choice(available_colors)]

        if isinstance(shape_colors, int):
            if shape_colors > len(available_colors):
                raise ValueError(
                    "Requested number of colors exceeds the available colors."
                )
            return random.sample(available_colors, shape_colors)
        elif isinstance(shape_colors, list):
            validated_colors = []
            for color in shape_colors:
                if isinstance(color, str):
                    try:
                        validated_color = self.image_builder.shape_gen.validate_color(
                            color
                        )
                        validated_colors.append(validated_color)
                    except ValueError as e:
                        raise ValueError(
                            f"Invalid color name: {color}. Error: {str(e)}"
                        )
                elif (
                    isinstance(color, tuple)
                    and len(color) == 4
                    and all(isinstance(c, int) and 0 <= c <= 255 for c in color)
                ):
                    validated_colors.append(color)
                else:
                    raise ValueError(
                        "Each color must be either a string name or a 4-element tuple of integers (RGBA)."
                    )
            if not validated_colors:
                return [self.image_builder.shape_gen.validate_color("blue")]
            return validated_colors
        else:
            raise ValueError(
                "shape_colors must be either an integer or a list of color names or RGBA tuples."
            )

    def generate_samples(self) -> None:
        """
        Placeholder method for generating the samples either for balanced or imbalanced datasets.
        """
        pass

    def shuffle_dataset(self) -> None:
        """
        Randomly shuffles the dataset samples and corresponding labels to ensure variety
        in training and evaluation phases.

        Raises:
            ValueError: If the dataset is empty and shuffling is not possible.
        """
        if self.samples:  # Only shuffle if there are samples
            combined = list(
                zip(
                    self.samples,
                    self.labels,
                    self.fg_shapes,
                    self.bg_labels,
                    self.fg_colors,
                    self.ground_truth,
                )
            )
            random.shuffle(combined)
            (
                self.samples,
                self.labels,
                self.fg_shapes,
                self.bg_labels,
                self.fg_colors,
                self.ground_truth,
            ) = zip(*combined)
        else:
            raise ValueError("Cannot shuffle an empty dataset.")

    def __len__(self) -> int:
        """
        Returns thet number of samples in the dataset.

        Returns:
            int: number of samples contained by the dataset.
        """
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, int, Dict[str, Union[str, torch.Tensor, Image.Image]]]:
        """
        Retrieves an image and its label by index.

        The image is transformed into a tensor if a transform is applied.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the transformed image tensor, label, a dict of other attributes.
        """
        img, label, fg_shape, bg_label, fg_color, ground_truth = (
            self.samples[idx],
            self.labels[idx],
            self.fg_shapes[idx],
            self.bg_labels[idx],
            self.fg_colors[idx],
            self.ground_truth[idx],
        )
        if self.transform:
            img = self.transform(img)
            ground_truth = self.transform(ground_truth)
        return (
            img,
            label,
            {
                "fg_shape": fg_shape,
                "bg_label": bg_label,
                "fg_color": torch.tensor(fg_color),
                "ground_truth_attribute": ground_truth,
            },
        )

    def _re_label(self) -> None:
        """Re-labels the dataset labels with integer indices."""
        label_index = {label: idx for idx, label in enumerate(set(self.labels))}
        self.labels = [label_index[y] for y in self.labels]

    @staticmethod
    def show_image(img_tensor: torch.Tensor) -> None:
        """
        Displays an image given its tensor representation.

        Args:
            img_tensor (torch.Tensor): The image tensor to display.
        """
        img = transforms.ToPILImage()(img_tensor).convert("RGB")
        img.show()

    @property
    def default_metric(self) -> Callable:
        """
        The default metric for evaluating the performance of explanation methods applied
        to this dataset.

        For this dataset, the default metric is the mask ratio metric that is constructed
        based on the ground truth and context. Mask ratio is defined as the ratio of absolute
        attribution score that lies within the foreground and the image.

        Returns:
            type: A class that wraps around the default metric to be instantiated
                within the pipeline.
        """
        from xaiunits.metrics import wrap_metric

        def metric_ratio_mapping(
            metric,
            feature_input,
            y_labels,
            target,
            context,
            attribute,
            method_instance,
            model,
            **other,
        ):
            metric_inputs = {
                "inputs": attribute,
                "masks": context["ground_truth_attribute"],
            }
            return metric_inputs

        def mask_ratio(inputs, masks):
            inputs = torch.abs(inputs)
            in_regions = (inputs * masks).flatten(1).sum(dim=1)
            all = (inputs).flatten(1).sum(dim=1)
            ratios = in_regions / all
            return ratios

        return wrap_metric(
            mask_ratio,
            input_generator_fns=metric_ratio_mapping,
            out_processing=lambda x: x,
        )


class BalancedImageDataset(ImageDataset):
    """
    A dataset for images where each each image consists of a background and a foreground overlay.

    This 'balanced' dataset ensures that each combination of background (bg), foreground (fg), and foreground color (fg_color)
    appears the same number of times across the dataset, making it ideal for machine learning models that benefit from
    uniform exposure to all feature combinations.

    Inherits all parameters from ImageDataset, and introduces no additional parameters, but it overrides the behavior
    to ensure balance in the dataset composition.

    Inherits from:
        ImageDataset: Standard dataset that contains images with backgorunds and foregrounds.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initializes a BalancedImageDataset with the same parameters as ImageDataset, ensuring each combination
        of background, foreground, and color appears uniformly across the dataset.

        After initialization, it automatically generates the samples and shuffles them if the 'shuffled' attribute is True.

        Args:
            *args: Additional arguments passed to the superclass initializer.
            **kwargs: Additional keyword arguments passed to the superclass initializer.
        """
        super().__init__(*args, **kwargs)
        self.generate_samples()
        if self.shuffled:
            self.shuffle_dataset()
        self._re_label()

    def generate_samples(self) -> None:
        """
        Generates a balanced set of image samples by uniformly distributing each combination of background,
        foreground shape, and color.

        Iterates over each background, each shape, and each color to create the specified number of variants per combination.
        Each generated image is stored in the 'samples' list, with corresponding labels in 'labels', and other metadata
        like foreground shapes, background labels, and foreground colors stored in their respective lists.

        Raises:
            ValueError: If there is an issue with image generation parameters or overlay combinations.
        """
        for bg in self.backgrounds:
            for shape in self.shapes:
                for color in self.shape_colors:
                    for _ in range(self.n_variants):
                        img, lbl, fg_shape, gt = self.image_builder.generate_sample(
                            name=shape,
                            background_imagefile=bg,
                            color=color,
                            contour_thickness=self.contour_thickness,
                        )
                        self.samples.append(img)
                        self.labels.append(lbl)
                        self.fg_shapes.append(fg_shape)
                        self.ground_truth.append(gt)
                        self.bg_labels.append(bg)
                        self.fg_colors.append(color)


class ImbalancedImageDataset(ImageDataset):
    """
    Creates Image Dataset where each image comprises of a background image an a foreground image.

    Background images, type of foreground, color of foreground as well as other parameters can be specified.

    Imbalance refers to the fact users can specify the percentage of dominant (bg, fg) pair vs other pair.

    Inherits from:
        ImageDataset: Standard dataset that contains images with backgorunds and foregrounds.

    Attributes:
        imbalance (float): The proportion of samples that should favor a particular background per shape.
            Should be within the range (0.0 to 1.0) inclusive.
    """

    def __init__(
        self,
        backgrounds: Union[int, List[str]] = 5,
        shapes: Union[int, List[str]] = 3,
        n_variants: int = 100,
        shape_colors: Union[str, Tuple[int, int, int, int]] = "red",
        imbalance: float = 0.8,
        **kwargs: Any,
    ) -> None:
        """
        Initializes an ImbalancedImageDataset object with specified parameters, focusing on creating dataset variations
        based on an imbalance parameter that dictates the dominance of certain shape-background pairs.

        Args:
            backgrounds (int | list): The number or list of specific background filenames. Defaults to 5.
            shapes (int | list): The number or list of specific shapes. Defaults to 3.
            n_variants (int): Number of variations per shape-background combination, affects dataset size.
                Defaults to 100.
            shape_colors (str | tuple): The default color for all shapes in the dataset. Defaults to 'red'.
            imbalance (float): The proportion (0.0 to 1.0) of samples that should favor a particular background per shape.
                Defaults to 0.8.
            **kwargs: Additional keyword arguments passed to the superclass initializer.
        """
        kwargs["shape_colors"] = shape_colors
        super().__init__(
            backgrounds=backgrounds, shapes=shapes, n_variants=n_variants, **kwargs
        )
        assert len(self.shapes) >= 2
        assert len(self.backgrounds) >= len(self.shapes)

        self.imbalance = self._validate_imbalance(imbalance)
        self.generate_samples()
        if self.shuffled:
            self.shuffle_dataset()
        self._re_label()

    def _prepare_shape_color(
        self, shape_colors: Optional[Union[str, Tuple[int, int, int, int]]]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Prepares a single shape color based on the input.

        Selects a random color if None is provided, validates a provided color string or RGBA tuple.

        Args:
            shape_colors (str | tuple | NoneType): A specific color name, RGBA tuple, or None to select a random color.

        Returns:
            list: A list containing a single validated RGBA tuple representing the color.

        Raises:
            ValueError: If the input is invalid or if the color name is not found in the predefined color dictionary.
        """
        if isinstance(shape_colors, str):
            try:
                # Assuming validate_color method returns a tuple if the color name is valid
                shape_colors = [
                    self.image_builder.shape_gen.validate_color(shape_colors)
                ]
            except KeyError:
                raise ValueError(
                    f"Invalid color name: {shape_colors}. Color name must be a valid key in the color dictionary."
                )

        elif (
            isinstance(shape_colors, tuple)
            and len(shape_colors) == 4
            and all(isinstance(c, int) and 0 <= c <= 255 for c in shape_colors)
        ):
            shape_colors = [shape_colors]
        elif shape_colors is None:
            available_colors = list(self.image_builder.shape_gen._colors_rgba.values())
            shape_colors = [random.choice(available_colors)]
        else:
            raise ValueError(
                "shape_color must be either a color name string or a 4-element tuple of integers (RGBA)."
            )

        return super()._prepare_shape_color(shape_colors)

    def _validate_imbalance(self, imbalance: float) -> float:
        """
        Validates that the imbalance parameter is a float between 0.0 and 1.0 inclusive, or None.

        Ensures that the dataset can properly reflect the desired level of imbalance, adjusting for the
        number of variants and available backgrounds.

        Args:
            imbalance (float | NoneType): The imbalance value to validate. If None is given as input, then
                the argument will be treated as 0.3.

        Returns:
            float: The validated imbalance value.

        Raises:
            ValueError: If the imbalance is not within the inclusive range [0.0, 1.0] or if the imbalance
                settings are not feasible with the current settings of n_variants and backgrounds.
        """
        if imbalance is None:
            imbalance = (
                0.3  # if set to 0.0 then if main_bg_samples < 1 always causes error
            )

        if not isinstance(imbalance, (float, int)):
            raise ValueError("Imbalance must be a float or an integer.")

        if not (0.0 < float(imbalance) <= 1.0):
            raise ValueError(
                "Imbalance must be between (0.0,  1.0], 1.0 being inclusive."
            )

        main_bg_samples = int(self.n_variants * imbalance)
        other_bg_samples = (
            (self.n_variants - main_bg_samples) if len(self.backgrounds) > 1 else 0
        )

        if main_bg_samples < 1:
            raise ValueError(
                "Imbalance Value too small given size of n_variants parameters."
            )
        if len(self.backgrounds) > 1 and other_bg_samples < 1:
            raise ValueError(
                "Imbalance value too high, leaving no variants for other backgrounds."
            )

        return float(imbalance)

    def generate_samples(self) -> None:
        """
        Generates a set of image samples with overlay shapes or dinosaurs on backgrounds, considering imbalance.

        Depending on the 'imbalance' parameter, this method either:
            - Allocates a specific fraction (defined by 'imbalance') of the samples for each shape to a particular background,
              with the remainder distributed among the other backgrounds.
            - Assigns all samples for a shape to a single background (imbalance = 1.0).
        """
        # Generate samples for each shape
        color = self.shape_colors[0]
        for i, shape in enumerate(self.shapes):
            shape_backgrounds = list(self.backgrounds)  # Make a copy to manipulate

            # Select a main background for the imbalance
            main_background = shape_backgrounds[i % len(shape_backgrounds)]
            shape_backgrounds.remove(main_background)

            # Calculate the number of samples for the main background
            main_bg_samples = int(self.n_variants * self.imbalance)
            # Calculate the samples for other backgrounds
            total_other_samples = self.n_variants - main_bg_samples
            other_bg_samples = (
                total_other_samples // len(shape_backgrounds)
                if len(shape_backgrounds) > 0
                else 0
            )

            # Generate samples for the main background
            for _ in range(main_bg_samples):
                img, lbl, fg_shape, gt = self.image_builder.generate_sample(
                    name=shape,
                    background_imagefile=main_background,
                    color=color,
                    contour_thickness=self.contour_thickness,
                )
                self.samples.append(img)
                self.labels.append(lbl)
                self.fg_shapes.append(fg_shape)
                self.ground_truth.append(gt)
                self.bg_labels.append(main_background)
                self.fg_colors.append(color)

            # Generate samples for the other backgrounds
            for bg in shape_backgrounds:
                for _ in range(other_bg_samples):
                    img, lbl, fg_shape, gt = self.image_builder.generate_sample(
                        name=shape,
                        background_imagefile=bg,
                        color=color,
                        contour_thickness=self.contour_thickness,
                    )
                    self.samples.append(img)
                    self.labels.append(lbl)
                    self.fg_shapes.append(fg_shape)
                    self.ground_truth.append(gt)
                    self.bg_labels.append(bg)
                    self.fg_colors.append(color)


if __name__ == "__main__":
    import pandas as pd
    import seaborn as sns

    balanced_dataset = BalancedImageDataset(
        backgrounds=4,
        shapes=3,
        n_variants=10,
        shape_colors=["green", "blue", "red"],
        shuffled=False,
        contour_thickness=3,
    )
    not_imbalanced_but_not_balanced = ImbalancedImageDataset(
        imbalance=0.97, backgrounds=4, shapes=3, n_variants=100, shuffled=False
    )
    imbalanced_dataset = ImbalancedImageDataset(
        imbalance=0.5, backgrounds=4, shapes=3, n_variants=100
    )
    # totally_imbalanced_dataset = ImbalancedImageDataset(
    #     imbalance=0.8, backgrounds=4, shapes=3, n_variants=100
    # )
    datasets = [
        balanced_dataset,
        not_imbalanced_but_not_balanced,
        imbalanced_dataset,
        # totally_imbalanced_dataset,
    ]
    print(balanced_dataset[0])
    data = {
        "fg_label": [],
        "fg_shape": [],
        "bg_label": [],
        "fg_color": [],
        "imbalance": [],
    }

    for dataset in datasets:

        for _, label, context in dataset:
            data["fg_label"].append(label)
            data["fg_shape"].append(context["fg_shape"])
            data["bg_label"].append(context["bg_label"])
            data["fg_color"].append(context["fg_color"])
            data["imbalance"].append(getattr(dataset, "imbalance", 0.0))

    df = pd.DataFrame(data)

    # Plotting
    g = sns.catplot(
        x="fg_shape",
        hue="bg_label",
        col="imbalance",
        data=df,
        kind="count",
        palette="viridis",
        aspect=0.7,
        sharey=False,
    )

    for t in g._legend.texts:
        original_text = t.get_text()
        # stripping .jpg and last 5 characters which are like "_0045"
        cleaned_text = original_text.replace(".jpg", "")[:-5]
        t.set_text(cleaned_text)
    g._legend.set_title("Background")
    g.set_axis_labels("Foreground Shape", "Count")
    g.fig.subplots_adjust(wspace=0.3)
    g.set(ylim=(0, 100))

    # Save the plot as an svg file
    g.savefig("correlated_slide.svg")

    plt.show()

import numbers
import random
import warnings
from collections.abc import Sequence
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import functional as F
from torchvision.transforms.functional import (InterpolationMode,
                                               _interpolation_modes_from_int)


def _check_sequence_input(x, name, req_sizes):
    msg = req_sizes[0] if len(req_sizes) < 2 else " or ".join(
        [str(s) for s in req_sizes])
    if not isinstance(x, Sequence):
        raise TypeError("{} should be a sequence of length {}.".format(
            name, msg))
    if len(x) not in req_sizes:
        raise ValueError("{} should be sequence of length {}.".format(
            name, msg))


def _setup_angle(x, name, req_sizes=(2, )):
    if isinstance(x, numbers.Number):
        if x < 0:
            raise ValueError(
                "If {} is a single number, it must be positive.".format(name))
        x = [-x, x]
    else:
        _check_sequence_input(x, name, req_sizes)

    return [float(d) for d in x]


class Compose(object):
    """
    Composes a sequence of transforms.
    Arguments:
        transforms: A list of transforms.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, groundtruth):
        for t in self.transforms:
            image, groundtruth = t(image, groundtruth)
        return image, groundtruth

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ToTensor(object):
    """
    Converts image to torch Tensor.
    """
    def __call__(self, image, groundtruth):
        return F.to_tensor(image), F.to_tensor(groundtruth)


class Normalize(object):
    """
    Normalizes image by mean and std.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, groundtruth):
        image = F.normalize(image, mean=self.mean, std=self.std)
        groundtruth = F.normalize(groundtruth, mean=self.mean, std=self.std)
        return image, groundtruth


class RandomScale(object):
    """
    Applies random scale augmentation.
    Arguments:
        min_scale: Minimum scale value.
        max_scale: Maximum scale value.
        scale_step_size: The step size from minimum to maximum value.
    """
    def __init__(self, min_scale, max_scale, scale_step_size):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_step_size = scale_step_size

    @staticmethod
    def get_random_scale(min_scale_factor, max_scale_factor, step_size):
        """Gets a random scale value.
        Args:
            min_scale_factor: Minimum scale value.
            max_scale_factor: Maximum scale value.
            step_size: The step size from minimum to maximum value.
        Returns:
            A random scale value selected between minimum and maximum value.
        Raises:
            ValueError: min_scale_factor has unexpected value.
        """
        if min_scale_factor < 0 or min_scale_factor > max_scale_factor:
            raise ValueError('Unexpected value of min_scale_factor.')

        if min_scale_factor == max_scale_factor:
            return min_scale_factor

        # When step_size = 0, we sample the value uniformly from [min, max).
        if step_size == 0:
            return random.uniform(min_scale_factor, max_scale_factor)

        # When step_size != 0, we randomly select one discrete value from [min, max].
        num_steps = int((max_scale_factor - min_scale_factor) / step_size + 1)
        scale_factors = np.linspace(min_scale_factor, max_scale_factor,
                                    num_steps)
        np.random.shuffle(scale_factors)
        return scale_factors[0]

    def __call__(self, image, groundtruth):
        f_scale = self.get_random_scale(self.min_scale, self.max_scale,
                                        self.scale_step_size)
        # TODO: cv2 uses align_corner=False
        # TODO: use fvcore (https://github.com/facebookresearch/fvcore/blob/master/fvcore/transforms/transform.py#L377)
        image_dtype = image.dtype
        groundtruth_dtype = groundtruth.dtype
        image = cv2.resize(image.astype(np.float),
                           None,
                           fx=f_scale,
                           fy=f_scale,
                           interpolation=cv2.INTER_LINEAR)
        groundtruth = cv2.resize(groundtruth.astype(np.float),
                                 None,
                                 fx=f_scale,
                                 fy=f_scale,
                                 interpolation=cv2.INTER_NEAREST)
        return image.astype(image_dtype), groundtruth.astype(groundtruth_dtype)


class RandomCrop(object):
    """
    Applies random crop augmentation.
    Arguments:
        crop_h: Integer, crop height size.
        crop_w: Integer, crop width size.
        pad_value: Tuple, pad value for image, length 3.
        ignore_groundtruth: Tuple, pad value for groundtruth, length could be 1 (semantic) or 3 (panoptic).
        random_pad: Bool, when crop size larger than image size, whether to randomly pad four boundaries,
            or put image to top-left and only pad bottom and right boundaries.
    """
    def __init__(self, crop_h, crop_w, pad_value, ignore_groundtruth,
                 random_pad):
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.pad_value = pad_value
        self.ignore_groundtruth = ignore_groundtruth
        self.random_pad = random_pad

    def __call__(self, image, groundtruth):
        img_h, img_w = image.shape[0], image.shape[1]
        # save dtype
        image_dtype = image.dtype
        groundtruth_dtype = groundtruth.dtype
        # padding
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            if self.random_pad:
                pad_top = random.randint(0, pad_h)
                pad_bottom = pad_h - pad_top
                pad_left = random.randint(0, pad_w)
                pad_right = pad_w - pad_left
            else:
                pad_top, pad_bottom, pad_left, pad_right = 0, pad_h, 0, pad_w
            img_pad = cv2.copyMakeBorder(image,
                                         pad_top,
                                         pad_bottom,
                                         pad_left,
                                         pad_right,
                                         cv2.BORDER_CONSTANT,
                                         value=self.pad_value)
            groundtruth_pad = cv2.copyMakeBorder(groundtruth,
                                                 pad_top,
                                                 pad_bottom,
                                                 pad_left,
                                                 pad_right,
                                                 cv2.BORDER_CONSTANT,
                                                 value=self.ignore_groundtruth)
        else:
            img_pad, groundtruth_pad = image, groundtruth
        img_h, img_w = img_pad.shape[0], img_pad.shape[1]
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(
            img_pad[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w],
            np.float32)
        groundtruth = np.asarray(
            groundtruth_pad[h_off:h_off + self.crop_h,
                            w_off:w_off + self.crop_w], np.float32)
        return image.astype(image_dtype), groundtruth.astype(groundtruth_dtype)


class RandomHorizontalFlip(object):
    """
    Applies random flip augmentation.
    Arguments:
        prob: Probability of flip.
    """
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, groundtruth):
        if random.random() < self.prob:
            # https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663
            image = image[:, ::-1].copy()
            groundtruth = groundtruth[:, ::-1].copy()
        return image, groundtruth


class RandomRotation(object):
    """Rotate the image by angle.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        degrees (sequence or number): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (sequence, optional): Optional center of rotation, (x, y). Origin is the upper left corner.
            Default is the center of the image.
        fill (sequence or number): Pixel fill value for the area outside the rotated
            image. Default is ``0``. If given a number, the value is used for all bands respectively.
        resample (int, optional): deprecated argument and will be removed since v0.10.0.
            Please use the ``interpolation`` parameter instead.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """
    def __init__(self,
                 degrees,
                 interpolation=InterpolationMode.NEAREST,
                 expand=False,
                 center=None,
                 fill=0,
                 resample=None):
        super().__init__()
        if resample is not None:
            warnings.warn(
                "Argument resample is deprecated and will be removed since v0.10.0. Please, use interpolation instead"
            )
            interpolation = _interpolation_modes_from_int(resample)

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum.")
            interpolation = _interpolation_modes_from_int(interpolation)

        self.degrees = _setup_angle(degrees, name="degrees", req_sizes=(2, ))

        if center is not None:
            _check_sequence_input(center, "center", req_sizes=(2, ))

        self.center = center

        self.resample = self.interpolation = interpolation
        self.expand = expand

        if fill is None:
            fill = 0
        elif not isinstance(fill, (Sequence, numbers.Number)):
            raise TypeError("Fill should be either a sequence or a number.")

        self.fill = fill

    @staticmethod
    def get_params(degrees: List[float]) -> float:
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            float: angle parameter to be passed to ``rotate`` for random rotation.
        """
        angle = float(
            torch.empty(1).uniform_(float(degrees[0]),
                                    float(degrees[1])).item())
        return angle

    def __call__(self, img, gt):
        """
        Args:
            img (PIL Image or Tensor): Image to be rotated.

        Returns:
            PIL Image or Tensor: Rotated image.
        """
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F._get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]
        angle = self.get_params(self.degrees)

        return F.rotate(Image.fromarray(img), angle, self.resample,
                        self.expand, self.center,
                        fill), F.rotate(Image.fromarray(gt), angle,
                                        self.resample, self.expand,
                                        self.center, fill)

    def __repr__(self):
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + '(degrees={0}'.format(
            self.degrees)
        format_string += ', interpolation={0}'.format(interpolate_str)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        if self.fill is not None:
            format_string += ', fill={0}'.format(self.fill)
        format_string += ')'
        return format_string

import re
import base64
import warnings
import numpy as np
from tensorflow.keras import backend
from PIL import Image, ImageMode
from io import BytesIO
from enum import IntEnum
import cv2

def getPoint(dict):
    """
    The function takes in a dictionary and returns a list of the minimum and maximum x and y coordinates
    
    :param dict: the dictionary that contains the coordinates of the bounding boxes
    :return: the xmin, xmax, ymin, ymax values for each of the coordinates in the dictionary.
    """
    listxmin, listxmax, listymin, listymax = [], [], [], []
    for i in range(len(dict['CoordinateXY'])):
        xmin = dict['CoordinateXY'][i][0]
        xmax = dict['CoordinateXY'][i][2]
        ymin = dict['CoordinateXY'][i][1]
        ymax = dict['CoordinateXY'][i][3]
        listxmin.append(xmin)
        listxmax.append(xmax)
        listymin.append(ymin)
        listxmax.append(ymax)
    return listxmin, listxmax, listymin, listymax
    
# "The Palette class is an enumeration of two values, WEB and ADAPTIVE."
# 
# The first line of the class definition is the class header. It consists of the keyword class,
# followed by the name of the class, followed by a colon. The class header is followed by the class
# body, which is indented
class Palette(IntEnum):
    WEB = 0
    ADAPTIVE = 1

# `Dither` is an enumeration of the different dithering algorithms that can be used to convert a color
# image to a black and white image
class Dither(IntEnum):
    NONE = 0
    ORDERED = 1  # Not yet implemented
    RASTERIZE = 2  # Not yet implemented
    FLOYDSTEINBERG = 3  # default

def estimate_blur(image: np.array, threshold: int = 100):
    """
    It calculates the variance of the Laplacian of the image
    
    :param image: The image that you want to check for blur
    :type image: np.array
    :param threshold: The minimum variance of the Laplacian to qualify a region as blurry, defaults to
    100
    :type threshold: int (optional)
    :return: The variance of the Laplacian of the image.
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_map = cv2.Laplacian(image, cv2.CV_64F)
    score = np.var(blur_map)
    return score, bool(score < threshold)


def getmodebase(mode):
    """
    Gets the "base" mode for given mode.  This function returns "L" for
    images that contain grayscale data, and "RGB" for images that
    contain color data.

    :param mode: Input mode.
    :returns: "L" or "RGB".
    :exception KeyError: If the input mode was not a standard mode.
    """
    return ImageMode.getmode(mode).basemode

def base64_to_pil(img_base64, type, ext = False):
    """
    Convert base64 image data to PIL image
    """
    image_data = re.sub('^data:image/.+;base64,', '', img_base64)
    if type == 1:
        decoded_data = base64.b64decode(image_data)
        np_data = np.fromstring(decoded_data,np.uint8)
        cv_img = cv2.imdecode(np_data,1)
        if ext:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            return cv_img
        else:
            return cv_img
    elif type == 2:
        pil_image = Image.open(BytesIO(base64.b64decode(image_data)))
        return pil_image
    else:
        return cv_img, pil_image


def convert(
        self, mode=None, matrix=None, dither=None, palette=Palette.WEB, colors=256
    ):
        """
        Returns a converted copy of this image. For the "P" mode, this
        method translates pixels through the palette.  If mode is
        omitted, a mode is chosen so that all information in the image
        and the palette can be represented without a palette.

        The current version supports all possible conversions between
        "L", "RGB" and "CMYK." The ``matrix`` argument only supports "L"
        and "RGB".

        When translating a color image to greyscale (mode "L"),
        the library uses the ITU-R 601-2 luma transform::

            L = R * 299/1000 + G * 587/1000 + B * 114/1000

        The default method of converting a greyscale ("L") or "RGB"
        image into a bilevel (mode "1") image uses Floyd-Steinberg
        dither to approximate the original image luminosity levels. If
        dither is ``None``, all values larger than 127 are set to 255 (white),
        all other values to 0 (black). To use other thresholds, use the
        :py:meth:`~PIL.Image.Image.point` method.

        When converting from "RGBA" to "P" without a ``matrix`` argument,
        this passes the operation to :py:meth:`~PIL.Image.Image.quantize`,
        and ``dither`` and ``palette`` are ignored.

        :param mode: The requested mode. See: :ref:`concept-modes`.
        :param matrix: An optional conversion matrix.  If given, this
           should be 4- or 12-tuple containing floating point values.
        :param dither: Dithering method, used when converting from
           mode "RGB" to "P" or from "RGB" or "L" to "1".
           Available methods are :data:`Dither.NONE` or :data:`Dither.FLOYDSTEINBERG`
           (default). Note that this is not used when ``matrix`` is supplied.
        :param palette: Palette to use when converting from mode "RGB"
           to "P".  Available palettes are :data:`Palette.WEB` or
           :data:`Palette.ADAPTIVE`.
        :param colors: Number of colors to use for the :data:`Palette.ADAPTIVE`
           palette. Defaults to 256.
        :rtype: :py:class:`~PIL.Image.Image`
        :returns: An :py:class:`~PIL.Image.Image` object.
        """

        self.load()

        has_transparency = self.info.get("transparency") is not None
        if not mode and self.mode == "P":
            # determine default mode
            if self.palette:
                mode = self.palette.mode
            else:
                mode = "RGB"
            if mode == "RGB" and has_transparency:
                mode = "RGBA"
        if not mode or (mode == self.mode and not matrix):
            return self.copy()

        if matrix:
            # matrix conversion
            if mode not in ("L", "RGB"):
                raise ValueError("illegal conversion")
            im = self.im.convert_matrix(mode, matrix)
            new = self._new(im)
            if has_transparency and self.im.bands == 3:
                transparency = new.info["transparency"]

                def convert_transparency(m, v):
                    v = m[0] * v[0] + m[1] * v[1] + m[2] * v[2] + m[3] * 0.5
                    return max(0, min(255, int(v)))

                if mode == "L":
                    transparency = convert_transparency(matrix, transparency)
                elif len(mode) == 3:
                    transparency = tuple(
                        convert_transparency(matrix[i * 4 : i * 4 + 4], transparency)
                        for i in range(0, len(transparency))
                    )
                new.info["transparency"] = transparency
            return new

        if mode == "P" and self.mode == "RGBA":
            return self.quantize(colors)

        trns = None
        delete_trns = False
        # transparency handling
        if has_transparency:
            if (self.mode in ("1", "L", "I") and mode in ("LA", "RGBA")) or (
                self.mode == "RGB" and mode == "RGBA"
            ):
                # Use transparent conversion to promote from transparent
                # color to an alpha channel.
                new_im = self._new(
                    self.im.convert_transparent(mode, self.info["transparency"])
                )
                del new_im.info["transparency"]
                return new_im
            elif self.mode in ("L", "RGB", "P") and mode in ("L", "RGB", "P"):
                t = self.info["transparency"]
                if isinstance(t, bytes):
                    # Dragons. This can't be represented by a single color
                    warnings.warn(
                        "Palette images with Transparency expressed in bytes should be "
                        "converted to RGBA images"
                    )
                    delete_trns = True
                else:
                    # get the new transparency color.
                    # use existing conversions
                    core = ImportError("The _imaging C module is not installed.")
                    trns_im = Image()._new(core.new(self.mode, (1, 1)))
                    if self.mode == "P":
                        trns_im.putpalette(self.palette)
                        if isinstance(t, tuple):
                            err = "Couldn't allocate a palette color for transparency"
                            try:
                                t = trns_im.palette.getcolor(t, self)
                            except ValueError as e:
                                if str(e) == "cannot allocate more than 256 colors":
                                    # If all 256 colors are in use,
                                    # then there is no need for transparency
                                    t = None
                                else:
                                    raise ValueError(err) from e
                    if t is None:
                        trns = None
                    else:
                        trns_im.putpixel((0, 0), t)

                        if mode in ("L", "RGB"):
                            trns_im = trns_im.convert(mode)
                        else:
                            # can't just retrieve the palette number, got to do it
                            # after quantization.
                            trns_im = trns_im.convert("RGB")
                        trns = trns_im.getpixel((0, 0))

            elif self.mode == "P" and mode in ("LA", "PA", "RGBA"):
                t = self.info["transparency"]
                delete_trns = True

                if isinstance(t, bytes):
                    self.im.putpalettealphas(t)
                elif isinstance(t, int):
                    self.im.putpalettealpha(t, 0)
                else:
                    raise ValueError("Transparency for P mode should be bytes or int")

        if mode == "P" and palette == Palette.ADAPTIVE:
            im = self.im.quantize(colors)
            new = self._new(im)
            from . import ImagePalette

            new.palette = ImagePalette.ImagePalette("RGB", new.im.getpalette("RGB"))
            if delete_trns:
                # This could possibly happen if we requantize to fewer colors.
                # The transparency would be totally off in that case.
                del new.info["transparency"]
            if trns is not None:
                try:
                    new.info["transparency"] = new.palette.getcolor(trns, new)
                except Exception:
                    # if we can't make a transparent color, don't leave the old
                    # transparency hanging around to mess us up.
                    del new.info["transparency"]
                    warnings.warn("Couldn't allocate palette entry for transparency")
            return new

        # colorspace conversion
        if dither is None:
            dither = Dither.FLOYDSTEINBERG

        try:
            im = self.im.convert(mode, dither)
        except ValueError:
            try:
                # normalize source image and try again
                im = self.im.convert(getmodebase(self.mode))
                im = im.convert(mode, dither)
            except KeyError as e:
                raise ValueError("illegal conversion") from e

        new_im = self._new(im)
        if mode == "P" and palette != Palette.ADAPTIVE:
            from . import ImagePalette

            new_im.palette = ImagePalette.ImagePalette("RGB", list(range(256)) * 3)
        if delete_trns:
            # crash fail if we leave a bytes transparency in an rgb/l mode.
            del new_im.info["transparency"]
        if trns is not None:
            if new_im.mode == "P":
                try:
                    new_im.info["transparency"] = new_im.palette.getcolor(trns, new_im)
                except ValueError as e:
                    del new_im.info["transparency"]
                    if str(e) != "cannot allocate more than 256 colors":
                        # If all 256 colors are in use,
                        # then there is no need for transparency
                        warnings.warn(
                            "Couldn't allocate palette entry for transparency"
                        )
            else:
                new_im.info["transparency"] = trns
        return new_im

def np_to_base64(img_np):
    """
    It converts a numpy array to a base64 string
    
    :param img_np: the numpy array of the image
    :return: A string of the image in base64 format.
    """
    """
    Convert numpy image (RGB) to base64 string
    """
    img = Image.fromarray(img_np.astype('uint8'), 'RGB')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return u"data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("ascii")


def img_to_array(img, data_format=None, dtype=None):
    """Converts a PIL Image instance to a Numpy array.

    Usage:

    ```python
    from PIL import Image
    img_data = np.random.random(size=(100, 100, 3))
    img = tf.keras.preprocessing.image.array_to_img(img_data)
    array = tf.keras.preprocessing.image.img_to_array(img)
    ```


    Args:
        img: Input PIL Image instance.
        data_format: Image data format, can be either `"channels_first"` or
            `"channels_last"`. Defaults to `None`, in which case the global setting
            `tf.keras.backend.image_data_format()` is used (unless you changed it,
            it defaults to `"channels_last"`).
        dtype: Dtype to use. Default to `None`, in which case the global setting
            `tf.keras.backend.floatx()` is used (unless you changed it, it defaults
            to `"float32"`).

    Returns:
        A 3D Numpy array.

    Raises:
        ValueError: if invalid `img` or `data_format` is passed.
    """

    if data_format is None:
        data_format = backend.image_data_format()
    if dtype is None:
        dtype = backend.floatx()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError(f'Unknown data_format: {data_format}')
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=dtype)
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError(f'Unsupported image shape: {x.shape}')
    return x

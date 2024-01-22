import numpy as np
import fractions
from src import interp
from src.convolution import conv, insert_padding
from src import kernels as krn
from src import utilities as utils


def enlarge(img, multiplier):
    """
    Enlarges image by given multiplier

    :param img: Original image
    :param multiplier: Multiplier
    :return: Enlarged image
    """
    if len(img) < 1:
        print("Err: Empty image sent to the enlarge func!")
        return 0

    res_img = interp.interpolate_image(img, krn.rectangular_kernel, multiplier)

    res_img = np.asarray(res_img).astype(np.uint8)

    return res_img


def change_img_size(img, multiplier):
    if multiplier > 10 or multiplier <= 0:
        print("Err: Multiplier must be between 0 and 10!")
        return 0
    if utils.count_decimal_places(multiplier) > 1 or multiplier == 0:
        print("Err: Multiplier can't have more than 1 decimal place!")
        return 0
    if len(img) < 1:
        print("Err: Empty image sent to the change_img_size func!")
        return 0

    """ Changing multiplier value from ratio to fraction """
    frac = fractions.Fraction(multiplier).limit_denominator()
    upscale, downscale = frac.numerator, frac.denominator
    print("Upscale: " + str(upscale) + "\n" +
          "Downscale: " + str(downscale))

    """ RGB image handling """
    if len(img.shape) > 2 and img.shape[2] > 1:
        final_img = np.zeros((int(len(img) * multiplier), int(len(img[0]) * multiplier), img.shape[2])).astype(np.uint8)
        for ch in range(img.shape[2]):
            final_img[:, :, ch] = change_img_size(img[:, :, ch], multiplier)
        return np.clip(final_img / final_img.max() * 255, 0, 255).astype(np.uint8)

    """ Up-scaling and down-scaling image for float multiplier """
    upscaled_img = interp.interpolate_image(img, krn.rectangular_kernel, upscale)
    downscaled_img = conv(upscaled_img, krn.mean_kernel, 9, step=downscale, padding=True)

    return np.clip(downscaled_img, 0, 255).astype(np.uint8)


def max_pooling(img, krn_area: int, step: int, padding=False):
    """
    Convolution function that iterates through image and kernel matrix, doing matrix multiplication and putting results
    into result_image.

    :param img: Original image
    :param krn_area: Area of the ones kernel (int)
    :param step: Number of pixels between next pixel reading (it determines result_matrix size)
    :param padding: Number of pixels to put around original image (to prevent image shrinking)
    :return:
    """
    if len(img) < 1:
        print("Err: Empty image sent to the convolution func!")
        return 0

    res_img = np.zeros_like(img)
    kernel = np.ones((krn_area, krn_area))

    if len(img.shape) > 2 and img.shape[2] > 1:
        for ch in range(img.shape[2]):
            res_img[:, :, ch] = conv(img[:, :, ch], krn_area, step, padding)
        res_img = np.clip(res_img, 0, 255).astype(np.uint8)
        return res_img

    # Apply padding if needed
    if not padding:
        result_rows = int(np.ceil((len(img) - len(kernel) + 1) / step))
        result_cols = np.floor(np.ceil((len(img[0]) - len(kernel[0]) + 1) / step)).astype(int)
        padding_value = 0
    else:
        padding_value = np.floor(len(kernel) / 2).astype(int)
        img = insert_padding(img, padding_value, 255)  # Add padding to the processed image. Last argument determines COLOR of padding

        result_rows = int(np.ceil((len(img) - np.floor(len(kernel)) + 1) / step))
        result_cols = int(np.ceil((len(img[0]) - np.floor(len(kernel[0])) + 1) / step))

    # Convolution process
    result_image = np.zeros((result_rows, result_cols))
    for i in range(result_image.shape[0]):
        for j in range(result_image.shape[1]):
            result_image[i, j] = np.max(img[i*step:i*step+len(kernel), j*step:j*step+len(kernel[0])] * kernel ).astype(int)

    # Normalization
    result_image = np.clip(result_image / result_image.max() * 255, 0, 255).astype(np.uint8)

    return result_image

import numpy as np


def insert_padding(img, padding, value=0):
    for p in range(padding):
        img = np.insert(img, 0, value * np.ones(len(img[0])), axis=0)
        img = np.append(img, [value * np.ones(len(img[0]))], axis=0)

    for p in range(padding):
        # for row in range(len(image)):
        img = np.insert(img, 0, value, axis=1)
        img = np.append(img, value * np.ones((len(img), 1)), axis=1)

    return img


def conv(img, kernel, multiplier, step, padding=False):
    """
    Convolution function that iterates through image and kernel matrix, doing matrix multiplication and putting results
    into result_image.

    :param img: Original image
    :param kernel: Convolution kernel (squared matrix, can be any size smaller than the original image)
    :param multiplier: Kernel's matrix multiplier (1/multiplier)*matrix
    :param step: Number of pixels between next pixel reading (it determines result_matrix size)
    :param padding: Number of pixels to put around original image (to prevent image shrinking)
    :return:
    """
    if len(img) < 1:
        print("Err: Empty image sent to the convolution func!")
        return 0

    res_img = np.zeros_like(img)

    if len(img.shape) > 2 and img.shape[2] > 1:
        for ch in range(img.shape[2]):
            res_img[:, :, ch] = conv(img[:, :, ch], kernel, multiplier, step, padding)
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
            result_image[i, j] = np.abs(np.sum(img[i*step:i*step+len(kernel), j*step:j*step+len(kernel[0])] * kernel / multiplier)).astype(int)

    # Normalization
    result_image = np.clip(result_image / result_image.max() * 255, 0, 255).astype(np.uint8)

    return result_image

import numpy as np
from src.utilities import find_closest, find_closest_indexes


def interpolate(x_arr, y_arr, x_result, kernel, interp_range=1):
    """
    Interpolate data using the specified kernel

    Args:
        :param x_arr: The x-values of the original data (function).
        :param y_arr: The y-values of the original data (function).
        :param x_result: The x-values for interpolation.
        :param kernel: The interpolation kernel function
        :param interp_range: The range of points near the interpolated point on which we perform interpolation  # UPDATE

    :return: numpy.ndarray: The interpolated y-values
    """
    if len(x_arr) < 1:
        print("Err: x_arr can't be empty!")
        return 0

    y_result = []
    # range_len = np.abs(x_arr[0] - x_arr[-1])  # Length of measured range
    # distance = range_len / (len(x_result) - 1)  # Distance between two points

    for i in range(len(x_result)):
        if interp_range > 0:
            temp_x_arr = find_closest(x_result[i], x_arr, interp_range)
            temp_y_arr = []
            for index in find_closest_indexes(x_result[i], x_arr, interp_range):
                temp_y_arr.append(y_arr[int(index)])
        else:
            temp_x_arr = x_arr
            temp_y_arr = y_arr

        weights = kernel(x_result[i] - temp_x_arr)
        weights = weights.astype(float)
        total_weight = np.sum(weights)

        if total_weight != 0:
            weights /= total_weight

            y = np.sum(weights * temp_y_arr)
            y_result.append(y)
        else:
            y_result.append(0)

    return y_result


def interpolate_image(image, kernel, multiplier):
    # Create copy of an original image
    image = np.copy(image)

    # Calculate new dimensions
    new_dims = tuple(dim * multiplier for dim in image.shape)

    # Initialize new image
    new_image = np.zeros(new_dims)

    # Iterate over new image
    for i in range(new_image.shape[0]):
        for j in range(new_image.shape[1]):
            # Calculate corresponding index in original image
            orig_i = i / multiplier
            orig_j = j / multiplier

            # Calculate surrounding indices in original image
            # i_range = range(int(np.floor(orig_i - 0.5)), int(np.ceil(orig_i + 0.5)))
            # j_range = range(int(np.floor(orig_j - 0.5)), int(np.ceil(orig_j + 0.5)))
            i_range = range(max(0, int(np.floor(orig_i - 0.5))), min(image.shape[0], int(np.ceil(orig_i + 0.5))))
            j_range = range(max(0, int(np.floor(orig_j - 0.5))), min(image.shape[1], int(np.ceil(orig_j + 0.5))))

            # Calculate weights
            weights = np.array([kernel(np.sqrt((orig_i - ii) ** 2 + (orig_j - jj) ** 2)) for ii in i_range for jj in j_range]).astype(float)

            # Normalize weights
            if np.sum(weights) != 0:
                weights /= np.sum(weights)
            else:
                weights = np.zeros_like(weights)

            # Calculate new pixel value
            new_pixel_value = np.sum(weights * np.array([image[ii, jj] for ii in i_range for jj in j_range]))

            # Assign new pixel value to new image
            new_image[i, j] = new_pixel_value

    return new_image

from matplotlib import pyplot as plt
import numpy as np
import skimage as sk

""" CONVOLUTION KERNELS """

mean_kernel = [[1, 1, 1],
               [1, 1, 1],
               [1, 1, 1]]

edge_kernel = [[-1, -1, -1],
               [-1, 8, -1],
               [-1, -1, -1]]

gauss_kernel = [[1, 4, 6, 4, 1],
                [4, 16, 24, 16, 4],
                [6, 24, 36, 24, 6],
                [4, 16, 24, 16, 4],
                [1, 4, 6, 4, 1]]


def insert_padding(img, padding, value=0):
    for p in range(padding):
        img = np.insert(img, 0, value * np.ones(len(img[0])), axis=0)
        img = np.append(img, [value * np.ones(len(img[0]))], axis=0)

    for p in range(padding):
        # for row in range(len(image)):
        img = np.insert(img, 0, value, axis=1)
        img = np.append(img, value * np.ones((len(img), 1)), axis=1)

    return img


""" CONVOLUTION FUNCTION """


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

    if padding == 0:
        result_rows = int(round(len(img) / step) - len(kernel) + 1)
        result_cols = np.floor(round(len(img[0]) / step) - len(kernel[0]) + 1).astype(int)
    else:
        padding_value = np.floor(len(kernel) / 2).astype(int)
        img = insert_padding(img, padding_value, 0)
        print(img)

        result_rows = int(round(len(img) / step) - np.floor(len(kernel)) + 1)
        result_cols = int(round(len(img[0]) / step) - np.floor(len(kernel[0])) + 1)

    print("Original img width: " + str(len(img[0])) + "\n" +
          "Original img height: " + str(len(img)) + "\n" +
          "Result img width: " + str(result_cols) + "\n" +
          "Result img height: " + str(result_rows) + "\n")

    result_image = np.zeros((result_rows, result_cols))
    for row in range(result_rows):
        for x in range(result_cols):
            sum = 0
            for row_kernel in range(len(kernel)):
                for x_kernel in range(len(kernel[row_kernel])):
                    if row * step + row_kernel < len(img) and x * step + x_kernel < len(img[0]):
                        sum += kernel[row_kernel][x_kernel] * img[row * step + row_kernel][x * step + x_kernel]

            if row < result_rows and x < result_cols:
                sum = round(sum / multiplier)
                result_image[row][x] = sum
                # print(sum)

    return result_image


if __name__ == "__main__":
    image = sk.data.coins()

    # image = [[255, 255, 255, 255, 255, 255, 255, 255, 255],
    #          [255, 255, 255, 255, 128, 255, 255, 255, 255],
    #          [255, 255, 255, 48, 64, 48, 255, 255, 255],
    #          [255, 255, 48, 32, 0, 32, 48, 255, 255],
    #          [255, 128, 64, 0, 0, 0, 64, 128, 255],
    #          [255, 255, 48, 32, 0, 32, 48, 255, 255],
    #          [255, 255, 255, 48, 64, 48, 255, 255, 255],
    #          [255, 255, 255, 255, 128, 255, 255, 255, 255],
    #          [255, 255, 255, 255, 255, 255, 255, 255, 255]]

    """########### PLOTTING PART ###########"""
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    original_img_ax = ax[0][0]
    first_img_ax = ax[0][1]
    second_img_ax = ax[1][0]
    third_img_ax = ax[1][1]

    original_img_ax.imshow(image, cmap='gray')
    original_img_ax.set_title("Original image")

    first_img_ax.imshow(conv(image, mean_kernel, 9, 1, True), cmap='gray')
    first_img_ax.set_title("Mean kernel")

    third_img_ax.imshow(conv(image, gauss_kernel, 256, 1, True), cmap='gray')
    third_img_ax.set_title("Gaussian blur kernel")

    second_img_ax.imshow(conv(image, edge_kernel, 1, 1, False), cmap='gray')
    second_img_ax.set_title("Edge detection kernel")

    plt.show()

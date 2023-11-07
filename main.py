from matplotlib import pyplot as plt
import numpy as np
import skimage as sk


def conv(img, kernel, multiplier, step, padding):
    result_rows = round(len(img) / step)
    result_cols = round(len(img[0]) / step)
    print("Original img width: " + str(len(img[0])) + "\n" +
          "Original img height: " + str(len(img)) + "\n" +
          "Result img width: " + str(result_cols) + "\n" +
          "Result img height: " + str(result_rows) + "\n")
    result_image = np.zeros((result_rows, result_cols))
    for row in range(len(img)):
        for x in range(len(img[row])):
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

    print(image)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    original_img_ax = ax[0][0]
    first_img_ax = ax[0][1]
    second_img_ax = ax[1][0]
    third_img_ax = ax[1][1]

    original_img_ax.imshow(image, cmap='gray')
    original_img_ax.set_title("Original image")

    first_img_ax.imshow(conv(image, mean_kernel, 9, 1, 0), cmap='gray')
    first_img_ax.set_title("Mean kernel")

    third_img_ax.imshow(conv(image, gauss_kernel, 256, 1, 0), cmap='gray')
    third_img_ax.set_title("Gaussian blur kernel")

    second_img_ax.imshow(conv(image, edge_kernel, 1, 1, 0), cmap='gray')
    second_img_ax.set_title("Edge detection kernel")
    plt.show()

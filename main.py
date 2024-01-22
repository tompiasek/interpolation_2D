from matplotlib import pyplot as plt
import numpy as np
import skimage as sk
from src import interp
from src import convolution
from src import scaling
from src import kernels as krn
from src import utilities as utils
from src.convolution import conv
from src.kernels import mean_kernel, edge_kernel, gauss_kernel


if __name__ == "__main__":
    image = sk.data.coins()
    img_rgb = sk.io.imread("img/panda.jpg")
    img_rgb = img_rgb[:, :, :3]

    # image = [[255, 255, 255, 255, 255, 255, 255, 255, 255],
    #          [255, 255, 255, 255, 128, 255, 255, 255, 255],
    #          [255, 255, 255, 48, 64, 48, 255, 255, 255],
    #          [255, 255, 48, 32, 0, 32, 48, 255, 255],
    #          [255, 128, 64, 0, 0, 0, 64, 128, 255],
    #          [255, 255, 48, 32, 0, 32, 48, 255, 255],
    #          [255, 255, 255, 48, 64, 48, 255, 255, 255],
    #          [255, 255, 255, 255, 128, 255, 255, 255, 255],
    #          [255, 255, 255, 255, 255, 255, 255, 255, 255]]

    """ INTERPOLATION ENLARGING IMAGES """
    img_x16 = sk.io.imread("img/coins_x16.jpg")
    img_x16_diff = sk.io.imread("img/coins_x16_sequence.jpg")

    _, ax6 = plt.subplots(1, 3, figsize=(12, 8))

    ax6[0].imshow(image, cmap="gray")
    ax6[0].set_title("Original image " + str(image.shape) + "px")
    ax6[1].imshow(img_x16, cmap="gray")
    ax6[1].set_title("x16 image " + str(img_x16.shape) + "px")
    ax6[2].imshow(img_x16_diff, cmap="gray")
    ax6[2].set_title("x16 sequence image " + str(img_x16_diff.shape) + "px")

    mse_x16 = sk.metrics.mean_squared_error(img_x16, img_x16_diff)

    print("MSE x16: " + str(mse_x16) + "\n")


    """ FLOAT MULTIPLIER IMAGES """
    # img_x1_5 = change_img_size(image, 1.5)
    # img_x0_4 = change_img_size(image, 0.4)
    # img_rgb_x1_5 = scaling.change_img_size(img_rgb, 1.5)
    #
    # PLOTTING
    #
    # _, ax3 = plt.subplots(1, 2, figsize=(12, 8))
    #
    # ax3[0].imshow(img_rgb)
    # ax3[0].set_title("Original image (" + str(img_rgb.shape) + "px")
    # ax3[1].imshow(img_rgb_x1_5)
    # ax3[1].set_title("Float scaling [RGB] (" + str(img_rgb_x1_5.shape) + "px")
    # sk.io.imsave("img/panda_x1_5.jpg", img_rgb_x1_5.astype(np.uint8))
    #
    # ax3[1].imshow(img_x1_5, cmap="gray")
    # ax3[1].set_title("x1.5 image (" + str(img_x1_5.shape) + "px)")
    #
    # ax3[2].imshow(img_x0_4, cmap="gray")
    # ax3[2].set_title("x0.4 image (" + str(img_x0_4.shape) + "px)")

    """########### PLOTTING PART ###########"""
    #
    # """ GRAYSCALE PLOT """
    #
    # _, ax = plt.subplots(2, 2, figsize=(12, 8))
    #
    # ax[0][0].imshow(image, cmap='gray')
    # ax[0][0].set_title("Original image " + str(image.shape) + "px")
    #
    # img_mean = conv(image, mean_kernel, 9, 1, True)
    # ax[0][1].imshow(img_mean, cmap='gray')
    # ax[0][1].set_title("Mean kernel " + str(img_mean.shape) + "px")
    #
    # img_gauss = conv(image, gauss_kernel, 256, 1, True)
    # ax[1][0].imshow(img_gauss, cmap='gray')
    # ax[1][0].set_title("Gaussian blur kernel " + str(img_gauss.shape) + "px")
    #
    # img_edge = conv(image, edge_kernel, 1, 1, True)
    # ax[1][1].imshow(img_edge, cmap='gray')
    # ax[1][1].set_title("Edge detection kernel " + str(img_edge.shape) + "px")
    #
    # """ RGB PLOT """
    #
    # _, ax2 = plt.subplots(2, 2, figsize=(12, 8))
    #
    # ax2[0][0].imshow(img_rgb)
    # ax2[0][0].set_title("Original image " + str(img_rgb.shape) + "px")
    #
    # img_mean_rgb = conv(img_rgb, mean_kernel, 9, 1, True)
    # ax2[0][1].imshow(img_mean_rgb)
    # ax2[0][1].set_title("Mean kernel " + str(img_mean_rgb.shape) + "px")
    #
    # img_edge_rgb = conv(img_rgb, edge_kernel, 1, 1, True)
    # ax2[1][0].imshow(np.clip(img_edge_rgb * 2, 0, 255))
    # ax2[1][0].set_title("Edge detection kernel " + str(img_edge_rgb.shape) + "px")
    #
    # img_gauss_rgb = conv(img_rgb, gauss_kernel, 256, 1, True)
    # ax2[1][1].imshow(img_gauss_rgb)
    # ax2[1][1].set_title("Gaussian blur kernel " + str(img_gauss_rgb.shape) + "px")

    plt.show()

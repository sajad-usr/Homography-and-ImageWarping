import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def get_transformation(src_points):
    """
    This function takes source points and returns the transformation.

    :param src_points: Source points. Input points should be in this order: First point is the one which will be mapped
    to top-left corner. After that are the ones that map to top-right, bottom-right, and bottom-left corner, respectively.
    :return transformation_matrix: The transformation
    :return (width, height): width and height of the image
    """
    # Width up = distance between top-left and top-right corners
    width_up = np.sqrt((src_points[0, 0] - src_points[1, 0]) ** 2 + (src_points[0, 1] - src_points[1, 1]) ** 2)
    # Width down = distance between bottom-left and bottom-right corners
    width_down = np.sqrt((src_points[2, 0] - src_points[3, 0]) ** 2 + (src_points[2, 1] - src_points[3, 1]) ** 2)
    # Set width as maximum of width up and width down
    width = int(np.ceil(np.max((width_up, width_down))))
    # Height left = distance between top-left and bottom-left corners
    height_left = np.sqrt((src_points[0, 0] - src_points[3, 0]) ** 2 + (src_points[0, 1] - src_points[3, 1]) ** 2)
    # Height right = distance between top-right and bottom-right corners
    height_right = np.sqrt((src_points[2, 0] - src_points[1, 0]) ** 2 + (src_points[2, 1] - src_points[1, 1]) ** 2)
    # Set height as maximum of height left and height right
    height = int(np.ceil(np.max((height_right, height_left))))

    # Destination points
    dst_points = np.array([[0, 0],
                           [0, width - 1],
                           [height - 1, width - 1],
                           [height - 1, 0]])

    # The following lines result in same transformation.
    # transformation_matrix = cv.getPerspectiveTransform(src_points.astype(np.float32), dst_points.astype(np.float32))
    transformation_matrix, status = cv.findHomography(src_points.astype(np.float32), dst_points.astype(np.float32))
    return transformation_matrix, (width, height)


def warping(image, transformation_matrix, final_size):
    """
    This function takes an image, a transformation matrix and the final size of the image and does inverse warping
    :param image: Input image
    :param transformation_matrix: transformation matrix
    :param final_size: final size of the image
    :return:
    """
    back_trans = np.linalg.pinv(transformation_matrix)  # pinv: in case transformation_matrix is not invertible.
    # I think it's always invertible.
    width, height = final_size
    final_img = np.zeros((height, width, 3))

    for i in range(height):
        for j in range(width):
            # Warping in inverse direction
            back_point = back_trans @ np.array([i, j, 1]).reshape((-1, 1))
            # Divide the points by third component
            back_point = back_point[0:2] / back_point[2]
            x_back = back_point[0]
            y_back = back_point[1]
            # Bilinear interpolation
            x = int(np.floor(x_back))
            y = int(np.floor(y_back))
            a = x_back - x
            b = y_back - y

            final_img[i, j, :] = (1 - b) * (1 - a) * image[x, y, :] + \
                                 (1 - b) * a * image[x + 1, y, :] + \
                                 (1 - a) * b * image[x, y + 1, :] + \
                                 a * b * image[x + 1, y + 1, :]
    return final_img


img = cv.imread('books.jpg', cv.IMREAD_COLOR)
if img is None:
    raise Exception("Couldn't load the image")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

book1_points = np.array([[208, 666],
                         [395, 599],
                         [288, 316],
                         [103, 382]])

book2_points = np.array([[742, 364],
                         [710, 153],
                         [427, 205],
                         [467, 412]])

book3_points = np.array([[968, 813],
                         [1100, 609],
                         [795, 420],
                         [667, 623]])

book1_transformation, book1_size = get_transformation(book1_points)
book2_transformation, book2_size = get_transformation(book2_points)
book3_transformation, book3_size = get_transformation(book3_points)

warped_1 = warping(img, book1_transformation, book1_size).astype(np.uint8)
warped_2 = warping(img, book2_transformation, book2_size).astype(np.uint8)
warped_3 = warping(img, book3_transformation, book3_size).astype(np.uint8)

print("First book: ", book1_transformation)
print("Second book: ", book2_transformation)
print("Third book: ", book3_transformation)

plt.imsave('res16.jpg', warped_1)
plt.imsave('res17.jpg', warped_2)
plt.imsave('res18.jpg', warped_3)

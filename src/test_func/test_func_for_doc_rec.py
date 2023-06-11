import cv2
import numpy as np
from fast_slic import Slic
from matplotlib import pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.signal import savgol_filter
from skimage.filters import threshold_local

from src.main.document_recovery import sort_points, line_point_cross_product


# Попытка интерполяции с учётом карты глубины
def interpolate_2(x, y, n_points, depth):
    phi = np.linspace(0, 1, n_points)
    weights = depth / np.max(depth)  # масштабирование глубины в диапазон [0, 1]
    print(depth)
    print(weights)
    tck, u = splprep([x, y], s=0, w=weights)
    new_points = splev(phi, tck)
    x1, y1 = new_points[0], new_points[1]
    return x1, y1


# Попытка интерполяции с учётом карты глубины
def opposite_interpolate_2(side_1, side_2, side_1_depth, side_2_depth, smooth=False):
    side_1_x, side_1_y = zip(*side_1)
    side_2_x, side_2_y = zip(*side_2)
    pts_num_lr = np.max((len(side_1_x), len(side_2_x)))

    new_side_1_x, new_side_1_y = interpolate_2(side_1_x, side_1_y, pts_num_lr, side_1_depth)
    new_side_2_x, new_side_2_y = interpolate_2(side_2_x, side_2_y, pts_num_lr, side_2_depth)

    if smooth:
        window = 51
        coef = 5
        new_side_1_x = savgol_filter(new_side_1_x, window, coef)
        new_side_1_y = savgol_filter(new_side_1_y, window, coef)
        new_side_2_x = savgol_filter(new_side_2_x, window, coef)
        new_side_2_y = savgol_filter(new_side_2_y, window, coef)
    return new_side_1_x, new_side_1_y, new_side_2_x, new_side_2_y


# Попытка поиска edge с учётом карты глубины
def find_edges_2(img_test, contour, convex, depth_map):
    edges = np.squeeze(np.array(contour), axis=1)
    corners = np.reshape(np.array(convex), (4, 2))

    corners = sort_points(corners)
    diag_lines = np.array([[corners[0], corners[2]], [corners[1], corners[3]]])

    for num, line in enumerate(diag_lines):
        cv2.line(img_test, line[0], line[1], (0, 255 - 100 * num, 0), 2)

    plt.imshow(img_test)
    plt.show()

    left = [edge for edge in np.array(edges).tolist() if
            (line_point_cross_product(diag_lines[0], edge) < 0 < line_point_cross_product(diag_lines[1], edge))]
    bottom = [edge for edge in np.array(edges).tolist() if
              (line_point_cross_product(diag_lines[0], edge) < 0 and line_point_cross_product(diag_lines[1], edge) < 0)]
    right = [edge for edge in np.array(edges).tolist() if
             (line_point_cross_product(diag_lines[0], edge) > 0 > line_point_cross_product(diag_lines[1], edge))]
    top = [edge for edge in np.array(edges).tolist() if
           (line_point_cross_product(diag_lines[0], edge) > 0 and line_point_cross_product(diag_lines[1], edge) > 0)]

    # compute depth for each side
    left_depth = [depth_map[edge[1], edge[0]] for edge in left]
    bottom_depth = [depth_map[edge[1], edge[0]] for edge in bottom]
    right_depth = [depth_map[edge[1], edge[0]] for edge in right]
    top_depth = [depth_map[edge[1], edge[0]] for edge in top]

    # sort edges by increasing x-coordinate (left to right) and increasing y-coordinate (top to bottom)
    top.sort()
    right = right[::-1]

    return left, right, top, bottom, left_depth, right_depth, top_depth, bottom_depth


def average_image_vectorized(img, img_seg):
    av_img = np.zeros_like(img)
    unique_segments = np.unique(img_seg)
    for seg in unique_segments:
        mask = img_seg == seg
        av_img[mask] = np.mean(img[mask], axis=0)
    return cv2.normalize(av_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def get_paper_mask(image):
    kernel = np.ones((5, 5), np.uint8)
    morpho = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=3)
    bw = cv2.cvtColor(morpho, cv2.COLOR_BGR2RGB)
    slic = Slic(num_components=500, compactness=40)
    paper_seg = slic.iterate(bw, 20)
    plt.imshow(paper_seg)
    plt.show()
    nn = average_image_vectorized(bw, paper_seg)
    plt.imshow(nn)
    plt.show()

    nn = cv2.cvtColor(nn, cv2.COLOR_RGB2HSV)
    nn_h_0 = nn[:, :, 0]
    nn_h_1 = nn[:, :, 1]
    nn_h = cv2.addWeighted(nn_h_1, 1, nn_h_0, 0.4, 0.1)
    plt.imshow(nn_h)
    plt.show()

    gauss = cv2.GaussianBlur(nn_h, (5, 5), cv2.BORDER_DEFAULT)
    _, bw2 = cv2.threshold(gauss, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw2 = cv2.bitwise_not(bw2)
    plt.imshow(bw2)
    plt.show()
    return bw2


def interpolation_approximation(img, shape, num_ln, degree, num_p, linfit_x, linfit_y):
    polys_bt = []
    shape_flag = 0 if shape == img.shape[0] else 1

    for i in range(1, num_ln + 1):
        x = linfit_x(i)
        y = linfit_y(i)
        z = np.polyfit(y, x, degree) if shape_flag == 0 else np.polyfit(x, y, degree)
        f = np.poly1d(z)

        if shape_flag == 0:
            y_new = np.linspace(0, shape, num_p)
            x_new = f(y_new)
        else:
            x_new = np.linspace(0, shape, num_p)
            y_new = f(x_new)

        poly_coor = np.hstack((np.reshape(x_new, (x_new.shape[0], 1)), np.reshape(y_new, (y_new.shape[0], 1))))
        polys_bt.append(poly_coor)
        plt.plot(x_new, y_new)
    return polys_bt


def interpolate(x, y, n_points):
    x_array = np.array(x)
    y_array = np.array(y)
    xy = [x_array, y_array]

    phi = np.linspace(0, 1, n_points)
    tck, u = splprep(xy, s=0)
    print("True")
    new_points = splev(phi, tck)
    x1, y1 = new_points[0], new_points[1]
    return x1, y1


def get_black_white_scan(crop_warped_image):
    crop_warped_image_gray = cv2.cvtColor(crop_warped_image, cv2.COLOR_RGB2GRAY)
    T = threshold_local(crop_warped_image_gray, 11, offset=10, method="gaussian")
    return (crop_warped_image_gray > T).astype("uint8") * 255

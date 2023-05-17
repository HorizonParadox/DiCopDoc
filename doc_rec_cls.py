import cv2
import numpy as np
from shapely.geometry import Polygon
from fast_slic import Slic
from matplotlib import pyplot as plt
from numpy import ndarray
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.interpolate import splprep, splev, interp1d
from scipy import spatial
import itertools
from skimage.filters import threshold_local
from ultralytics import YOLO
from itertools import product
from scipy.interpolate import griddata


def get_black_white_scan(crop_warped_image):
    crop_warped_image_gray = cv2.cvtColor(crop_warped_image, cv2.COLOR_RGB2GRAY)
    T = threshold_local(crop_warped_image_gray, 11, offset=10, method="gaussian")
    return (crop_warped_image_gray > T).astype("uint8") * 255


def remap_image(img, height, width, points_set, xy_pairs, depth):
    comp_w = complex(f'{width}j')
    comp_h = complex(f'{height}j')
    grid_x, grid_y = np.mgrid[0:(height - 1):comp_h, 0:(width - 1):comp_w]
    grid_z = griddata(np.fliplr(xy_pairs), np.fliplr(points_set), (grid_x, grid_y), method='cubic')

    map_x = np.append([], [ar[:, 1] for ar in grid_z]).reshape(height, width)
    map_y = np.append([], [ar[:, 0] for ar in grid_z]).reshape(height, width)

    map_x_32 = map_x.astype('float32')
    map_y_32 = map_y.astype('float32')

    warped_image = cv2.remap(img, map_x_32, map_y_32, cv2.INTER_CUBIC)
    warped_image = np.fliplr(np.rot90(warped_image, 2))

    return warped_image


def get_fix_grid_coordinates(length, width, num_of_line, shift):
    x = np.linspace(0, width, num_of_line)
    y = np.linspace(0, length, num_of_line)
    xv, yv = np.meshgrid(x, y)
    return np.dstack([xv, yv]).reshape(-1, 2) + shift


def get_intersection_interpolation_approximation(length, width, polys_bt, polys_lr):
    points_set = []
    # tolerance = ceil(length * 0.2) if length > width else ceil(width * 0.2)
    tolerance = 20.0
    print(f'TOLERANCE {tolerance}')
    for line1, line2 in product(polys_bt, polys_lr):
        # Поиск переченения линий интерполяции аппроксимации
        a = find_intersection(line1, line2, tolerance)
        points_set.append([a[0], a[1]])
        plt.plot(round(a[0]), round(a[1]), 'ro')
    return points_set


# Метод наименьших квадратов
def interpolation_approximation_3(img, shape, num_ln, degree, num_p, linfit_x, linfit_y):
    polys_bt = []
    shape_flag = 0 if shape == img.shape[0] else 1
    for i in range(1, num_ln + 1):
        x = linfit_x(i)
        y = linfit_y(i)

        def fun(p, x=x, y=y):
            if shape_flag == 0:
                return p[0] * y ** 2 + p[1] * y + p[2] - x
            else:
                return p[0] * x ** 2 + p[1] * x + p[2] - y

        p0 = np.zeros(degree + 1)
        result = least_squares(fun, p0)
        if shape_flag == 0:
            y_new = np.linspace(0, shape, num_p)
            x_new = result.x[0] * y_new ** 2 + result.x[1] * y_new + result.x[2]
        else:
            x_new = np.linspace(0, shape, num_p)
            y_new = result.x[0] * x_new ** 2 + result.x[1] * x_new + result.x[2]
        poly_coor = np.hstack((np.reshape(x_new, (x_new.shape[0], 1)), np.reshape(y_new, (y_new.shape[0], 1))))
        polys_bt.append(poly_coor)
        plt.plot(x_new, y_new)

    return polys_bt


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


def get_lines_of_approximation(img, num, l_x, l_y, r_x, r_y, b_x, b_y, t_x, t_y):
    linfit_x_lr = interp1d([1, num], np.vstack((l_x, r_x)), axis=0)
    linfit_y_lr = interp1d([1, num], np.vstack((l_y, r_y)), axis=0)

    linfit_x_bt = interp1d([1, num], np.vstack((b_x, t_x)), axis=0)
    linfit_y_bt = interp1d([1, num], np.vstack((b_y, t_y)), axis=0)

    for i in range(1, num + 1):
        plt.imshow(img)
        plt.plot(linfit_x_bt(i), linfit_y_bt(i))
        plt.plot(linfit_x_lr(i), linfit_y_lr(i))
    plt.show()
    return linfit_x_bt, linfit_y_bt, linfit_x_lr, linfit_y_lr


def plot_polynomial(img, concave, conv, cont):
    cv2.drawContours(img, conv, -1, (255, 0, 0), 8)
    cv2.drawContours(img, cont, -1, (0, 255, 0), 8)
    cv2.drawContours(img, concave, -1, (0, 0, 255), 8)
    plt.imshow(img)
    plt.show()


def plot_colorful_edges(img, left, right, top, bottom):
    for point in left:
        cv2.circle(img, point, 2, [0, 0, 255], -1)
    for point in bottom:
        cv2.circle(img, point, 2, [0, 255, 0], -1)
    for point in right:
        cv2.circle(img, point, 2, [255, 0, 0], -1)
    for point in top:
        cv2.circle(img, point, 2, [0, 255, 255], -1)
    plt.imshow(img)
    plt.show()


def output_resize(img):
    return cv2.resize(src=img, dsize=(0, 0), fx=0.15, fy=0.15)


def average_image_vectorized(img, img_seg):
    av_img = np.zeros_like(img)
    unique_segments = np.unique(img_seg)
    for seg in unique_segments:
        mask = img_seg == seg
        av_img[mask] = np.mean(img[mask], axis=0)
    return cv2.normalize(av_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def get_dimensions(left, right, top, bottom):
    left_length = round(spatial.distance.euclidean(left[0], left[-1]))
    right_length = round(spatial.distance.euclidean(right[0], right[-1]))
    top_length = round(spatial.distance.euclidean(top[0], top[-1]))
    bottom_length = round(spatial.distance.euclidean(bottom[0], bottom[-1]))
    length = int(np.mean([left_length, right_length]))
    width = int(np.mean([top_length, bottom_length]))
    return length, width


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


def find_mask_intersection(seg1, seg2):
    polygon1 = Polygon(seg1)
    polygon2 = Polygon(seg2)
    return polygon1.intersects(polygon2)


def get_paper_masks_yolo(image):
    plt.imshow(image)
    plt.show()
    model = YOLOSegmentation('best.pt')
    bboxes, _, segments, scores = model.detect(image)
    masks = []

    yolo_result = []
    yolo_output = []
    single_image_flag = False
    segments_intersection_flag = False

    if len(segments) == 1:
        single_image_flag = True
        yolo_output = [[bboxes[0], segments[0], scores[0]]]
    elif len(segments) > 1:
        single_image_flag = False
        for bbox, seg, score in zip(bboxes, segments, scores):
            yolo_result.append([bbox, seg, score])
        max1 = max(yolo_result, key=lambda x: x[2])
        yolo_result.remove(max1)
        max2 = max(yolo_result, key=lambda x: x[2])

        segments_intersection_flag = find_mask_intersection(max1[1], max2[1])
        if segments_intersection_flag:
            yolo_output = [max1, max2]
        else:
            single_image_flag = True
            yolo_output = [max([max1, max2], key=lambda x: x[2])]

    print(f'SHAPE: {image.shape}')
    for bbox, seg, score in yolo_output:
        if score > 0.6 and (single_image_flag or segments_intersection_flag):
            print(f'Score: {score}')
            mask = np.zeros(image.shape[:2], np.uint8)
            cv2.drawContours(mask, [seg], -1, (255, 255, 255), -1, cv2.LINE_AA)
            plt.imshow(mask)
            plt.show()

            radius = int(min(image.shape[0], image.shape[1]) * 0.03)
            if radius < 2:
                radius = 3
            epsilon = radius // 2
            print(radius)
            fil_mask = cv2.ximgproc.guidedFilter(image, mask, radius, epsilon)
            _, bw = cv2.threshold(fil_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # bw = 255 - bw if bw[0, 0] else bw

            masks.append(bw)
            plt.imshow(bw)
            plt.show()
    return masks


def draw_polynomial(image, thres_coef=0.01):
    masks = get_paper_masks_yolo(image)
    results = []

    for mask in masks:
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        big_contour = max(contours, key=cv2.contourArea)
        result = np.zeros_like(image)
        cv2.drawContours(result, [big_contour], -1, (0, 255, 0), cv2.FILLED)
        epsilon = thres_coef * cv2.arcLength(big_contour, True)
        approx = cv2.approxPolyDP(big_contour, epsilon, True)
        hull = cv2.convexHull(approx)
        results.append((approx, hull, big_contour))
    return results


def sort_points(points):
    figure_center = np.mean(points, axis=0)
    points_list = list(points)
    points_list.sort(key=lambda point: np.arctan2(point[0] - figure_center[0], point[1] - figure_center[1]))
    return np.array(points_list)


def line_point_cross_product(line, point) -> ndarray:
    v1 = np.subtract(line[1], line[0])
    v2 = np.subtract(line[1], point)
    xp = np.cross(v1, v2)
    return xp


def find_edges(img_test, contour, convex):
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

    unique_top = list(set(map(tuple, top)))
    unique_top.sort()

    right = right[::-1]
    return left, right, unique_top, bottom


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


def interpolate(x, y, n_points):
    phi = np.linspace(0, 1, n_points)
    tck, u = splprep([x, y], s=0)
    print("True")
    new_points = splev(phi, tck)
    x1, y1 = new_points[0], new_points[1]
    return x1, y1

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


def opposite_interpolate(side_1, side_2, smooth=False):
    side_1_x, side_1_y = zip(*side_1)
    side_2_x, side_2_y = zip(*side_2)
    pts_num_lr = np.max((len(side_1_x), len(side_2_x)))

    new_side_1_x, new_side_1_y = interpolate(side_1_x, side_1_y, pts_num_lr)
    new_side_2_x, new_side_2_y = interpolate(side_2_x, side_2_y, pts_num_lr)

    if smooth:
        window = 51
        coef = 5
        new_side_1_x = savgol_filter(new_side_1_x, window, coef)
        new_side_1_y = savgol_filter(new_side_1_y, window, coef)
        new_side_2_x = savgol_filter(new_side_2_x, window, coef)
        new_side_2_y = savgol_filter(new_side_2_y, window, coef)
    return new_side_1_x, new_side_1_y, new_side_2_x, new_side_2_y


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


def flatten(t):
    return list(itertools.chain.from_iterable(t))


def find_intersection(curve_1, curve_2, tolerance):
    tree_big = spatial.cKDTree(curve_1)
    tree_small = spatial.cKDTree(curve_2)
    inds = tree_small.query_ball_tree(tree_big, r=tolerance)
    small_inds = flatten([i for i in inds if i])
    unique_inds = np.unique(small_inds).tolist()
    unique_coords = [curve_1[i] for i in unique_inds]
    return np.mean(np.asarray(unique_coords), axis=0)


class YOLOSegmentation:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, img):
        height, width, channels = img.shape

        results = self.model.predict(source=img.copy(), save=False, save_txt=False)
        result = results[0]
        segmentation_contours_idx = []

        for seg in result.masks.xyn:
            seg[:, 0] *= width
            seg[:, 1] *= height
            segment = np.array(seg, dtype=np.int32)
            segmentation_contours_idx.append(segment)

        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
        scores = np.array(result.boxes.conf.cpu(), dtype="float").round(3)

        return bboxes, class_ids, segmentation_contours_idx, scores

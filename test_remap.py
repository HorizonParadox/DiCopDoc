import SimpleITK as sitk
import cv2
import nib as nib
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from mayavi import mlab
from scipy.interpolate import griddata, interpn
from scipy.ndimage import map_coordinates
import nibabel
from doc_rec_cls import output_resize
import doc_rec_cls as DC
import fastremap


def resample_image(img, height, width, points_set, xy_pairs, depth):
    comp_w = complex(f'{width}j')
    comp_h = complex(f'{height}j')
    grid_x, grid_y = np.mgrid[0:(height - 1):comp_h, 0:(width - 1):comp_w]
    grid_z = griddata(np.fliplr(xy_pairs), np.fliplr(points_set), (grid_x, grid_y), method='cubic')

    map_x = np.append([], [ar[:, 1] for ar in grid_z]).reshape(height, width)
    map_y = np.append([], [ar[:, 0] for ar in grid_z]).reshape(height, width)

    map_z = np.full((height, width), np.nan)
    mask = ~np.isnan(map_x)
    map_z[mask] = depth[mask]

    map_x_32 = map_x.astype('float64')
    map_y_32 = map_y.astype('float64')
    map_z_32 = map_z.astype('float32')

    image_sitk = sitk.GetImageFromArray(img)

    map_x_sitk = sitk.GetImageFromArray(map_x)
    map_y_sitk = sitk.GetImageFromArray(map_y)

    displacement_transform = sitk.DisplacementFieldTransform(sitk.GetImageFromArray([map_x_sitk, map_y_sitk]))

    resampled_image_sitk = sitk.Resample(image_sitk, (height, width), displacement_transform)

    resampled_image_cv = sitk.GetArrayFromImage(resampled_image_sitk)
    cv2.imshow('Resampled Image', resampled_image_cv)
    cv2.waitKey(0)

    warped_image = cv2.remap(img, map_x_32, map_y_32, cv2.INTER_CUBIC)
    warped_image = np.fliplr(np.rot90(warped_image, 2))

    return warped_image


NUM_OF_LINE = 9
POLYNOMIAL_DEGREE = 4

initial_img = cv2.imread('oleg_images/free_a4.jpg')

height_image, width_image, _ = initial_img.shape
num_points = int(width_image * 0.1)
depth_image = cv2.imread('2.pfm', cv2.IMREAD_UNCHANGED)
polynomial = DC.draw_polynomial(initial_img.copy(), thres_coef=0.03)
for concave, conv, cont in polynomial:
    left, right, top, bottom = DC.find_edges(initial_img.copy(), cont, conv)
    shift = left[0]
    length, width = DC.get_dimensions(left, right, top, bottom)
    new_left_x, new_left_y, new_right_x, new_right_y = DC.opposite_interpolate(left, right, True)
    new_bottom_x, new_bottom_y, new_top_x, new_top_y = DC.opposite_interpolate(bottom, top, True)
    linfit_x_bt, linfit_y_bt, linfit_x_lr, linfit_y_lr = DC.get_lines_of_approximation(
        initial_img.copy(), NUM_OF_LINE, new_left_x, new_left_y, new_right_x, new_right_y, new_bottom_x,
        new_bottom_y, new_top_x, new_top_y)
    polys_bt = DC.interpolation_approximation(
        initial_img.copy(), width_image, NUM_OF_LINE, POLYNOMIAL_DEGREE, num_points, linfit_x_bt, linfit_y_bt)
    polys_lr = DC.interpolation_approximation(
        initial_img.copy(), height_image, NUM_OF_LINE, POLYNOMIAL_DEGREE, num_points, linfit_x_lr, linfit_y_lr)
    points_set = np.asarray(DC.get_intersection_interpolation_approximation(length, width, polys_bt, polys_lr))
    XYpairs = DC.get_fix_grid_coordinates(length, width, NUM_OF_LINE, shift)
    warped_image = DC.remap_image(initial_img, height_image, width_image, points_set, XYpairs, depth_image)
    cv2.imwrite('2.jpg', warped_image)


def test():
    image = sitk.ReadImage("2.jpg")
    # Применить ResampleImageFilter
    resampler = sitk.ResampleImageFilter()
    if image.GetDimension() == 2:
        resampler.SetSize([512, 512])
    else:
        resampler.SetSize([512, 512, image.GetSize()[2]])

    resampled_image = resampler.Execute(image)

    # Сохранить измененное изображение
    sitk.WriteImage(resampled_image, "1.jpg")

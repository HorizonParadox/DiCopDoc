import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_root_mse as nrmse


def mse(image1, image2):
    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    err /= float(image1.shape[0] * image2.shape[1])
    return err


directory_3_1 = '../../images/OCR/test_images/3_1/not_scan/'
directory_10_1 = '../../images/OCR/test_images/10_1/'
directory_13_2 = '../../images/OCR/test_images/13_2/'

my_image = cv.imread(directory_10_1 + '10_1.jpg')
scan_image = cv.imread(directory_10_1 + '10_scan.png')
orig_image = cv.imread(directory_10_1 + '10_1_orig.jpg')
any_scan_image = cv.imread(directory_10_1 + '10_1_any_scanner.jpg')
cam_scanner_image = cv.imread(directory_10_1 + '10_1_cam_scanner.jpg')
tap_scanner_image = cv.imread(directory_10_1 + '10_1_tap_scanner.jpg')

height_image, width_image, _ = scan_image.shape
scan_gray = cv.cvtColor(scan_image, cv.COLOR_BGR2GRAY)

my_image = cv.resize(my_image, (width_image, height_image))
orig_image = cv.resize(orig_image, (width_image, height_image))
any_scan_image = cv.resize(any_scan_image, (width_image, height_image))
cam_scanner_image = cv.resize(cam_scanner_image, (width_image, height_image))
tap_scanner_image = cv.resize(tap_scanner_image, (width_image, height_image))

print(f'SSIM for my image: {ssim(scan_gray, cv.cvtColor(my_image, cv.COLOR_BGR2GRAY))}')
print(f'SSIM for orig image: {ssim(scan_gray, cv.cvtColor(orig_image, cv.COLOR_BGR2GRAY))}')
print(f'SSIM for Any scan: {ssim(scan_gray, cv.cvtColor(any_scan_image, cv.COLOR_BGR2GRAY))}')
print(f'SSIM for Cam scanner: {ssim(scan_gray, cv.cvtColor(cam_scanner_image, cv.COLOR_BGR2GRAY))}')
print(f'SSIM for Tap scanner: {ssim(scan_gray, cv.cvtColor(tap_scanner_image, cv.COLOR_BGR2GRAY))}')

print('\n')

print(f'MSE for my image: {mse(scan_image, my_image)}')
print(f'MSE for orig image: {mse(scan_image, orig_image)}')
print(f'MSE for Any scan: {mse(scan_image, any_scan_image)}')
print(f'MSE for Cam scanner: {mse(scan_image, cam_scanner_image)}')
print(f'MSE for Tap scanner: {mse(scan_image, tap_scanner_image)}')

print('\n')

print(f'NRMSE for my image: {nrmse(scan_image, my_image)}')
print(f'NRMSE for orig image: {nrmse(scan_image, orig_image)}')
print(f'NRMSE for Any scan: {nrmse(scan_image, any_scan_image)}')
print(f'NRMSE for Cam scanner: {nrmse(scan_image, cam_scanner_image)}')
print(f'NRMSE for Tap scanner: {nrmse(scan_image, tap_scanner_image)}')


import cv2
import imutils

my_image = cv2.imread('OCR/test_images/3_1/3_1.jpg')
scan_image = cv2.imread('OCR/test_images/3_1/3_1_scan.png')
orig_image = cv2.imread('OCR/test_images/3_1/3_1_orig.jpg')
any_scan_image = cv2.imread('OCR/test_images/3_1/3_1_any_scanner.jpg')
cam_scanner_image = cv2.imread('OCR/test_images/3_1/3_1_cam_scanner.jpg')
tap_scanner_image = cv2.imread('OCR/test_images/3_1/3_1_tap_scanner.jpg')

my_image = imutils.resize(my_image, height=1500)
scan_image = imutils.resize(scan_image, height=1500)
orig_image = imutils.resize(orig_image, height=1500)
any_scan_image = imutils.resize(any_scan_image, height=1500)
cam_scanner_image = imutils.resize(cam_scanner_image, height=1500)
tap_scanner_image = imutils.resize(tap_scanner_image, height=1500)

cv2.imwrite('OCR/test_images/3_1/3_1.jpg', my_image)
cv2.imwrite('OCR/test_images/3_1/3_1_scan.png', scan_image)
cv2.imwrite('OCR/test_images/3_1/3_1_orig.jpg', orig_image)
cv2.imwrite('OCR/test_images/3_1/3_1_any_scanner.jpg', any_scan_image)
cv2.imwrite('OCR/test_images/3_1/3_1_cam_scanner.jpg', cam_scanner_image)
cv2.imwrite('OCR/test_images/3_1/3_1_tap_scanner.jpg', tap_scanner_image)
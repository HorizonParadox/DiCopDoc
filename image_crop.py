import imutils
import cv2 as cv


def image_crop(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_OTSU + cv.THRESH_BINARY)[1]

    contour = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour = contour[0] if len(contour) == 2 else contour[1]
    contour = sorted(contour, key=cv.contourArea, reverse=True)

    if len(contour) > 0:
        x, y, w, h = cv.boundingRect(contour[0])
        img = img[y:y + h, x:x + w]
    return img
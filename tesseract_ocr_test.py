import pytesseract
from PIL import Image


img = Image.open('OCR/test_images/3_1/3_1_bw.jpg')
img2 = Image.open('OCR/test_images/3_1/3_1_scan.png')
pytesseract.pytesseract.tesseract_cmd = r'D:\Programs\tesseract\tesseract.exe'
config = r'--oem 3 --psm 6'

text = pytesseract.image_to_string(img, config=config)
text2 = pytesseract.image_to_string(img2, config=config)

data = pytesseract.image_to_data(img, lang='eng', output_type=pytesseract.Output.DICT)
confidences = data['conf']
accuracy = sum(confidences) / len(confidences)

data2 = pytesseract.image_to_data(img2, lang='eng', output_type=pytesseract.Output.DICT)
confidences2 = data2['conf']
accuracy2 = sum(confidences2) / len(confidences2)

print(f'SCORE: {accuracy}')
print(f'SCORE: {accuracy2}')

print(f'SCORE: {text}')
print(f'SCORE: {text2}')
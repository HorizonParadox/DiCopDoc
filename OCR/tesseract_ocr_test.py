import statistics
import pytesseract
from PIL import Image
import src.OCR.text_metrics as mtr

directory_3_1 = '../../images/OCR/test_images/3_1/not_scan/'
directory_10_1 = '../../images/OCR/test_images/10_1/'
directory_1_2 = '../../images/OCR/test_images/1_2/'

my_image = Image.open(directory_10_1 + '10_1.jpg')
scan_image = Image.open(directory_10_1 + '10_scan.png')
orig_image = Image.open(directory_10_1 + '10_1_orig.jpg')
any_scan_image = Image.open(directory_10_1 + '10_1_any_scanner.jpg')
cam_scanner_image = Image.open(directory_10_1 + '10_1_cam_scanner.jpg')
tap_scanner_image = Image.open(directory_10_1 + '10_1_tap_scanner.jpg')

pytesseract.pytesseract.tesseract_cmd = r'D:\Programs\tesseract\tesseract.exe'
config = r'--oem 3 --psm 6'

my_image_data = pytesseract.image_to_data(my_image, lang='eng', output_type=pytesseract.Output.DICT, config=config)
scan_image_data = pytesseract.image_to_data(scan_image, lang='eng', output_type=pytesseract.Output.DICT, config=config)
orig_image_data = pytesseract.image_to_data(orig_image, lang='eng', output_type=pytesseract.Output.DICT, config=config)
any_scan_image_data = pytesseract.image_to_data(any_scan_image, lang='eng', output_type=pytesseract.Output.DICT, config=config)
cam_scanner_image_data = pytesseract.image_to_data(cam_scanner_image, lang='eng', output_type=pytesseract.Output.DICT, config=config)
tap_scanner_image_data = pytesseract.image_to_data(tap_scanner_image, lang='eng', output_type=pytesseract.Output.DICT, config=config)

rights_text = mtr.text_10_1
my_image_text = ' '.join(my_image_data['text']).replace(' ', '')
scan_image_text = ' '.join(scan_image_data['text']).replace(' ', '')
orig_image_text = ' '.join(orig_image_data['text']).replace(' ', '')
any_scan_image_text = ' '.join(any_scan_image_data['text']).replace(' ', '')
cam_scanner_image_text = ' '.join(cam_scanner_image_data['text']).replace(' ', '')
tap_scanner_image_text = ' '.join(tap_scanner_image_data['text']).replace(' ', '')

my_image_confidences = my_image_data['conf']
my_image_confidences = [x for x in my_image_confidences if x != -1]
scan_image_confidences = scan_image_data['conf']
scan_image_confidences = [x for x in scan_image_confidences if x != -1]
orig_image_confidences = orig_image_data['conf']
orig_image_confidences = [x for x in orig_image_confidences if x != -1]
any_scan_image_confidences = any_scan_image_data['conf']
any_scan_image_confidences = [x for x in any_scan_image_confidences if x != -1]
cam_scanner_image_confidences = cam_scanner_image_data['conf']
cam_scanner_image_confidences = [x for x in cam_scanner_image_confidences if x != -1]
tap_scanner_image_confidences = tap_scanner_image_data['conf']
tap_scanner_image_confidences = [x for x in tap_scanner_image_confidences if x != -1]

my_image_accuracy = statistics.median(my_image_confidences)
scan_image_accuracy = statistics.median(scan_image_confidences)
orig_image_accuracy = statistics.median(orig_image_confidences)
any_scan_accuracy = statistics.median(any_scan_image_confidences)
cam_scanner_accuracy = statistics.median(cam_scanner_image_confidences)
tap_scanner_accuracy = statistics.median(tap_scanner_image_confidences)

print(f'my_image_accuracy: {my_image_accuracy}')
print(f'scan_image_accuracy: {scan_image_accuracy}')
print(f'orig_image_accuracy: {orig_image_accuracy}')
print(f'any_scan_accuracy: {any_scan_accuracy}')
print(f'cam_scanner_accuracy: {cam_scanner_accuracy}')
print(f'tap_scanner_accuracy: {tap_scanner_accuracy}')

print('\n')


mtr.get_metrics(my_image_text, scan_image_text, orig_image_text, any_scan_image_text, cam_scanner_image_text,
                tap_scanner_image_text, rights_text)

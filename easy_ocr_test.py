import statistics
import easyocr
from Levenshtein import distance as lev


reader = easyocr.Reader(["en"])

my_image = reader.readtext('OCR/test_images/3_1/3_1.jpg')
scan_image = reader.readtext('OCR/test_images/3_1/3_1_scan.png')
orig_image = reader.readtext('OCR/test_images/3_1/3_1_orig.jpg')
any_scan_image = reader.readtext('OCR/test_images/3_1/3_1_any_scanner.jpg')
cam_scanner_image = reader.readtext('OCR/test_images/3_1/3_1_cam_scanner.jpg')
tap_scanner_image = reader.readtext('OCR/test_images/3_1/3_1_tap_scanner.jpg')

scan_scores = []
scan_words = []

my_scores = []
my_words = []

orig_scores = []
orig_words = []

any_scan_scores = []
any_scan_words = []

cam_scanner_scores = []
cam_scanner_words = []

tap_scanner_scores = []
tap_scanner_words = []


rights_words_str = "SOUTHERN GATWICK AIRPORT SOUTH TERMINAL 51 CUSTOMER COPY Keep this copy for your records Sales see " \
               "receipt €40.00 Customer Present ICC CHASE VISA PAN: **** **** **** 4939 PAN Seq No: 01 AID: " \
               "A0000000031010 Merchant No:***48571 TID: ****0532 Auth code: 03510I Date: 04/09/2017 Time: 18:24:49 " \
               "Ref No: 5730F-66287-51LU2-66287 Please debit my account by amount €40.00 NO CARDHOLDER VERIFICATION " \
               "Please retain for your records 04/09/2017 18:25:02 55433 S26017 51 5730"

for data in my_image:
    my_scores.append(round(data[2], 2))
    my_words.append(data[1])

for data in scan_image:
    scan_scores.append(round(data[2], 2))
    scan_words.append(data[1])

for data in orig_image:
    orig_scores.append(round(data[2], 2))
    orig_words.append(data[1])

for data in any_scan_image:
    any_scan_scores.append(round(data[2], 2))
    any_scan_words.append(data[1])

for data in cam_scanner_image:
    cam_scanner_scores.append(round(data[2], 2))
    cam_scanner_words.append(data[1])

for data in tap_scanner_image:
    tap_scanner_scores.append(round(data[2], 2))
    tap_scanner_words.append(data[1])

my_median = statistics.median(my_scores)
scan_median = statistics.median(scan_scores)
orig_median = statistics.median(orig_scores)
any_scan_median = statistics.median(any_scan_scores)
cam_scanner_median = statistics.median(cam_scanner_scores)
tap_scanner_median = statistics.median(tap_scanner_scores)

my_words_str = ' '.join(my_words)
scan_words_str = ' '.join(scan_words)
orig_words_str = ' '.join(orig_words)
any_scan__words_str = ' '.join(any_scan_words)
cam_scanner_words_str = ' '.join(cam_scanner_words)
tap_scanner_words_str = ' '.join(tap_scanner_words)

my_words = my_words_str.split()
scan_words = scan_words_str.split()
orig_words = orig_words_str.split()
any_scan_words = any_scan__words_str.split()
cam_scanner_words = cam_scanner_words_str.split()
tap_scanner_words = tap_scanner_words_str.split()
rights_words = rights_words_str.split()

print(f'My image: {my_words}')
print(f'Scan image: {scan_words}')
print(f'Orig image: {orig_words}')
print(f'Any scan image: {any_scan_words}')
print(f'Cam scanner image: {cam_scanner_words}')
print(f'Tap scanner image: {tap_scanner_words}')
print(f'Original text: {rights_words}')

print('\n')
print(f'Scan image median accuracy: {scan_median}')
print(f'My image median accuracy: {my_median}')
print(f'Orig image median accuracy: {orig_median}')
print(f'Any scan image median accuracy: {any_scan_median}')
print(f'Cam scanner image median accuracy: {cam_scanner_median}')
print(f'Tap scanner image median accuracy: {tap_scanner_median}')

print('\n')
lev_my_words = []
lev_scan_words = []
lev_orig_words = []
lev_any_scan_words = []
lev_cam_scanner_words = []
lev_tap_scanner_words = []
for i in range(0, len(rights_words)):
    lev_my_words.append(lev(rights_words[i], my_words[i]))
    lev_scan_words.append(lev(rights_words[i], scan_words[i]))
    lev_orig_words.append(lev(rights_words[i], orig_words[i]))
    lev_any_scan_words.append(lev(rights_words[i], any_scan_words[i]))
    lev_cam_scanner_words.append(lev(rights_words[i], cam_scanner_words[i]))
    lev_tap_scanner_words.append(lev(rights_words[i], tap_scanner_words[i]))

print(f'Median Levenshtein for scan image: {statistics.median(lev_scan_words)}')
print(f'Median Levenshtein for my image: {statistics.median(lev_my_words)}')
print(f'Levenshtein for orig image: {statistics.median(lev_orig_words)}')
print(f'Median Levenshtein for Any scan: {statistics.median(lev_any_scan_words)}')
print(f'Median Levenshtein for Cam scanner: {statistics.median(lev_cam_scanner_words)}')
print(f'Median Levenshtein for Tap scanner: {statistics.median(lev_tap_scanner_words)}')

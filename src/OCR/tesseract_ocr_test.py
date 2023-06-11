import statistics
from Levenshtein import distance as lev
import pytesseract
from PIL import Image

directory_3_1 = 'OCR/test_images/3_1/not_scan/'
directory_10_1 = 'OCR/test_images/10_1/'


rights_text_3_1 = "SOUTHERN GATWICK AIRPORT SOUTH TERMINAL 51 CUSTOMER COPY Keep this copy for your records Sales - see " \
              "receipt £40.00 Customer Present ICC CHASE VISA PAN: **** **** **** 4939 PAN Seq No: 01 AID: " \
              "A0000000031010 Merchant No:***48571 TID: ****0532 Auth code: 03510I Date: 04/09/2017 Time: 18:24:49 " \
              "Ref No: 5730F-66287-51LU2-66287 Please debit my account by amount £40.00 NO CARDHOLDER VERIFICATION " \
              "Please retain for your records 04/09/2017 18:25:02 55433 S26017 51 5730".replace(' ', '')

rights_text_10_1 = "PLATINUM® CARD CONCIERGE” Let us take care of it. Whether you need a dining recommendation for " \
                   "an upcoming trip or tickets for a Broadway show, Concierge is your trusted resource, available " \
                   "24/7. Around the world, personalized service is one tap away. Seamlessly access Platinum " \
                   "Concierge, Travel or your Customer Care Professionals through enhancements Amex® Mobile app. " \
                   "GLOBAL DINING COLLECTION” Concierge is about more than getting you a reservation. With the Global" \
                   " Dining Collection, you can now enjoy special access to culinary events and experiences " \
                   "customized for Platinum Card® Members like you. BY INVITATION ONLY®” RSVP to events tailored " \
                   "for you. Your Platinum Card® is your key to specially curated By Invitation Only® experiences. We" \
                   " customize once-in-a-lifetime events - from sporting and fashion to fine dining, art and " \
                   "performances. PLATINUM TRAVEL SERVICE” Enjoy extraordinary travel experiences with Platinum " \
                   "Travel Services. Our dedicated Travel Counselors help dream up ways to make your trips more " \
                   "memorable. CARD MEMBER SERVICES Whenever you need us, we're here. Our Member Services team will " \
                   "ensure you are taken care of. From lost Card replacement to statement questions, we are available" \
                   " to help 24/7. Please see the Additional Disclosures insert for more details.".replace(' ', '')

my_image = Image.open(directory_10_1 + '10_1.jpg')
scan_image = Image.open(directory_10_1 + '10_scan.png')
orig_image = Image.open(directory_10_1 + '10_1_orig.jpg')
any_scan_image = Image.open(directory_10_1 + '10_1_any_scanner.jpg')
cam_scanner_image = Image.open(directory_10_1 + '10_1_cam_scanner.jpg')
tap_scanner_image = Image.open(directory_10_1 + '10_1_tap_scanner.jpg')

pytesseract.pytesseract.tesseract_cmd = r'D:\Programs\tesseract\tesseract.exe'
config = r'--oem 3 --psm 6'

# 6, 11, 12
# my_image_text = pytesseract.image_to_string(my_image, config=config)

my_image_data = pytesseract.image_to_data(my_image, lang='eng', output_type=pytesseract.Output.DICT, config=config)
scan_image_data = pytesseract.image_to_data(scan_image, lang='eng', output_type=pytesseract.Output.DICT, config=config)
orig_image_data = pytesseract.image_to_data(orig_image, lang='eng', output_type=pytesseract.Output.DICT, config=config)
any_scan_image_data = pytesseract.image_to_data(any_scan_image, lang='eng', output_type=pytesseract.Output.DICT, config=config)
cam_scanner_image_data = pytesseract.image_to_data(cam_scanner_image, lang='eng', output_type=pytesseract.Output.DICT, config=config)
tap_scanner_image_data = pytesseract.image_to_data(tap_scanner_image, lang='eng', output_type=pytesseract.Output.DICT, config=config)

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

print(f'right_text: \n{rights_text_10_1}')
print(f'my_image_text: \n{my_image_text}')
print(f'scan_image_text: \n{scan_image_text}')
print(f'orig_image_text: \n{orig_image_text}')
print(f'any_scan_text: \n{any_scan_image_text}')
print(f'cam_scanner_text: \n{cam_scanner_image_text}')
print(f'tap_scanner_text: \n{tap_scanner_image_text}')

print('\n')

print(f'my_image_accuracy: {my_image_accuracy}')
print(f'scan_image_accuracy: {scan_image_accuracy}')
print(f'orig_image_accuracy: {orig_image_accuracy}')
print(f'any_scan_accuracy: {any_scan_accuracy}')
print(f'cam_scanner_accuracy: {cam_scanner_accuracy}')
print(f'tap_scanner_accuracy: {tap_scanner_accuracy}')

print('\n')

print(f'Levenshtein for my image: {lev(rights_text_10_1, my_image_text)}')
print(f'Levenshtein for scan image: {lev(rights_text_10_1, scan_image_text)}')
print(f'Levenshtein for orig image: {lev(rights_text_10_1, orig_image_text)}')
print(f'Levenshtein for Any scan: {lev(rights_text_10_1, any_scan_image_text)}')
print(f'Levenshtein for Cam scanner: {lev(rights_text_10_1, cam_scanner_image_text)}')
print(f'Levenshtein for Tap scanner: {lev(rights_text_10_1, tap_scanner_image_text)}')

print('\n')

print(f'Number of find "words" for right text: {len(rights_text_10_1)}')
print(f'Number of find "words" for scan image: {len(scan_image_text)}')
print(f'Number of find "words" for my image: {len(my_image_text)}')
print(f'Number of find "words" for orig image: {len(orig_image_text)}')
print(f'Number of find "words" for Any scan: {len(any_scan_image_text)}')
print(f'Number of find "words" for Cam scanner: {len(cam_scanner_image_text)}')
print(f'Number of find "words" for Tap scanner: {len(tap_scanner_image_text)}')

with open(f'../1.txt', 'w', encoding="utf-8") as text_file:
    text_file.write(f'{rights_text_10_1}\n{scan_image_text}\n{my_image_text}\n{orig_image_text}\n{any_scan_image_text}\n{cam_scanner_image_text}\n{tap_scanner_image_text}')
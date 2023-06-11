import statistics
import easyocr
from Levenshtein import distance as lev


def get_words_and_scores(image):
    image_score = []
    image_word = []
    for data in image:
        image_score.append(round(data[2], 2))
        image_word.append(data[1])
    return image_word, image_score


rights_text_3_1 = "SOUTHERN GATWICK AIRPORT SOUTH TERMINAL 51 CUSTOMER COPY Keep this copy for your records Sales see " \
               "receipt €40.00 Customer Present ICC CHASE VISA PAN: **** **** **** 4939 PAN Seq No: 01 AID: " \
               "A0000000031010 Merchant No:***48571 TID: ****0532 Auth code: 03510I Date: 04/09/2017 Time: 18:24:49 " \
               "Ref No: 5730F-66287-51LU2-66287 Please debit my account by amount €40.00 NO CARDHOLDER VERIFICATION " \
               "Please retain for your records 04/09/2017 18:25:02 55433 S26017 51 5730"

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
                   " to help 24/7. Please see the Additional Disclosures insert for more details."

directory_10_1 = 'OCR/test_images/10_1/'
reader = easyocr.Reader(["en"])

my_image = reader.readtext(directory_10_1 + '10_1.jpg')
scan_image = reader.readtext(directory_10_1 + '10_scan.png')
orig_image = reader.readtext(directory_10_1 + '10_1_orig.jpg')
any_scan_image = reader.readtext(directory_10_1 + '10_1_any_scanner.jpg')
cam_scanner_image = reader.readtext(directory_10_1 + '10_1_cam_scanner.jpg')
tap_scanner_image = reader.readtext(directory_10_1 + '10_1_tap_scanner.jpg')


scan_words, scan_scores = get_words_and_scores(scan_image)
my_words,   my_scores = get_words_and_scores(my_image)
orig_words, orig_scores = get_words_and_scores(orig_image)
any_scan_words, any_scan_scores = get_words_and_scores(any_scan_image)
cam_scanner_words, cam_scanner_scores = get_words_and_scores(cam_scanner_image)
tap_scanner_words, tap_scanner_scores = get_words_and_scores(tap_scanner_image)


my_median = statistics.median(my_scores)
scan_median = statistics.median(scan_scores)
orig_median = statistics.median(orig_scores)
any_scan_median = statistics.median(any_scan_scores)
cam_scanner_median = statistics.median(cam_scanner_scores)
tap_scanner_median = statistics.median(tap_scanner_scores)

my_words = ' '.join(my_words).split()
scan_words = ' '.join(scan_words).split()
orig_words = ' '.join(orig_words).split()
any_scan_words = ' '.join(any_scan_words).split()
cam_scanner_words = ' '.join(cam_scanner_words).split()
tap_scanner_words = ' '.join(tap_scanner_words).split()
rights_words = rights_text_10_1.split()

print(f'My image: {my_words}')
print(f'Scan image: {scan_words}')
print(f'Orig image: {orig_words}')
print(f'Any scan image: {any_scan_words}')
print(f'Cam scanner image: {cam_scanner_words}')
print(f'Tap scanner image: {tap_scanner_words}')
print(f'Original text: {rights_words}')

print('\n')
print(f'Scan image median accuracy (min, median, max): {min(scan_scores), scan_median, max(scan_scores)}')
print(f'My image median accuracy (min, median, max): {min(my_scores), my_median, max(my_scores)}')
print(f'Orig image median accuracy (min, median, max): {min(orig_scores), orig_median, max(orig_scores)}')
print(f'Any scan image median accuracy (min, median, max): {min(any_scan_scores), any_scan_median, max(any_scan_scores)}')
print(f'Cam scanner image median accuracy (min, median, max): {min(cam_scanner_scores), cam_scanner_median, max(cam_scanner_scores)}')
print(f'Tap scanner image median accuracy (min, median, max): {min(tap_scanner_scores), tap_scanner_median, max(tap_scanner_scores)}')

print('\n')
lev_my_words = []
lev_scan_words = []
lev_orig_words = []
lev_any_scan_words = []
lev_cam_scanner_words = []
lev_tap_scanner_words = []
for i in range(0, len(rights_words)):
    try:
        lev_my_words.append(lev(rights_words[i], my_words[i]))
        lev_scan_words.append(lev(rights_words[i], scan_words[i]))
        lev_orig_words.append(lev(rights_words[i], orig_words[i]))
        lev_any_scan_words.append(lev(rights_words[i], any_scan_words[i]))
        lev_cam_scanner_words.append(lev(rights_words[i], cam_scanner_words[i]))
        lev_tap_scanner_words.append(lev(rights_words[i], tap_scanner_words[i]))
    except IndexError:
        continue

print(f'Levenshtein for scan image (min, median, max): {min(lev_scan_words), statistics.median(lev_scan_words), max(lev_scan_words)}')
print(f'Levenshtein for my image (min, median, max): {min(lev_my_words), statistics.median(lev_my_words), max(lev_my_words)}')
print(f'Levenshtein for orig image (min, median, max): {min(lev_orig_words), statistics.median(lev_orig_words), max(lev_orig_words)}')
print(f'Levenshtein for Any scan (min, median, max): {min(lev_any_scan_words), statistics.median(lev_any_scan_words), max(lev_any_scan_words)}')
print(f'Levenshtein for Cam scanner (min, median, max): {min(lev_cam_scanner_words), statistics.median(lev_cam_scanner_words), max(lev_cam_scanner_words)}')
print(f'Levenshtein for Tap scanner (min, median, max): {min(lev_tap_scanner_words), statistics.median(lev_tap_scanner_words), max(lev_tap_scanner_words)}')

print('\n')
print(f'Number of find symbols for right text: {len(rights_words)}')
print(f'Number of find symbols for scan image: {len(scan_words)}')
print(f'Number of find symbols for my image: {len(my_words)}')
print(f'Number of find symbols for orig image: {len(orig_words)}')
print(f'Number of find symbols for Any scan: {len(any_scan_words)}')
print(f'Number of find symbols for Cam scanner: {len(cam_scanner_words)}')
print(f'Number of find symbols for Tap scanner: {len(tap_scanner_words)}')
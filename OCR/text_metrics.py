import jaro
from Levenshtein import distance as lev
from jiwer import cer

text_3_1 = "SOUTHERN GATWICK AIRPORT SOUTH TERMINAL 51 CUSTOMER COPY Keep this copy for your records Sales - see " \
              "receipt £40.00 Customer Present ICC CHASE VISA PAN: **** **** **** 4939 PAN Seq No: 01 AID: " \
              "A0000000031010 Merchant No:***48571 TID: ****0532 Auth code: 03510I Date: 04/09/2017 Time: 18:24:49 " \
              "Ref No: 5730F-66287-51LU2-66287 Please debit my account by amount £40.00 NO CARDHOLDER VERIFICATION " \
              "Please retain for your records 04/09/2017 18:25:02 55433 S26017 51 5730".replace(' ', '')

text_10_1 = "PLATINUM® CARD CONCIERGE” Let us take care of it. Whether you need a dining recommendation for " \
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

text_1_2 = "10 ENGLISH Personalizing your brushing experience Philips Sonicare automatically starts in the default " \
           "Clean mode. To personalize your brushing: 1 Prior to turning on the Philips Sonicare, press the " \
           "personalized brushing button to cycle through the modes and routines. The green LED indicates the " \
           "selected mode or routine. Brushing modes Clean mode Standard mode for superior teeth cleaning. White " \
           "mode 2 minutes of Clean mode, with an additional 30 seconds of White mode to focus on your " \
           "visible front teeth. White mode brushing instructions 1 Brush the first 2 minutes as explained " \
           "in section 'Brushing instructions'. 2 After the 2 minutes of Clean mode, the White mode starts with a " \
           "change in brushing sound and motion. This is your signal to start brushing the upper front teeth for " \
           "15 seconds. 3 At the next beep and pause, move to the bottom front teeth for the final 15 seconds of " \
           "brushing. Massage mode Mode for gentle gum stimulation.".replace(' ', '')


def get_metrics(my_image_text, scan_image_text, orig_image_text, any_scan_image_text, cam_scanner_image_text,
                tap_scanner_image_text, rights_text_10_1):

    print(f'My image: {my_image_text}')
    print(f'Scan image: {scan_image_text}')
    print(f'Orig image: {orig_image_text}')
    print(f'Any scan image: {any_scan_image_text}')
    print(f'Cam scanner image: {cam_scanner_image_text}')
    print(f'Tap scanner image: {tap_scanner_image_text}')
    print(f'Original text: {rights_text_10_1}')

    print('\n')

    print(f'Levenshtein for my image: {lev(rights_text_10_1, my_image_text)}')
    print(f'Levenshtein for scan image: {lev(rights_text_10_1, scan_image_text)}')
    print(f'Levenshtein for orig image: {lev(rights_text_10_1, orig_image_text)}')
    print(f'Levenshtein for Any scan: {lev(rights_text_10_1, any_scan_image_text)}')
    print(f'Levenshtein for Cam scanner: {lev(rights_text_10_1, cam_scanner_image_text)}')
    print(f'Levenshtein for Tap scanner: {lev(rights_text_10_1, tap_scanner_image_text)}')

    print('\n')

    print(f'Number of find "symbols" for right text: {len(rights_text_10_1)}')
    print(f'Number of find "symbols" for scan image: {len(scan_image_text)}')
    print(f'Number of find "symbols" for my image: {len(my_image_text)}')
    print(f'Number of find "symbols" for orig image: {len(orig_image_text)}')
    print(f'Number of find "symbols" for Any scan: {len(any_scan_image_text)}')
    print(f'Number of find "symbols" for Cam scanner: {len(cam_scanner_image_text)}')
    print(f'Number of find "symbols" for Tap scanner: {len(tap_scanner_image_text)}')

    print('\n')

    print(f'JaroWinkler for my image: {jaro.jaro_winkler_metric(rights_text_10_1, my_image_text)}')
    print(f'JaroWinkler for scan image: {jaro.jaro_winkler_metric(rights_text_10_1, scan_image_text)}')
    print(f'JaroWinkler for orig image: {jaro.jaro_winkler_metric(rights_text_10_1, orig_image_text)}')
    print(f'JaroWinkler for Any scan: {jaro.jaro_winkler_metric(rights_text_10_1, any_scan_image_text)}')
    print(f'JaroWinkler for Cam scanner: {jaro.jaro_winkler_metric(rights_text_10_1, cam_scanner_image_text)}')
    print(f'JaroWinkler for Tap scanner: {jaro.jaro_winkler_metric(rights_text_10_1, tap_scanner_image_text)}')

    print('\n')

    print(f'CER for my image: {cer(rights_text_10_1, my_image_text)}')
    print(f'CER for scan image: {cer(rights_text_10_1, scan_image_text)}')
    print(f'CER for orig image: {cer(rights_text_10_1, orig_image_text)}')
    print(f'CER for Any scan: {cer(rights_text_10_1, any_scan_image_text)}')
    print(f'CER for Cam scanner: {cer(rights_text_10_1, cam_scanner_image_text)}')
    print(f'CER for Tap scanner: {cer(rights_text_10_1, tap_scanner_image_text)}')

    with open(f'../1.txt', 'w', encoding="utf-8") as text_file:
        text_file.write(f'{rights_text_10_1}\n{scan_image_text}\n{my_image_text}\n{orig_image_text}\n{any_scan_image_text}\n{cam_scanner_image_text}\n{tap_scanner_image_text}')

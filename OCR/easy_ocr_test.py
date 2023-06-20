import easyocr
import src.OCR.text_metrics as mtr


def get_words_and_scores(image):
    image_score = []
    image_word = []
    for data in image:
        image_score.append(round(data[2], 2))
        image_word.append(data[1])
    return image_word, image_score


directory_10_1 = '../../images/OCR/test_images/10_1/'
directory_1_2 = '../../images/OCR/test_images/1_2/'
reader = easyocr.Reader(["en"])

my_image = reader.readtext(directory_10_1 + '10_1.jpg')
scan_image = reader.readtext(directory_10_1 + '10_scan.png')
orig_image = reader.readtext(directory_10_1 + '10_1_orig.jpg')
any_scan_image = reader.readtext(directory_10_1 + '10_1_any_scanner.jpg')
cam_scanner_image = reader.readtext(directory_10_1 + '10_1_cam_scanner.jpg')
tap_scanner_image = reader.readtext(directory_10_1 + '10_1_tap_scanner.jpg')

scan_words, scan_scores = get_words_and_scores(scan_image)
my_words, my_scores = get_words_and_scores(my_image)
orig_words, orig_scores = get_words_and_scores(orig_image)
any_scan_words, any_scan_scores = get_words_and_scores(any_scan_image)
cam_scanner_words, cam_scanner_scores = get_words_and_scores(cam_scanner_image)
tap_scanner_words, tap_scanner_scores = get_words_and_scores(tap_scanner_image)

rights_text = mtr.text_10_1
my_image_text = ' '.join(my_words).replace(' ', '')
scan_image_text = ' '.join(scan_words).replace(' ', '')
orig_image_text = ' '.join(orig_words).replace(' ', '')
any_scan_image_text = ' '.join(any_scan_words).replace(' ', '')
cam_scanner_image_text = ' '.join(cam_scanner_words).replace(' ', '')
tap_scanner_image_text = ' '.join(tap_scanner_words).replace(' ', '')

mtr.get_metrics(my_image_text, scan_image_text, orig_image_text, any_scan_image_text, cam_scanner_image_text,
                tap_scanner_image_text, rights_text)

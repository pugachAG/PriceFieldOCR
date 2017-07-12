from tesserocr import PyTessBaseAPI, PSM
from img_utils import resize_to_height, to_binary
from common import crop_to_field_height, format_field, extract_chars

class FinalRecognizer:
    def __init__(self, apply_morphologyEx=False):
        self._apply_morphologyEx = apply_morphologyEx;

    def recognize(self, proc, img):
        bin_img = to_binary(resize_to_height(img, 150))
        extracted_field = crop_to_field_height(bin_img)
        cropped_img, minus, char_imgs = extract_chars(to_binary(resize_to_height(extracted_field, 100)), self._apply_morphologyEx)

        with PyTessBaseAPI(psm=PSM.SINGLE_LINE) as api:
            api.SetVariable("tessedit_char_whitelist", "0123456789")
            api.SetImageFile(proc.dump_debug_img("crop", cropped_img))
            return ('-' if minus else '') + '$' + format_field(api.GetUTF8Text())

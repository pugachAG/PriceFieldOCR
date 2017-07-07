from tesserocr import PyTessBaseAPI, PSM
from img_utils import *
from common import *

class DemoRecognizer:
    def recognize(self, proc, img):
        result = ""
        proc.add_debug_info("size: %dx%d" % img.shape[:2])

        resized_img = resize_to_height(img, 150)
        bin_img = to_binary(resized_img)
        #path_bin = proc.dump_debug_img("binary", bin_img)
        tmp_img = crop_to_field_height(bin_img)
        cropped_img, minus, char_imgs = extract_chars(to_binary(resize_to_height(tmp_img, 100)))
        text, conf = self.detect_line(proc.dump_debug_img("crop", cropped_img))
        proc.add_debug_info("conf: %s" % conf)

        for i, char_img in enumerate(char_imgs):
            char_path = proc.dump_debug_img("crop_%d" % i, char_img)

        return ('-' if minus else '') + self.fix_field_str(text)

        #proc.dump_debug_img("crop", cropped_img)

    def fix_field_str(self, price):
        res, chars_cnt = '', 0
        for ch in price[::-1]:
            if ch in '0123456789':
                if chars_cnt == 2:
                    res += '.'
                if chars_cnt > 2 and chars_cnt%3 == 2:
                    res += ','
                res += ch
                chars_cnt += 1
        return '$' + res[::-1]



    def detect_digit_sign(self, path):
        with PyTessBaseAPI(psm=PSM.SINGLE_CHAR) as api:
            api.SetVariable("tessedit_char_whitelist", "0123456789")
            api.SetImageFile(path)
            conf = api.AllWordConfidences()
            return (api.GetUTF8Text(), conf[0] if len(conf) == 1 else 0)

    def detect_line(self, path):
        with PyTessBaseAPI(psm=PSM.SINGLE_LINE) as api:
            api.SetVariable("tessedit_char_whitelist", "0123456789")
            api.SetImageFile(path)
            return (api.GetUTF8Text(), api.AllWordConfidences())


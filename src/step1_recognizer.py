from tesserocr import PyTessBaseAPI, PSM
from img_utils import *

class Step1Recognizer:
    def recognize(self, proc, img):
        with PyTessBaseAPI() as api:
            resized_img = resize_to_height(img, 150)
            bin_img = to_binary(resized_img)
            crop_img = resize_to_height(crop_to_field_height(bin_img), 50)

            white_list = ".,-$0123456789"
            api.SetVariable("tessedit_char_whitelist", white_list)
            api.SetImageFile(proc.dump_debug_img("crop_img", crop_img))
            return ''.join(filter(set(white_list).__contains__, api.GetUTF8Text()))



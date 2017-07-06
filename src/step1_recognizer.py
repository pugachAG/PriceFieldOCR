from tesserocr import PyTessBaseAPI, PSM
from img_utils import resize_to_height, to_binary
from common import crop_to_field_height, find_connected_areas

class Step1Recognizer:
    def recognize(self, proc, img):
        with PyTessBaseAPI() as api:
            field_img = extract_field(img)

            white_list = ".,-$0123456789"
            api.SetVariable("tessedit_char_whitelist", white_list)
            api.SetImageFile(proc.dump_debug_img("field_img", field_img))
            return ''.join(filter(set(white_list).__contains__, api.GetUTF8Text()))


def extract_field(img):
    resized_img = resize_to_height(img, 150)
    bin_img = to_binary(resized_img)
    return resize_to_height(crop_to_field_height(bin_img, 0.17), 50)

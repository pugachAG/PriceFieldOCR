from tesserocr import PyTessBaseAPI, PSM
from img_utils import resize_to_height, to_binary
from common import crop_to_field_height, find_connected_areas, format_field

# Rate 33/197 = 16% Lev 0.577518
class Step1Recognizer:
    def recognize(self, proc, img):
        with PyTessBaseAPI() as api:
            bin_img = to_binary(resize_to_height(img, 150))
            field_img = resize_to_height(crop_to_field_height(bin_img, 0.17), 50)

            api.SetVariable("tessedit_char_whitelist", "-$0123456789")
            api.SetImageFile(proc.dump_debug_img("field_img", field_img))
            return format_field(api.GetUTF8Text())


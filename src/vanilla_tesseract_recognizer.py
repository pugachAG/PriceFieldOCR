from tesserocr import PyTessBaseAPI, PSM
from img_utils import *
from common import format_field

# Rate 28/197 = 14% Lev 0.470739
class VanillaTesseractRecognizer:
    def recognize(self, proc, img):
        with PyTessBaseAPI() as api:
            white_list = "-$0123456789"
            api.SetVariable("tessedit_char_whitelist", white_list)
            api.SetImageFile(proc.cur_img_path())
            return format_field(api.GetUTF8Text())



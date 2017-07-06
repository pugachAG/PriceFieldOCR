from tesserocr import PyTessBaseAPI, PSM
from img_utils import *

# Rate 29/198 = 14% Lev 0.484381
class VanillaTesseractRecognizer:
    def recognize(self, proc, img):
        with PyTessBaseAPI() as api:
            white_list = ".,-$0123456789"
            api.SetVariable("tessedit_char_whitelist", white_list)
            api.SetImageFile(proc.cur_img_path())
            return ''.join(filter(set(white_list).__contains__, api.GetUTF8Text()))



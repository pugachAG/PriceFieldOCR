import logging
from tesserocr import PyTessBaseAPI, PSM

from ocr_process import OCRProcess
from demo_recognizer import DemoRecognizer

logging.basicConfig(level=logging.DEBUG, format='%(levelname)-5s %(message)s')

class VanillaTesseractRecognizer:
    def recognize(self, proc, img):
        with PyTessBaseAPI() as api:
            api.SetVariable("tessedit_char_whitelist", "-$0123456789")
            api.SetImageFile(proc.cur_img_path())
            return api.GetUTF8Text()

def main():
    #proc = OCRProcess("data/Fields.zip", name='big', max_files=10)
    proc = OCRProcess("data/Images.zip", "data/Images_expected.txt")
    #proc = OCRProcess("data/Images_OK.zip", "data/Images_OK_expected.txt")
    proc.run(DemoRecognizer())

if __name__ == "__main__":
    main()

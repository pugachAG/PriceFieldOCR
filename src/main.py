import logging
from tesserocr import PyTessBaseAPI, PSM

from ocr_process import OCRProcess
from demo_recognizer import DemoRecognizer
from vanilla_tesseract_recognizer import VanillaTesseractRecognizer

logging.basicConfig(level=logging.DEBUG, format='%(levelname)-5s %(message)s')


def main():
    proc = OCRProcess("data/Images.zip", "data/Images_expected.txt")
    proc.run(VanillaTesseractRecognizer())

if __name__ == "__main__":
    main()

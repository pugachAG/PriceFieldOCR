import logging
from tesserocr import PyTessBaseAPI, PSM

from ocr_process import OCRProcess

from demo_recognizer import DemoRecognizer
from vanilla_tesseract_recognizer import VanillaTesseractRecognizer
from step1_recognizer import Step1Recognizer

logging.basicConfig(level=logging.DEBUG, format='%(levelname)-5s %(message)s')


def main():
    #proc = OCRProcess("data/Images.zip", "data/Images_expected.txt", max_files=20)
    proc = OCRProcess("data/Images.zip", "data/Images_expected.txt")
    proc.run(DemoRecognizer())
    #proc.run(Step1Recognizer())

if __name__ == "__main__":
    main()

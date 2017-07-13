import logging
from tesserocr import PyTessBaseAPI, PSM

from ocr_process import OCRProcess

from demo_recognizer import DemoRecognizer
from vanilla_tesseract_recognizer import VanillaTesseractRecognizer
from step1_recognizer import Step1Recognizer
from final_recognizer import FinalRecognizer

logging.basicConfig(level=logging.DEBUG, format='%(levelname)-5s %(message)s')


def main():
    proc = OCRProcess("data/DemoImages.zip", "data/Images_expected.txt", debug=True)
    #proc = OCRProcess("data/Images.zip", "data/Images_expected.txt")
    #proc.run(DemoRecognizer())
    proc.run(VanillaTesseractRecognizer())

if __name__ == "__main__":
    main()

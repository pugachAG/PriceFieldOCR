import time
import logging
from os import listdir, path, remove
import multiprocessing as mp

from ocr_process import OCRProcess
from demo_recognizer import DemoRecognizer

logging.basicConfig(level=logging.DEBUG, format='%(levelname)-5s %(message)s')

QUEUE_PATH = "/tmp/ocrdemo"
mp.set_start_method('fork')

def main():
    while True:
        for fname in listdir(QUEUE_PATH):
            logging.info("Got %s" % fname)
            full_path = path.join(QUEUE_PATH, fname)
            proc = OCRProcess(full_path, name=fname)
            remove(full_path)
            def run_target():
                proc.run(DemoRecognizer())
            mp.Process(target=run_target).start()
        time.sleep(1)

    #proc = OCRProcess("data/Fields.zip", name='big', max_files=10)
    #proc = OCRProcess("data/Images.zip", "data/Images_expected.txt")
    #proc = OCRProcess("data/Images_OK.zip", "data/Images_OK_expected.txt")
    #proc.run(DemoRecognizer())

if __name__ == "__main__":
    main()

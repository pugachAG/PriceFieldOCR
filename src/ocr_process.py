import zipfile
import os
import shutil
import logging
import cv2
import time
from collections import defaultdict

PROCESS_FOLDER_NAME = "process"
SOURCE_FOLDER = "source"

class OCRProcess:
    def __init__(self, zip_path, expected_path=None, max_files=2000, name='default', debug=False):
        self._debug = debug
        self._path = os.path.join(PROCESS_FOLDER_NAME, name)
        self._path_src = os.path.join(self._path, SOURCE_FOLDER)
        self._reset_path()
        self._extract_images(zip_path)
        self._convert_images(max_files)
        self._parse_expected(expected_path)
        logging.info("new OCRProcess created with %d images, check localhost:8228/%s"
                % (len(self._image_names), self._path))
        self._save_report("<h3>In the queue, waiting for processing...</h3>");


    def url(self):
        return self._path

    def run(self, recognizer):
        result = dict()
        for img_i, img_name in enumerate(self._image_names):
            self._save_report("<h3>Processing image %d/%d, please refresh this page later</h3>" % (img_i+1, len(self._image_names)))
            self._cur_img_name = img_name
            img = cv2.imread(self.cur_img_path(), 0)
            start = time.time()
            result[img_name] = recognizer.recognize(self, img)
            self.add_debug_info("Elapsed:%f" % (time.time()-start))
        self._build_report(result)

    def dump_debug_img(self, label, img):
        if label not in self._labels:
            self._labels.append(label)
        path = self._build_image_path(label, self._cur_img_name)
        cv2.imwrite(path, img)
        return path

    def add_debug_info(self, info):
        self._debug_info[self._cur_img_name] += info + " <br/> "

    def cur_img_path(self):
        return self._build_image_path(SOURCE_FOLDER, self._cur_img_name)

    def _extract_images(self, zip_path):
        zip_ref = zipfile.ZipFile(zip_path, 'r')
        zip_ref.extractall(self._path_src)
        zip_ref.close()

    def _convert_images(self, max_files):
        all_files = os.listdir(self._path_src)
        self._image_names = dict()
        for img_filename in all_files[:min(len(all_files), max_files)]:
            img_name = os.path.splitext(img_filename)[0]
            png_path = self._build_image_path(SOURCE_FOLDER, img_name)
            cv2.imwrite(png_path, cv2.imread(os.path.join(self._path, SOURCE_FOLDER, img_filename)))
            self._image_names[img_name] = png_path 
        self._labels = list()
        self._debug_info = defaultdict(lambda: "")

    def _parse_expected(self, file_path):
        self._expected = dict()
        if file_path is not None:
            with open(file_path, 'r') as fl:
                for ln in fl.readlines():
                    name, price = tuple(ln.split(".tif"))
                    price = price.strip()
                    if "$" not in price:
                        logging.warn("Expected for %s is '%s', which does not contain $ sign" % (name, price))
                    self._expected[name] = price

    def _reset_path(self):
        if os.path.exists(self._path):
            shutil.rmtree(self._path)
        os.makedirs(self._path)

    def _build_image_path(self, label, img_name):
        folder = os.path.join(self._path, label)
        if not os.path.exists(folder):
            os.makedirs(folder)
        return os.path.join(folder, img_name + '.png')

    def _build_report(self, result):
        headers = ["name", "detected"]
        image_labels = [SOURCE_FOLDER]
        if self._debug:
            headers.extend(["expected", "debug_info"])
            image_labels.extend(list(self._labels))
        headers.extend(image_labels)

        report = ''
        total, success, colors = 0, 0, dict()
        for img_name in self._image_names:
            col = '#EDEDED'
            if img_name in self._expected:
                total += 1
                if self._expected[img_name] == result[img_name]:
                    success += 1
                    col = '#85FFA6'
                else:
                    col = '#FF8080'
            colors[img_name] = col

        if total > 0:
            report += "<h3>Rate %d/%d = %d%%</h3>" % (success, total, int(100*success/total))
        report += '<table border="1" style="width:100%">'
        report += "<tr>%s</tr>" % ''.join(map(lambda l: "<th>%s</th>" % l, headers))
        for img_name in sorted(self._image_names):
            rows = [
                img_name,
                result[img_name],
            ]
            if self._debug:
                rows.append(self._debug_info.get(img_name, ""))
                rows.append(self._expected[img_name] if img_name in self._expected else "NOT AVAILABLE")

            for lbl in image_labels:
                rows.append('<img src="%s">' % os.path.join(lbl, img_name + '.png'))
            report += "<tr bgcolor=\"%s\">%s</tr>" % (colors[img_name], ''.join(map(lambda l: "<td>%s</td>" % l, rows)))
        report += '</table>'
        self._save_report(report)

    def _save_report(self, report):
        page = '<html><body>' + report + '</body></html>'
        with open(os.path.join(self._path, "index.html"), 'w') as fl:
            fl.write(page)





import zipfile
import os
import shutil
import logging
import cv2
import time
from collections import defaultdict
import plotly.plotly as py
import matplotlib.pyplot as plt
import plotly.tools as tls

PROCESS_FOLDER_NAME = "process"
SOURCE_FOLDER = "input"


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def calc_d(s1, s2):
    return 1.0 - levenshtein(s1, s2)  / max(len(s1), len(s2))


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
        self._build_report(recognizer, result)

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

    def _build_report(self, recognizer, result):
        image_labels = [SOURCE_FOLDER]
        if self._debug:
            image_labels.extend(list(self._labels))
        headers = list(image_labels)
        headers.append("detected")
        if self._debug:
            headers.extend(["expected", "M2"])

        report = ''
        total, success, colors = 0, 0, dict()
        lev, all_lev = 0.0, dict()
        for img_name in self._image_names:
            col = "rgba(255, 255, 255, 0.1)"
            if img_name in self._expected:
                s_expected, s_res = self._expected[img_name], result[img_name]
                total += 1
                lev_d = calc_d(s_expected, s_res)
                lev += lev_d
                all_lev[img_name] = lev_d
                if self._expected[img_name] == result[img_name]:
                    success += 1
                col = "rgba({0}, {1}, 0, 0.1)".format(*list((min(int(2*255*x), 255) for x in [1.0-lev_d, lev_d])))
            colors[img_name] = col

        if total > 0:
            report += "<h3>Rate %d/%d = %d%%    Lev %f </h3>" % (success, total, int(100*success/total), lev/total)

        try:
            plt.hist(list(all_lev.values()))
            plt.title("M2 Histogram")
            plt.xlabel("M2 Value")
            plt.ylabel("Number of images")
            fig = plt.gcf()
            plot_url = py.plot_mpl(fig, filename=recognizer.__class__.__name__, auto_open=False)
            report += tls.get_embed(plot_url)
        except:
            logging.warn("Failed to generate plot")


        report += '<table border="1" style="width:100%">'
        report += "<tr>%s</tr>" % ''.join(map(lambda l: "<th>%s</th>" % l, headers))
        for img_name in sorted(self._image_names):
            rows = list()
            for lbl in image_labels:
                rows.append('<img src="%s">' % os.path.join(lbl, img_name + '.png'))
            rows.append(result[img_name])
            if self._debug:
                rows.append(self._expected[img_name] if img_name in self._expected else "NOT AVAILABLE")
                rows.append("{0:.2f}".format(all_lev[img_name]))

            report += "<tr style=\"background-color:%s;\">%s</tr>" % (colors[img_name], ''.join(map(lambda l: "<td>%s</td>" % l, rows)))
        report += '</table>'
        self._save_report(report)

    def _save_report(self, report):
        page = '<html><body>' + report + '</body></html>'
        with open(os.path.join(self._path, "index.html"), 'w') as fl:
            fl.write(page)





import random
import time
from flask import Flask, request, redirect, url_for

app = Flask("Field OCR demo")

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        fl = request.files['file']
        proc_name = str(int(100*time.time()))
        zip_path = "/tmp/ocrdemo/%s" % proc_name
        fl.save(zip_path)
        time.sleep(3)
        return redirect("/process/" + proc_name)
    return '''
    <!doctype html>
    <title>OCR demo</title>
    <h3>Upload zip archive with images for OCR</h3>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
      <input type=submit value=GO!>
    </form>
    '''

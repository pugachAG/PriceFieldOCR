# System
sudo apt-get install libopencv-dev python-opencv tesseract-ocr=3.04.01-4 libtesseract-dev=3.04.01-4 libleptonica-dev python3-dev 
sudo pip install virtualenv
# Python env
virtualenv -p /usr/bin/python3 ENV
source ENV/bin/activate
pip install Cython
pip install tesserocr opencv-python
pip install uwsgi Flask


# System
sudo apt-get install libopencv-dev python-opencv tesseract-ocr=3.04.01-4 libtesseract-dev=3.04.01-4 libleptonica-dev python3-dev python3-tk
sudo apt install python-pip
sudo pip install virtualenv
# Python env
virtualenv -p /usr/bin/python3 ENV
source ENV/bin/activate
pip install Cython
pip install matplotlib plotly
pip install tesserocr opencv-python
pip install uwsgi Flask


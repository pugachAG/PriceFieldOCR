#!/bin/bash
uwsgi --http :8229 --wsgi-file src/flask_app.py --callable app

#!/bin/bash

jupyter nbconvert --to html --execute /app/data_analysis.ipynb --output data_analysis.html --output-dir /app/templates/

python app.py
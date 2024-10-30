#!/bin/bash

python src/extract_scans.py $1
python src/process_scans.py $1

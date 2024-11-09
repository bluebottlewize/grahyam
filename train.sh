#! /usr/bin/bash

source myenv/bin/activate

python ./scripts/train.py "./models/$1.h5" $2
python ./scripts/convert_to_tflite.py "./models/$1.h5"

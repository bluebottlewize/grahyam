#! /usr/bin/bash

label_string="$(python ./generate_label.py)"

echo "label_enum = $label_string" > label.py

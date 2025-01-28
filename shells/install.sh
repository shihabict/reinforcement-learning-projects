#!/bin/bash
python3 -m venv rlenv
source rlenv/bin/activate

pip install wheel setuptools pip --upgrade
pip3 install wheel setuptools pip --upgrade

pip install -r requirements.txt
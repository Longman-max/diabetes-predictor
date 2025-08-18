#!/usr/bin/env bash
python -m pip install --upgrade pip setuptools wheel

python -m pip install --only-binary=:all: -r requirements.txt

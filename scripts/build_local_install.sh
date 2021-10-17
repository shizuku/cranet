#!/usr/bin/bash
rm -f ./dist/*
python3 -m build
python3 -m pip install ./dist/*.whl --force-reinstall

#!/bin/bash
for config in /configs/*
do
    conda run -n experiment python experiment.py with ${config}
done
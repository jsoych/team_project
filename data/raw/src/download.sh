#!/bin/bash
kaggle datasets download tolgadincer/labeled-chest-xray-images
unzip labeled-chest-xray-images.zip -d ${RAW_DATA_DIR}
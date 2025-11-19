#!/bin/bash

# config directory
export CONFIGS_DIR=/home/jsoychak/cs/team_project/configs/sklearn/test/json

# data directory
export DATA_DIR=/home/jsoychak/cs/team_project/data/processed

# test configs 
echo running json tests from: $CONFIGS_DIR
echo 
sudo docker run --name json-tests \
    -v $CONFIGS_DIR:/root/experiment/configs \
    -v $DATA_DIR:/mnt/data \
    experiment-sklearn:test

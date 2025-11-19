#!/bin/bash

# config directory
export CONFIGS_DIR=/home/jsoychak/cs/team_project/configs/sklearn/test/yaml

# data directory
export DATA_DIR=/home/jsoychak/cs/team_project/data/processed

# test configs 
echo running yaml tests
echo
sudo docker run --name $name \
    -v $CONFIGS_DIR:/root/experiment/configs \
    -v $DATA_DIR:/mnt/data \
    experiment-sklearn:test

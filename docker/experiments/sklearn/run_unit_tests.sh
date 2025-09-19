#!/bin/bash

# config directory
export CONFIGS_DIR=/home/jsoychak/cs/team_project/configs/sklearn/test

# data directory
export DATA_DIR=/home/jsoychak/cs/team_project/data/processed

# test configs 
for file in $(ls $CONFIGS_DIR)
do
    name=$(basename -s .yaml $file)
    echo running $name experiment
    echo
    sudo docker run --name $name \
        -v $CONFIGS_DIR/$file:/root/experiment/configs/experiment.yaml \
        -v $DATA_DIR:/mnt/data \
        experiment-sklearn:test
done

#!/bin/bash

# config directory
export CONFIGS_DIR=/home/jsoychak/cs/team_project/configs/sklearn/test

# data directory
export DATA_DIR=/home/jsoychak/cs/team_project/data/processed

# test configs with kubernetes integration
for file in $(ls $CONFIGS_DIR)
do
    name=$(basename -s .yaml $file)
    echo running $name experiment
    sudo docker run --name $name \
        -e MODELS_URL=postgresql://postgres:classifier@10.0.0.123:32638/experiments \
        -e RESULTS_URL=postgresql://postgres:classifier@10.0.0.123:32638/experiments \
        -e REGISTRY_URI=http://10.0.0.123:30885 \
        -v $CONFIGS_DIR/$file:/root/experiment/configs/experiment.yaml \
        -v $DATA_DIR:/mnt/data \
        experiment-sklearn:test
    echo
done

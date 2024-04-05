#!/bin/bash

envName="PJA-ASI-12C-GR4"
envFile="environment.yml"

if [ ! -f "$envFile" ]; then
    echo "Error: '$envFile' not found."
    exit 1
fi

conda create -n $envName -f $envFile
conda activate $envName

echo "'$envName' activated"
#!/bin/bash

datagen_model_name=$1

if [ "$datagen_model_name"x == "x" ]; then
  echo "Usage: prototype.sh DATAGEN_MODEL_NAME"
	exit -1
fi

echo "Run  DataGen Model: \"$datagen_model_name\" Python virtual environment"

venv_directory="venv_"$datagen_model_name

if ! [ -d $venv_directory ]; then
  echo "Python virtual environment does not exist: \"$venv_directory\""
	exit -1
fi

source $venv_directory/bin/activate
./prototype.py run $datagen_model_name

echo "Exit DataGen Model: \"$datagen_model_name\""

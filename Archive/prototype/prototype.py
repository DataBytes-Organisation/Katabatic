#!/usr/bin/env python3
#
# Declaratively dynamically load the configured module and instantiate class
#
# - Edit the CONFIGURATION_FILE and change parameters as needed
#
# - python -m venv venv_mock_datagen_1
# - . venv_mock_datagen_1/bin/activate
# - pip install --upgrade click pip setuptools  # plus DataGen dependencies
# - deactivate
#
# - python -m venv venv_mock_datagen_2
# - . venv_mock_datagen_1/bin/activate
# - pip install --upgrade click pip setuptools  # plus DataGen dependencies
# - deactivate
#
# - python -m venv venv_katabatic
# - . venv_katabatic/bin/activate
# - pip install --upgrade click pip setuptools  # plus prototype dependencies
# - ./prototype.py evaluate mock_datagen_1 mock_datagen_2
#
# - ./prototype.py run mock_datagen_1  # requires correct Python venv set-up
# - ./prototype.py run mock_datagen_2  # requires correct Python venv set-up
#
# Resources
# ~~~~~~~~~
# - https://www.datacamp.com/tutorial/python-subprocess
#
# To Do
# ~~~~~
# - Compare and graph all DataGen Model results

import click
import json
import os
import subprocess
import sys

from katabatic_spi import KatabaticSPI

from aiko_services.importer import load_module
# from aiko_services.process_manager import ProcessManager

NAME = "prototype"
CONFIGURATION_FILE = f"{NAME}.json"

# --------------------------------------------------------------------------- #

def run_datagen_model_process(datagen_model_name):
#   command = f"./{NAME}.sh"
#   arguments = [datagen_model_name]
#   process_manager = ProcessManager(process_exit_handler)
#   process_manager.create(0, command, arguments)

    command = [f"./{NAME}.sh", datagen_model_name]
    try:
        result = subprocess.run(command, check=True, shell=False, timeout=None)
    except (FileNotFoundError, PermissionError) as error:
        raise SystemExit(f"Couldn't run DataGen Model: {error}")
    except subprocess.CalledProcessError as called_process_error:
        error_code = called_process_error.returncode
        raise SystemExit(f"Error code {error_code}: {' '.join(command)}")

def run_datagen_model(datagen_model_name):
    with open(CONFIGURATION_FILE, "r") as file:
        configuration = json.load(file)

        if not datagen_model_name in configuration:
            raise SystemExit(
                f"Configuration '{CONFIGURATION_FILE}' doesn't have DataGen model: {datagen_model_name}")
        configuration = configuration[datagen_model_name]

    try:
        datagen_module_descriptor = configuration["datagen_module_descriptor"]
        datagen_class_name = configuration["datagen_class_name"]
    except KeyError as key_error:
        raise SystemExit(
            f"Configuration '{CONFIGURATION_FILE}' doesn't have: {key_error}")

    diagnostic = None
    try:
        datagen_module = load_module(datagen_module_descriptor)
        datagen_class = getattr(datagen_module, datagen_class_name)
    except FileNotFoundError:
        diagnostic = "couldn't be found"
    except Exception:
        diagnostic = "couldn't be loaded"
    if diagnostic:
        raise SystemExit(f"Module {datagen_module_descriptor} {diagnostic}")

    data_gen_ml = datagen_class()
    if not isinstance(data_gen_ml, KatabaticSPI):
        raise SystemExit(f"{datagen_class_name} doesn't implement KatabaticSPI")

    data_gen_ml.load_data(None)
    data_gen_ml.split_data(None, None)
    data_gen_ml.fit_model(None)

# --------------------------------------------------------------------------- #

@click.group()

def main():
    pass

@main.command(help="Evaluate DataGen Models")
@click.argument("datagen_model_names", nargs=-1)
def evaluate(datagen_model_names):
    print(f"[Katabatic evaluate {NAME} 0.2]")
    for datagen_model_name in datagen_model_names:
        print("------------------")
        run_datagen_model_process(datagen_model_name)

# TODO: Compare and graph all DataGen Model results

@main.command(help="Run DataGen Model")
@click.argument("datagen_model_name")
def run(datagen_model_name):
    print(f"[Katabatic run {NAME} 0.2]")
    print(f"    Parent process: {os.getppid()}, my process id: {os.getpid()}")
    run_datagen_model(datagen_model_name)

if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------- #

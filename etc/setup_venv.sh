#!/bin/bash

VIRTUALENV_INSTALL=$PWD/venv/

if [[ ! -d "${VIRTUALENV_INSTALL}" ]]; then
  python3 -m venv venv
fi

# set up virtual environment
source ${VIRTUALENV_INSTALL}/bin/activate


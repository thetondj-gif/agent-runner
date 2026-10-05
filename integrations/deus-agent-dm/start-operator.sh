#!/bin/bash
set -a
source ${HOME}/.agent-dm/.env
set +a
cd ${HOME}/studio-agent-dm
exec ${HOME}/agent-dm-venv/bin/python3 persistent_operator.py 2>&1
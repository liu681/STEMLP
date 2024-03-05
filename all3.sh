#!/bin/bash
python examples/run.py -c examples/STEMLP/STEMLP_PEMS04.py --gpus '0'
python examples/run.py -c examples/STEMLP/STEMLP_PEMS07.py --gpus '0'
python examples/run.py -c examples/STEMLP/STEMLP_PEMS08.py --gpus '0'
python examples/run.py -c examples/STEMLP/STEMLP_TFA.py --gpus '0'
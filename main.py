"""File that summarizes all key results.

To train networks in a specific experiment, run in command line
python main.py --train experiment_name
To analyze results from this experiment
python main.py --analyze experiment_name

To train and analyze all models quickly, run in command line
python main.py --train --analyze

To reproduce the results from paper, run
python main.py --train --analyze

To analyze pretrained networks, run
python main.py --analyze

To run specific experiments (e.g. orn2pn, vary_pn), run
python main.py --train --analyze --experiment orn2pn vary_pn
"""

import os
import platform
import sys
import argparse

from experiment_utils import train_experiment, analyze_experiment


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--device', help='CUDA device number', default=0,
                    type=int)
parser.add_argument('-t', '--train', help='Training', nargs='+', default=[])
parser.add_argument('-a', '--analyze', help='Analyzing', nargs='+', default=[])
parser.add_argument('--no-general', help='No general analysis',
                    dest='general', action='store_false')
parser.set_defaults(general=True)
args = parser.parse_args()

# For running from IDE instead of command line
if len(sys.argv) == 1: #no command line parameters passed
    args.train = [] #add experiments here
    args.analyze = []

for item in args.__dict__.items():
    print(item)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

experiments2train = args.train
experiments2analyze = args.analyze
use_cluster = 'columbia' in platform.node()  # on columbia cluster

for experiment in experiments2train:
    train_experiment(experiment, use_cluster=use_cluster)

for experiment in experiments2analyze:
    analyze_experiment(experiment, general=args.general)

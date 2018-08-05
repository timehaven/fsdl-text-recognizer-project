#!/bin/sh
# pipenv run python training/run_experiment.py --save '{"dataset": "IamLinesDataset", "model": "LineModelCtc", # "network": "line_lstm_ctc"}'


pipenv run python training/run_experiment.py --save '{"dataset": "IamLinesDataset", "model": "LineModelCtc", "network": "line_lstm_ctc", "train_args": {"epochs": 60, "username": "timehaven", "notes": "3stacked"}}' --gpu=1

pipenv run python training/run_experiment.py --save '{"dataset": "IamLinesDataset", "model": "LineModelCtc", "network": "line_lstm_ctc", "train_args": {"epochs": 39, "username": "timehaven", "notes": "128x128"}}' --gpu=0


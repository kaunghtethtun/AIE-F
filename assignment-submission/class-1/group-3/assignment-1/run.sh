#!/bin/bash
echo "[1/3] Starting Training..."
python hybrid-eliza.py --lang my --mode train --data data/train.csv --epochs 10 --val_split 0.2 --model_path ./models/eliza_mm8.pth | tee logs/training_log_8.log

echo "[2/3] Starting Testing..."
python test.py --model ./models/eliza_mm8.pth --test_data data/test.csv

echo "[3/3] Starting Interactive Chat..."
python hybrid-eliza.py --lang my --mode chat --model_path ./models/eliza_mm8.pth | tee logs/chat_log_8.log


#!/bin/bash

mkdir -p ./data
hf download baohao/sage_train --repo-type dataset --local-dir ./data/sage_train
hf download baohao/sage_validation --repo-type dataset --local-dir ./data/sage_validation
cp ./data/sage_train/train.parquet ./data/train.parquet
cp ./data/sage_validation/test.parquet ./data/test.parquet
rm -rf ./data/sage_train
rm -rf ./data/sage_validation
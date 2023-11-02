#!/bin/bash

# label=case2
dataset=case2
input="../../data/${dataset}/train.txt"
# input="../../data/multi-label/${label}/train.txt"
lr=0.25
ngram=5
epoch=80
dim=150
test_data="../../data/${dataset}/test.txt"
# test_data="../../data/multi-label/${label}/test.txt"
python main.py \
    --input $input \
    --lr $lr \
    --ngram $ngram \
    --epoch $epoch \
    --dim $dim \
    --test $test_data \
    > ../../log/fasttext/log_${dataset}_lr${lr}_ngram${ngram}_epoch${epoch}_dim${dim}_$(date +%Y_%m_%d_%H_%M_%S).txt

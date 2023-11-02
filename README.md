# FastText_Filter
Use fasttext to classify text.

## Preparing the dataset
python train_data.py

python val_data.py

## Train model
python train.py

## Test model
python val.py

## Train and val
sh run_main.sh

## Validate model in Mapreduce
cat val_data | python map.py

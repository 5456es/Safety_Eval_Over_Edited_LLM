#!/bin/bash


cd  ../../experiment/Qlora

pwd

data_parts=(0 1 2)
data_source=ZsRE
data_sizes=(1)

for data_part in "${data_parts[@]}"; do
    for data_size in "${data_sizes[@]}"; do
        bash train_mistral-7b-instruct-v0.3_wo_tmp.sh "$data_part" "$data_source" "$data_size"
    done
done

# data_part , data_source , data_size




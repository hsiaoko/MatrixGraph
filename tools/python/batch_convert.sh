#!/bin/bash


# 版本2：数字补零 (q01.g, q02.g, ..., q99.g)
for i in {500..999}; do
    #printf -v num "%02d" $i
    echo "Processing graph $i..."
    #python graph_reader.py "/data/zhuxiaoke/workspace/Rapids_workspace/queries/1000_label_range_6/graph_$i.txt" "/data/zhuxiaoke/workspace/Torch/pt/queries/1000_label_range_6/q$i.pt"
    #mkdir /data/zhuxiaoke/workspace/Torch/embedding/queries/1000_label_range_6/q$i/
    python embedding.py  /data/zhuxiaoke/workspace/Torch/pt/queries/1000_label_range_6/q$i.pt /data/zhuxiaoke/workspace/Torch/embedding/queries/1000_label_range_6/q$i/
done

echo "All queries processed!"

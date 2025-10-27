#!/bin/bash

# 版本2：数字补零 (q01.g, q02.g, ..., q99.g)
for i in {1..500}; do
    #printf -v num "%02d" $i
    echo "Processing graph $i..."
    #mv /data/zhuxiaoke/workspace/Rapids_workspace/queries/1000/graph_${i}.txt /data/zhuxiaoke/workspace/Rapids_workspace/queries/1000/graph_4${i}.txt
    #mv /data/zhuxiaoke/workspace/Rapids_workspace/queries/1000/graph_${i}.txt /data/zhuxiaoke/workspace/Rapids_workspace/queries/1000/graph_${i+500}.txt
    ls /data/zhuxiaoke/workspace/Rapids_workspace/queries/1000/graph_${i}.txt
done

echo "All queries processed!"

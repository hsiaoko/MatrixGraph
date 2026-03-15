#!/bin/bash

for i in {1..500}; do
    echo "Processing $i..."
    ls command_${i}.txt
done

echo "All commands  processed!"

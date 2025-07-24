#!/bin/bash

# 查找所有llamafactory进程并kill它们
pids=$(ps -ef | grep llama | grep -v grep | awk '{print $2}')

if [ -z "$pids" ]; then
    echo "No llama processes found"
else
    echo "Killing llama processes:"
    echo "$pids"
    kill -9 $pids
fi
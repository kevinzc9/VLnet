#!/bin/bash

# 获取子目录的子目录个数
get_subdirectory_count() {
    local count=0
    for dir in "$1"/*/; do
        if [ -d "$dir" ]; then
            ((count++))
        fi
    done
    echo "$count"
}

# 获取当前目录
current_dir=$(pwd)

# 遍历当前目录下的所有子目录
for dir in "$current_dir"/*/; do
    if [ -d "$dir" ]; then
        # 获取子目录的子目录个数
        subdirectory_count=$(get_subdirectory_count "$dir")
        
        # 打印子目录及其对应的个数
        echo "子目录: $dir"
        echo "子目录个数: $subdirectory_count"
        echo "---------------------"
    fi
done

#!/bin/bash

if [ "$1" = "-a" ]; then
    file="kddcup.data.gz"
else
    file="kddcup.data_10_percent.gz"
fi
dir="data"

# Check if the directory exists
if [ -d "$dir" ]; then
    echo "The directory '$dir' already exists."
else
    echo "The directory '$dir' does not exist. Creating it now."
    mkdir "$dir"
fi
if [ -d "$dir/output" ]; then
    echo "The directory 'output' already exists."
else
    echo "The directory 'putpt' does not exist. Creating it now."
    mkdir "$dir/output"
fi

if [ -d "$dir/output/parsing" ]; then
    echo "The directory 'parsing' already exists."
else
    mkdir "$dir/output/parsing"
fi

if [ -d "$dir/output/models" ]; then
    echo "The directory 'models' already exists."
else
    mkdir "$dir/output/models"
fi
if [ -d "$dir/output/temp" ]; then
    echo "The directory 'temp' already exists."
else
    mkdir "$dir/output/temp"
fi
# Download the dataset
# KDD cup 1999 dataset
echo "Downloading KDD Cup 1999 dataset..."

wget http://kdd.ics.uci.edu/databases/kddcup99/"$file"

cp "$file" "$dir"
rm "$file"

echo "extracting data..."
cd "$dir" && gzip -d "$file"

# Futur datasets:



echo "Done!"
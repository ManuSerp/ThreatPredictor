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
if [ -d "data/output" ]; then
    echo "The directory 'output' already exists."
else
    echo "The directory 'putpt' does not exist. Creating it now."
    mkdir "data/output"
fi
echo "Downloading KDD Cup 1999 dataset..."

wget http://kdd.ics.uci.edu/databases/kddcup99/"$file"

cp "$file" "$dir"
rm "$file"

echo "extracting data..."
cd "$dir" && gzip -d "$file"



echo "Done!"
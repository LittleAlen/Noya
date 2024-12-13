#!/bin/bash
#create directory build if not exist
if ! test -d "./build";then
mkdir build
mkdir build/release
fi
rm -rf build/release
cp -rvf ./resources ./build/release
cd build
cmake ..
cmake --build .
mv Mnist* release
mv Cifar* release
mv Test* release



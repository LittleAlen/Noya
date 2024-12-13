#!/bin/bash
p=`ps -ef | grep "./build/release*" | grep -v "\*"| awk '{ print $2 }'`
top -pid $p
if ! test $? -eq 0;then
top -p $p
fi
#!/bin/bash

rpath=$1
rclone sync onedrive-ucl:${rpath} /tmp/rclone-zip-tmp/ -P 
gzip /tmp/rclone-zip-tmp/*
rclone sync ./rclone-zip-tmp/ onedrive-ucl:${rpath} -P
rm -r /tmp/rclone-zip-tmp

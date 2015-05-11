#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../caffe_nets/" && pwd )"
cd $DIR

FILE=CaffeNet.v2.caffemodel
URL=http://www.cs.berkeley.edu/~rbg/dp-dpm-data/$FILE
CHECKSUM=6e47b642e2f261090c8fecdc05a57546

if [ -f $FILE ]; then
  echo "File already exists. Checking md5..."
  os=`uname -s`
  if [ "$os" = "Linux" ]; then
    checksum=`md5sum $FILE | awk '{ print $1 }'`
  elif [ "$os" = "Darwin" ]; then
    checksum=`cat $FILE | md5`
  fi
  if [ "$checksum" = "$CHECKSUM" ]; then
    echo "Checksum is correct. No need to download."
    exit 0
  else
    echo "Checksum is incorrect. Need to download again."
  fi
fi

echo "Downloading pretrained ImageNet model (233M)..."

wget $URL -O $FILE

echo "Done. Please run this command again to verify that checksum = $CHECKSUM."

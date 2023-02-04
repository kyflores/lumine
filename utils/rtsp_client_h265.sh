#!/bin/bash

gst-launch-1.0 \
    udpsrc port="${1}" ! \
    application/x-rtp,encoding-name=H265,payload=96 ! \
    rtph265depay ! \
    h265parse ! \
    queue ! \
    avdec_h265 ! \
    videoconvert ! \
    fpsdisplaysink sync=false -e

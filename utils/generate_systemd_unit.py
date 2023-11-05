#!/usr/bin/python3

import os

# We hide stdout so that we don't get spammed by this error message
# "Corrupt JPEG data: 1 extraneous bytes before marker 0xd3"
# It seems to only happen on logitech webcams and doesn't affect the image.
UNIT_FILE = """
[Unit]
Description=Lumine Inference Service

[Service]
Type=simple
StandardOutput=null
ExecStart=bash {lumine_root}/start.sh

[Install]
WantedBy=multi-user.target
"""

if __name__ == "__main__":
    curdur = os.getcwd()
    formats = {"lumine_root": curdur}

    unit_file = UNIT_FILE.format(**formats)

    with open("lumine.service", "w") as service:
        service.write(unit_file)

    exit(0)

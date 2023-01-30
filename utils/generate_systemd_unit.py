#!/usr/bin/python3

import os

UNIT_FILE = """
[Unit]
Description=Lumine Inference Service

[Service]
Type=simple
ExecStart=bash {lumine_root}/start.sh
EnvironmentFile={lumine_root}/config/lumine.conf

[Install]
WantedBy=multi-user.target
"""

if __name__ == '__main__':
    curdur = os.getcwd()
    formats = {
        "lumine_root": curdur
    }

    unit_file = UNIT_FILE.format(**formats)

    with open('lumine.service', 'w') as service:
        service.write(unit_file)
    
    exit(0)
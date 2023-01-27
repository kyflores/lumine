from networktables import NetworkTables
import time
import sys


NetworkTables.initialize()
print("Network tables up")
time.sleep(1)

path_watch = sys.argv[1]
table = NetworkTables.getTable(path_watch)

while 1:
    print(table.getKeys())
    time.sleep(1)

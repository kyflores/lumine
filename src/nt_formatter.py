# Restructure Lumine format into something that more usable by network tables.
# Temporary and kind of hacked up code.
from networktables import NetworkTables
import time
import cv2
import threading


class NtFormatter:
    def __init__(self, addr, parent_table="lumine", nt_server_override=None):
        ip = ""
        if len(addr) == 4:
            (lower, upper) = addr[:2], addr[2:]
            assert len(lower) == 2
            assert len(upper) == 2
            ip = "10.{}.{}.2".format(lower, upper)
        else:
            ip = addr

        if nt_server_override is None:
            print("Connection to networktables server on {}".format(ip))
            NetworkTables.initialize(server=ip)
        else:
            print("Connection to networktables server on {}".format(nt_server_override))
            NetworkTables.initialize(server=nt_server_override)

        time.sleep(1)
        # cond = threading.Condition()
        # notified = [False]
        # # Block until network tables is ready: https://robotpy.readthedocs.io/en/stable/guide/nt.html#networktables-guide
        # def connectionListener(connected, info):
        #     print(info, "; Connected=%s" % connected)
        #     with cond:
        #         notified[0] = True
        #         cond.notify()
        # NetworkTables.addConnectionListener(connectionListener, immediateNotify=True)
        # with cond:
        #     print("Waiting")
        #     if not notified[0]:
        #         cond.wait()

        self.parent_table = NetworkTables.getTable(parent_table)

    # Converts detections from "array of structs" to "struct of arrays"
    def update_yolo(self, detlist):
        # TODO: need to cache this for perf reasons?
        yolotbl = self.parent_table.getSubTable("yolo")

        ids = []  # int array
        sort_id = []
        area = []
        xmin = []
        ymin = []
        xmax = []
        ymax = []
        confidence = []  # float 0 < x < 1
        for d in detlist:
            if d["type"] == "yolov5":
                corners = d["corners"]
                assert corners.shape == (4, 2)
                (x0, y0, w, h) = cv2.boundingRect(corners.astype(int))
                area.append(w * h)

                ids.append(d["id"])
                sort_id.append(d.get("sort_id", -1))
                xmin.append(x0)
                ymin.append(x0)
                xmax.append(x0 + w)
                ymax.append(y0 + h)
                confidence.append(d["confidence"])
            else:
                continue

        yolotbl.putNumber("len", len(ids))
        yolotbl.putNumberArray("ids", ids)
        yolotbl.putNumberArray("sort_id", sort_id)
        yolotbl.putNumberArray("area", area)
        yolotbl.putNumberArray("xmin", xmin)
        yolotbl.putNumberArray("ymin", ymin)
        yolotbl.putNumberArray("xmax", xmax)
        yolotbl.putNumberArray("ymax", ymax)
        yolotbl.putNumberArray("confidence", confidence)

    def update_apriltags_rpy(self, detlist):
        atagtbl = self.parent_table.getSubTable("rpyapriltags")

        ids = []  # int array
        sort_id = []
        area = []
        xmin = []
        ymin = []
        xmax = []
        ymax = []
        confidence = []  # float 0 < x < 1

        txs = []
        tys = []
        tzs = []
        rxs = []
        rys = []
        rzs = []

        for d in detlist:
            if d["type"] == "apriltags":
                corners = d["corners"]
                assert corners.shape == (4, 2)
                (x0, y0, w, h) = cv2.boundingRect(corners.astype(int))
                area.append(w * h)

                ids.append(d["id"])
                sort_id.append(d.get("sort_id", -1))
                xmin.append(x0)
                ymin.append(x0)
                xmax.append(x0 + w)
                ymax.append(y0 + h)
                confidence.append(d["confidence"])

                tx, ty, tz = d["translation"]
                rx, ry, rz = d["rotation_euler"]
                txs.append(tx)
                tys.append(ty)
                tzs.append(tz)
                rxs.append(rx)
                rys.append(ry)
                rzs.append(rz)

                atagtbl.putNumber("len", len(ids))
                atagtbl.putNumberArray("ids", ids)
                atagtbl.putNumberArray("sort_id", sort_id)
                atagtbl.putNumberArray("area", area)
                atagtbl.putNumberArray("xmin", xmin)
                atagtbl.putNumberArray("ymin", ymin)
                atagtbl.putNumberArray("xmax", xmax)
                atagtbl.putNumberArray("ymax", ymax)
                atagtbl.putNumberArray("confidence", confidence)
                atagtbl.putNumberArray("tx", txs)
                atagtbl.putNumberArray("ty", tys)
                atagtbl.putNumberArray("tz", tzs)
                atagtbl.putNumberArray("rx", rxs)
                atagtbl.putNumberArray("ry", rys)
                atagtbl.putNumberArray("rz", rzs)
            else:
                continue

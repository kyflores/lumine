# Restructure Lumine format into something that more usable by network tables.
# Temporary and kind of hacked up code.
import ntcore
import time
import cv2


class NtcoreFormatter:
    def __init__(
        self, addr, parent_table="lumine", nt_server_override=None, update_rate=15
    ):
        ip = ""
        if len(addr) == 4:
            (lower, upper) = addr[:2], addr[2:]
            assert len(lower) == 2
            assert len(upper) == 2
            ip = "10.{}.{}.2".format(lower, upper)
        else:
            ip = addr

        self.nt = ntcore.NetworkTableInstance.getDefault()
        if nt_server_override is None:
            print("Connection to networktables server on {}".format(ip))
            self.nt.setServer(ip)
            self.nt.startDSClient()
        else:
            print("Connection to networktables server on {}".format(nt_server_override))
            self.nt.setServer(nt_server_override)
            self.nt.startDSClient()

        time.sleep(1)

        self.parenttbl = self.nt.getTable(parent_table)

        self.yolotbl = self.parenttbl.getSubTable("yolo")
        self.yolopub = self.setup_publishers(
            self.yolotbl,
            ["ids", "sort_id", "area", "xmin", "ymin", "xmax", "ymax", "confidence"],
        )
        self.yolopub["len"] = self.yolotbl.getIntegerTopic("len").publish()

        self.atagtbl = self.parenttbl.getSubTable("rpyapriltags")
        self.atagpub = self.setup_publishers(
            self.atagtbl,
            [
                "ids",
                "sort_id",
                "area",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
                "confidence",
                "tx",
                "ty",
                "tz",
                "rx",
                "ry",
                "rz",
            ],
        )
        self.atagpub["len"] = self.atagtbl.getIntegerTopic("len").publish()

        self.nt.startClient4("lumine-client")

    def setup_publishers(self, table, pub_names_list):
        ret = {}
        for nm in pub_names_list:
            ret[nm] = table.getFloatArrayTopic(nm).publish()

        return ret

    # Converts detections from "array of structs" to "struct of arrays"
    def update_yolo(self, detlist):
        ids = []  # int array
        sort_id = []
        area = []
        xmin = []
        ymin = []
        xmax = []
        ymax = []
        confidence = []  # float 0 < x < 1
        for d in detlist:
            if d["type"] == "yolov8":
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

        self.yolopub["len"].set(len(ids))
        self.yolopub["ids"].set(ids)
        self.yolopub["sort_id"].set(sort_id)
        self.yolopub["area"].set(area)
        self.yolopub["xmin"].set(xmin)
        self.yolopub["ymin"].set(ymin)
        self.yolopub["xmax"].set(xmax)
        self.yolopub["ymax"].set(ymax)
        self.yolopub["confidence"].set(confidence)

    def update_apriltags_rpy(self, detlist):
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

            else:
                continue

        self.atagpub["len"].set(len(ids))
        self.atagpub["ids"].set(ids)
        self.atagpub["sort_id"].set(sort_id)
        self.atagpub["area"].set(area)
        self.atagpub["xmin"].set(xmin)
        self.atagpub["ymin"].set(ymin)
        self.atagpub["xmax"].set(xmax)
        self.atagpub["ymax"].set(ymax)
        self.atagpub["confidence"].set(confidence)
        self.atagpub["tx"].set(txs)
        self.atagpub["ty"].set(tys)
        self.atagpub["tz"].set(tzs)
        self.atagpub["rx"].set(rxs)
        self.atagpub["ry"].set(rys)
        self.atagpub["rz"].set(rzs)

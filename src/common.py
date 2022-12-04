# Wrappers for different detector methods
import os
import prettytable as pt


def detections_as_table(detections):
    table = pt.PrettyTable()
    table.field_names = ["type", "what", "sort_id", "xyxy"]
    for d in detections:
        table.add_row(
            [d["type"], d["id"], d.get("sort_id", -1), d["sort_xyxy"][:4].astype(int)]
        )

    return table.get_string()

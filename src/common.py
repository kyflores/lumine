# Wrappers for different detector methods
import os
import prettytable as pt

def detections_as_table(detections):
    table = pt.PrettyTable()
    table.field_names = ["type", "what", "sort_id", "corners", "confidence"]
    for d in detections:
        table.add_row(
            [d["type"], d["id"], d.get("sort_id", -1), d["corners"].astype(int), d["confidence"]]
        )

    return table.get_string()

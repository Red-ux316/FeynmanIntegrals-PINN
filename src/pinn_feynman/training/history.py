"""Per-epoch training record."""

import csv


class History:
    """List of per-epoch rows (dicts), saved as CSV with the union of their keys (older rows may lack new columns)."""

    def __init__(self):
        self.rows = []

    def append(self, row):
        self.rows.append(dict(row))

    def column(self, key):
        return [row.get(key) for row in self.rows]

    def __len__(self):
        return len(self.rows)

    def state_dict(self):
        return {"rows": [dict(r) for r in self.rows]}

    def load_state_dict(self, state):
        self.rows = [dict(r) for r in state["rows"]]

    def save_csv(self, path):
        if not self.rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as stream:
            fields = list(dict.fromkeys(key for row in self.rows for key in row))
            writer = csv.DictWriter(stream, fieldnames=fields, restval="")
            writer.writeheader()
            writer.writerows(self.rows)

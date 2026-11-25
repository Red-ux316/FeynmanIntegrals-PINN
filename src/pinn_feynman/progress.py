"""Console output of the commands: progress bars, notes, phase headers, tables.

One place for the look of everything printed while data, trainings and tuning studies run.
Everything goes to stdout, so messages and bars come out in order, also when a notebook streams
the output of a subprocess (`stream_command`).

- `bar(total, desc, ...)`: a tqdm bar for an iterative process (points, epochs, trials);
- `note(text)`: a line printed above the active bar without breaking it;
- `header(text)`: the start of a phase, e.g. `== data: box_massless/r3 ==`;
- `table(rows)`: aligned columns (queue status, outcomes);
- `duration(seconds)`: `4 min 43 s`.

PINN_FEYNMAN_PROGRESS=0 turns the bars off (plain logs); notes, headers and tables stay.
"""

import codecs
import os
import subprocess
import sys

from tqdm import tqdm

BAR_FORMAT = "{desc} {percentage:3.0f}%|{bar:24}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}"


def enabled():
    """False when PINN_FEYNMAN_PROGRESS is 0/false/no/off."""
    return os.environ.get("PINN_FEYNMAN_PROGRESS", "1").strip().lower() not in ("0", "false", "no", "off")


def bar(total, desc, unit="it", initial=0, leave=True, disable=False):
    """tqdm bar on stdout; refreshed at most once per second when stdout is not a terminal (notebooks, logs).

    The bar itself has a fixed width; the line is cut to the width of a terminal, never elsewhere.
    """
    stream = sys.stdout
    interactive = hasattr(stream, "isatty") and stream.isatty()
    return tqdm(total=total, desc=desc, unit=unit, initial=initial, leave=leave, file=stream,
                disable=disable or not enabled(), bar_format=BAR_FORMAT, mininterval=0.2 if interactive else 1.0,
                dynamic_ncols=interactive)


def note(text):
    """Print a line; an active bar is redrawn below it."""
    tqdm.write(str(text), file=sys.stdout)
    sys.stdout.flush()


def header(text):
    note(f"== {text} ==")


def table(rows, indent="  "):
    """Print rows (sequences of cells) with aligned columns."""
    rows = [[str(cell) for cell in row] for row in rows]
    if not rows:
        return
    widths = [max(len(row[k]) for row in rows if k < len(row)) for k in range(max(map(len, rows)))]
    for row in rows:
        note(indent + "  ".join(cell.ljust(widths[k]) for k, cell in enumerate(row)).rstrip())


def duration(seconds):
    """Human-readable duration: `38.4 s`, `4 min 43 s`, `2 h 05 min`."""
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, secs = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes} min {secs:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"


def stream_command(cmd, cwd=None, out=None):
    """Run a shell command and copy its output (stderr merged into stdout) to `out` as it arrives.

    Carriage returns pass through unchanged, so a progress bar in the command is redrawn in place
    (terminal, Jupyter); Windows line ends (CR LF) become LF. Returns the exit code.
    """
    out = out if out is not None else sys.stdout
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    process = subprocess.Popen(cmd, shell=True, cwd=cwd, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending = ""                             # a CR at the end of a chunk may be the first half of CR LF
    while True:
        chunk = process.stdout.read1(4096)
        if not chunk:
            break
        text = (pending + decoder.decode(chunk)).replace("\r\n", "\n")
        text, pending = (text[:-1], "\r") if text.endswith("\r") else (text, "")
        out.write(text)
        out.flush()
    out.write(pending + decoder.decode(b"", final=True))
    process.stdout.close()
    return process.wait()

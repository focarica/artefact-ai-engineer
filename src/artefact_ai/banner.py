"""Colored ASCII robot shown when the interactive chat starts."""

from itertools import groupby
import os
import sys

import typer

# Robot with headphones: antenna, band arc, ear cups, visor, eyes and mouth.
ROBOT = (
    "                           @@@@",
    "                            @@",
    "                            @@",
    "                       @@@@@@@@@@@@",
    "                    @@@            @@@",
    "                   @@                @@",
    "                  @@  ++++++++++++++  @@",
    "              @@@@@ ++%%%%%%%%%%%%%%++ @@@@@",
    "             @####@ +%%            %%+ @####@",
    "             @#@@#@ +% OOO      OOO %+ @#@@#@",
    "             @#@@#@ +% OOO      OOO %+ @#@@#@",
    "             @#@@#@ +%              %+ @#@@#@",
    "             @####@ +%   \\      /   %+ @####@",
    "              @@@@  +%    \\____/    %+  @@@@",
    "                    +%%            %%+",
    "                    +%%%%%%%%%%%%%%%%+",
    "                    ++%%%%%%%%%%%%%%++",
    "                      ++++++++++++++",
)

COLORS = {
    "@": typer.colors.BRIGHT_BLUE,
    "#": typer.colors.MAGENTA,
    "%": typer.colors.CYAN,
    "+": typer.colors.BLUE,
    "O": typer.colors.BRIGHT_GREEN,
    "\\": typer.colors.BRIGHT_GREEN,
    "/": typer.colors.BRIGHT_GREEN,
    "_": typer.colors.BRIGHT_GREEN,
}

WIDTH = 58  # canvas width; the figure is centered on it


def color_enabled() -> bool:
    return sys.stdout.isatty() and 'NO_COLOR' not in os.environ


def styled(text: str, color: str | None = None, *, bold: bool = False) -> str:
    if not color_enabled():
        return text
    return typer.style(text, fg=color, bold=bold)


def _paint(line: str) -> str:
    if not color_enabled():
        return line
    parts = []
    for color, group in groupby(line, key=lambda char: COLORS.get(char)):
        text = ''.join(group)
        parts.append(typer.style(text, fg=color) if color else text)
    return ''.join(parts)


def banner() -> list[str]:
    """Colored robot lines ready to print."""
    return [_paint(line.ljust(WIDTH)) for line in ROBOT]

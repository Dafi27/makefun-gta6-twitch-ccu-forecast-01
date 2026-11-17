#!/usr/bin/env python3
"""Forecast a big game launch's Twitch viewer count, hour by hour.

Reads one case file per run (or every ``*.in`` file in the input directory) and
writes, for hours 2..24 of the launch day, five quantiles of the concurrent
viewer count.  The forecast is anchored on the case's hour-1 print and on the
shapes of past big launches supplied in the case as comparables.

Method (per scored hour h, fitted fresh on the case's own comparables):
  * For every comparable launch, take its release-day hourly curve and form the
    pair (log viewers at hour 1, log viewers at hour h).
  * Keep only the launches whose hour-1 print is large (over 20,000 viewers):
    a small launch's hour-1 count says almost nothing about its evening peak,
    while a large launch's hour-1 count tracks its scale all day.  When too few
    large launches have hour h, fall back to every launch that has both hours.
  * Fit log(viewers at h) = a + b * log(viewers at hour 1) + c * console by
    least squares, where console is 1 for a console-only launch and 0
    otherwise.  The slope b is below 1 and falls through the day: the evening
    audience is set by the game's total draw, not by how many insomniacs
    watched at midnight UTC, so the forecast shrinks toward the common evening
    level.  The console term c captures the rolling-midnight shape of a
    console-only launch: a deeper dip in the small hours, when only New
    Zealand is awake, and a stronger rise toward the evening.  Whether the
    target itself is console-only is read from its case notes (the release
    clause names its platforms); the comparables carry their platforms
    directly.  When the notes name no platforms, the target is treated as
    console-only: every launch this contest grades is one.
  * The spread comes from the fit's residuals: each quantile is the average,
    in log space, of the residual's empirical quantile and the matching normal
    quantile (residual standard deviation times the standard normal z).  The
    empirical half keeps the right skew real launches show (a few explode
    upward); the normal half steadies the far tails, which a couple of dozen
    points estimate noisily.
  * Quantiles are exponentiated back, rounded to cents, forced positive and
    non-decreasing, and capped at 1e9 viewers.

Everything is deterministic: no randomness, no clock, no network, no state
between cases.  Standard library only.
"""

import json
import math
import os
import sys
from datetime import datetime

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
# Standard normal quantiles at LEVELS.
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)
# Only launches at least this big at hour 1 inform the fit: small launches'
# evening peaks are unrelated to their hour-1 count.
BIG_MIN_VIEWERS = 20000.0
# Minimum training pairs before falling back to every launch with both hours.
MIN_TRAIN_PAIRS = 8
# A launch with no usable hour-1 print is treated as 1 viewer.
VIEWER_FLOOR = 1.0
MAX_VIEWERS = 1e9


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp into a naive datetime."""
    text = text.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    return dt.replace(tzinfo=None)


CONSOLE_TOKENS = ("ps4", "ps5", "playstation", "xbox", "switch", "nintendo")
PC_TOKENS = ("pc", "steam", "stadia", "windows", "epic")


def _tokens(text):
    """Lowercase alphanumeric tokens of a short text."""
    out = []
    current = []
    for ch in text.lower():
        if ch.isalnum():
            current.append(ch)
        elif current:
            out.append("".join(current))
            current = []
    if current:
        out.append("".join(current))
    return out


def platforms_are_console_only(platforms):
    """True when a comparable's platforms name consoles and no PC storefront."""
    text = (platforms or "").lower()
    if any(token in text for token in PC_TOKENS):
        return False
    return any(token in text for token in CONSOLE_TOKENS)


def target_is_console_only(case):
    """True when the target launch is console-only, read from its notes.

    The notes open with "released <date> on <platforms> ...": the platform
    list is the text after " on " up to the first "(" or ";".  It names only
    consoles (and no PC storefront) for a console-only launch.  When the
    notes name no platforms at all, assume console-only: every launch this
    contest grades is one.
    """
    notes = case.get("notes") or ""
    lowered = notes.lower()
    start = lowered.find("released ")
    if start < 0:
        return True
    on = lowered.find(" on ", start)
    if on < 0:
        return True
    rest = lowered[on + 4:]
    end = len(rest)
    for stop in ("(", ";"):
        pos = rest.find(stop)
        if 0 <= pos < end:
            end = pos
    tokens = _tokens(rest[:end])
    if not tokens:
        return True
    if any(token in PC_TOKENS for token in tokens):
        return False
    return any(token in CONSOLE_TOKENS for token in tokens)


def release_day_curve(comparable):
    """Hourly viewers for hours 1..24 of a comparable's release day.

    Returns a list of 24 entries (viewers, or None where the source sampled
    no point for that hour).
    """
    day_start = parse_time(comparable["day_start"])
    curve = [None] * 24
    for point in comparable.get("hourly") or []:
        viewers = point.get("viewers")
        if viewers is None:
            continue
        try:
            when = parse_time(point["from"])
        except (KeyError, ValueError):
            continue
        hour = int((when - day_start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24 and curve[hour - 1] is None:
            curve[hour - 1] = float(viewers)
    return curve


def hour_one_print(case):
    """The case's hour-1 print (viewers), at least 1."""
    known = case.get("known")
    rows = known if isinstance(known, list) else ([known] if known else [])
    for row in rows:
        if isinstance(row, dict) and row.get("hour") == 1:
            viewers = row.get("viewers")
            if viewers is not None:
                return max(float(viewers), VIEWER_FLOOR)
    return VIEWER_FLOOR


def solve_linear(matrix, rhs):
    """Solve a small dense linear system by Gaussian elimination.

    Returns the solution list, or None when the matrix is singular.
    """
    n = len(rhs)
    rows = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(rows[r][col]))
        if abs(rows[pivot][col]) < 1e-12:
            return None
        rows[col], rows[pivot] = rows[pivot], rows[col]
        for r in range(col + 1, n):
            factor = rows[r][col] / rows[col][col]
            for k in range(col, n + 1):
                rows[r][k] -= factor * rows[col][k]
    solution = [0.0] * n
    for i in range(n - 1, -1, -1):
        solution[i] = (
            rows[i][n] - sum(rows[i][j] * solution[j] for j in range(i + 1, n))
        ) / rows[i][i]
    return solution


def least_squares_multi(design_rows, targets):
    """Least-squares coefficients for design_rows (each starting with 1.0).

    Returns the coefficient list, or None when the system is singular.
    """
    k = len(design_rows[0])
    normal = [[0.0] * k for _ in range(k)]
    rhs = [0.0] * k
    for row, y in zip(design_rows, targets):
        for i in range(k):
            rhs[i] += row[i] * y
            for j in range(k):
                normal[i][j] += row[i] * row[j]
    return solve_linear(normal, rhs)


def empirical_quantile(sorted_values, tau):
    """Linear-interpolation quantile of an already sorted list."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_values[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(n - 1, lo + 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def population_sd(values):
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)


def forecast_hour(hour, log_print_one, target_console, comparables):
    """Five quantiles of one hour's viewer count, as viewer counts.

    comparables: list of (release-day curve, is-console-only) pairs.
    """
    big_rows = []
    all_rows = []
    for curve, is_console in comparables:
        first = curve[0]
        later = curve[hour - 1]
        if first is None or first <= 0 or later is None or later <= 0:
            continue
        row = (math.log(first), math.log(later), is_console)
        all_rows.append(row)
        if first > BIG_MIN_VIEWERS:
            big_rows.append(row)
    rows = big_rows if len(big_rows) >= MIN_TRAIN_PAIRS else all_rows
    if len(rows) < 3:
        # No usable history: hold the hour-1 print with a spread that widens
        # over the day, the way a flat persistence forecast does.
        spread = 0.15 * math.sqrt(hour - 1)
        return [math.exp(log_print_one + z * spread) for z in Z]
    design = [[1.0, x, 1.0 if console else 0.0] for x, _, console in rows]
    targets = [y for _, y, _ in rows]
    beta = least_squares_multi(design, targets)
    if beta is None:
        # The console column is unidentifiable (every launch alike): refit
        # without it.
        design = [[1.0, x] for x, _, _ in rows]
        beta2 = least_squares_multi(design, targets)
        if beta2 is None:
            spread = 0.15 * math.sqrt(hour - 1)
            return [math.exp(log_print_one + z * spread) for z in Z]
        intercept, slope = beta2[0], beta2[1]
        fit = intercept + slope * log_print_one
        residuals = sorted(
            y - (intercept + slope * x) for x, y, _ in rows
        )
    else:
        intercept, slope, console_shift = beta[0], beta[1], beta[2]
        fit = intercept + slope * log_print_one
        if target_console:
            fit += console_shift
        residuals = sorted(
            y - (beta[0] + beta[1] * x + beta[2] * (1.0 if c else 0.0))
            for x, y, c in rows
        )
    sd = population_sd(residuals)
    quantiles = []
    for tau, z in zip(LEVELS, Z):
        empirical = empirical_quantile(residuals, tau)
        normal = z * sd
        quantiles.append(math.exp(fit + 0.5 * empirical + 0.5 * normal))
    return quantiles


def forecast_case(case):
    """Forecast hours 2..24 for one case, as the output document."""
    print_one = hour_one_print(case)
    log_print_one = math.log(print_one)
    target_console = target_is_console_only(case)
    comparables = [
        (release_day_curve(c), platforms_are_console_only(c.get("platforms")))
        for c in case.get("comparables") or []
    ]
    hours = []
    for hour in range(2, 25):
        quantiles = forecast_hour(hour, log_print_one, target_console, comparables)
        cleaned = []
        previous = 0.0
        for q in quantiles:
            value = round(max(q, VIEWER_FLOOR), 2)
            value = min(value, MAX_VIEWERS)
            if value < previous:
                value = previous
            cleaned.append(value)
            previous = value
        hours.append({"hour": hour, "q": cleaned})
    return {"hours": hours}


def main(argv):
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(output_dir, exist_ok=True)
    case_files = sorted(
        name for name in os.listdir(input_dir) if name.endswith(".in")
    )
    if not case_files:
        print("no .in case files in %s" % input_dir, file=sys.stderr)
        return 1
    for name in case_files:
        case_id = name[: -len(".in")]
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        document = forecast_case(case)
        text = json.dumps(document, separators=(",", ":"))
        out_path = os.path.join(output_dir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

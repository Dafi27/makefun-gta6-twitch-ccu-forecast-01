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
  * Fit log(viewers at h) = a + b * log(viewers at hour 1) by least squares.
    The slope b is below 1 and falls through the day: the evening audience is
    set by the game's total draw, not by how many insomniacs watched at
    midnight UTC, so the forecast shrinks toward the common evening level.
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


def least_squares(xs, ys):
    """Return (intercept, slope) of the least-squares line through (xs, ys)."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x <= 0.0:
        return mean_y, 0.0
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = cov / var_x
    return mean_y - slope * mean_x, slope


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


def forecast_hour(hour, log_print_one, comparables):
    """Five quantiles of one hour's viewer count, as viewer counts."""
    big_pairs = []
    all_pairs = []
    for curve in comparables:
        first = curve[0]
        later = curve[hour - 1]
        if first is None or first <= 0 or later is None or later <= 0:
            continue
        pair = (math.log(first), math.log(later))
        all_pairs.append(pair)
        if first > BIG_MIN_VIEWERS:
            big_pairs.append(pair)
    pairs = big_pairs if len(big_pairs) >= MIN_TRAIN_PAIRS else all_pairs
    if len(pairs) < 3:
        # No usable history: hold the hour-1 print with a spread that widens
        # over the day, the way a flat persistence forecast does.
        spread = 0.15 * math.sqrt(hour - 1)
        return [math.exp(log_print_one + z * spread) for z in Z]
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    intercept, slope = least_squares(xs, ys)
    fit = intercept + slope * log_print_one
    residuals = sorted(y - (intercept + slope * x) for x, y in zip(xs, ys))
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
    comparables = [release_day_curve(c) for c in case.get("comparables") or []]
    hours = []
    for hour in range(2, 25):
        quantiles = forecast_hour(hour, log_print_one, comparables)
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

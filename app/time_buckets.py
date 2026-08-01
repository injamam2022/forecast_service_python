"""Time bucket generation and seasonal period helpers."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import List


def seasonal_period(granularity: str) -> int:
    g = (granularity or "monthly").lower().strip()
    return {
        "daily": 7,
        "day": 7,
        "weekly": 52,
        "week": 52,
        "monthly": 12,
        "month": 12,
        "quarterly": 4,
        "quarter": 4,
        "yearly": 1,
        "year": 1,
    }.get(g, 12)


def generate_future_buckets(last_bucket: str, horizon: int, granularity: str) -> List[str]:
    """Generate future period labels matching R forecast_api.R conventions."""
    g = (granularity or "monthly").lower().strip()
    last = str(last_bucket or "").strip()
    out: List[str] = []

    if re.match(r"^\d{4}-\d{2}$", last) or g in ("monthly", "month"):
        # Monthly YYYY-MM
        if re.match(r"^\d{4}-\d{2}-\d{2}$", last):
            last = last[:7]
        if not re.match(r"^\d{4}-\d{2}$", last):
            last = datetime.utcnow().strftime("%Y-%m")
        y, m = int(last[:4]), int(last[5:7])
        for _ in range(horizon):
            m += 1
            if m > 12:
                m = 1
                y += 1
            out.append(f"{y:04d}-{m:02d}")
        return out

    if re.match(r"^\d{4}-\d{2}-\d{2}$", last) or g in ("daily", "day"):
        d = datetime.strptime(last[:10], "%Y-%m-%d")
        for i in range(1, horizon + 1):
            out.append((d + timedelta(days=i)).strftime("%Y-%m-%d"))
        return out

    week_m = re.match(r"^(\d{4})-W(\d{2})$", last)
    if week_m or g in ("weekly", "week"):
        if week_m:
            y, w = int(week_m.group(1)), int(week_m.group(2))
        else:
            today = datetime.utcnow()
            y, w = today.year, today.isocalendar()[1]
        for _ in range(horizon):
            w += 1
            if w > 52:
                w = 1
                y += 1
            out.append(f"{y:04d}-W{w:02d}")
        return out

    q_m = re.match(r"^(\d{4})-Q(\d)$", last)
    if q_m or g in ("quarterly", "quarter"):
        if q_m:
            y, q = int(q_m.group(1)), int(q_m.group(2))
        else:
            y, q = datetime.utcnow().year, (datetime.utcnow().month - 1) // 3 + 1
        for _ in range(horizon):
            q += 1
            if q > 4:
                q = 1
                y += 1
            out.append(f"{y:04d}-Q{q}")
        return out

    # Yearly / fallback
    try:
        y = int(last[:4])
    except Exception:
        y = datetime.utcnow().year
    for i in range(1, horizon + 1):
        out.append(str(y + i))
    return out

#!/usr/bin/env python3
"""
Test script for the departure board functionality
"""
import requests
from datetime import datetime, timezone

def fetch_departures(station, limit):
    r = requests.get("https://transport.opendata.ch/v1/stationboard",
                     params={"station": station, "limit": limit}, timeout=8)
    r.raise_for_status()
    data = r.json()
    out = []
    now = datetime.now(timezone.utc)
    for it in data.get("stationboard", []):
        stop = it.get("stop", {})
        dep_iso = stop.get("departure")
        if not dep_iso:
            continue
        dep = datetime.fromisoformat(dep_iso.replace("Z", "+00:00"))
        delta = dep - now
        mins = max(0, int(delta.total_seconds() // 60))
        secs = max(0, int(delta.total_seconds() % 60))
        delay = int(stop.get("delay", 0) or 0)
        when_local = dep.astimezone()
        cat = it.get("category") or ""
        name = it.get("name") or (cat + str(it.get("number", "")))
        to = it.get("to", "")
        out.append({
            "time": when_local.strftime("%H:%M"),
            "mins": mins, "secs": secs, "delay": delay,
            "line": name, "cat": cat, "to": to
        })
    return out

if __name__ == "__main__":
    try:
        entries = fetch_departures("Winterthur, Else Züblin", 5)
        print(f"Successfully fetched {len(entries)} departures")
        for i, entry in enumerate(entries[:3]):
            print(f"{i+1}. {entry['line']} to {entry['to']} at {entry['time']} ({entry['mins']}min)")
    except Exception as e:
        print(f"API Error: {e}")

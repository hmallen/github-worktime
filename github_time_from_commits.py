import argparse
import csv
import dataclasses
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict, Any, Set

try:
    import yaml  # pyyaml
except Exception:
    yaml = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

ISO_DATE = "%Y-%m-%d"

@dataclass
class Config:
    output_dir: Path
    cache_dir: Path
    authors: List[str]
    repos: List[str]
    since: Optional[str] = None
    until: Optional[str] = None
    session_gap_minutes: int = 90
    min_session_minutes: int = 10
    chart_title: Optional[str] = None
    github_auto_discovery: Optional[Dict[str, Any]] = None

def load_config(path: Path) -> Config:
    if yaml is None:
        print("pyyaml is required: pip install pyyaml", file=sys.stderr)
        sys.exit(2)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    output_dir = Path(raw.get("output_dir", "./_git_time_output")).resolve()
    cache_dir = Path(raw.get("cache_dir", "./_git_time_cache")).resolve()
    authors = list(raw.get("authors") or [])
    repos = list(raw.get("repos") or [])
    since = raw.get("since")
    until = raw.get("until")
    session_gap_minutes = int(raw.get("session_gap_minutes", 90))
    min_session_minutes = int(raw.get("min_session_minutes", 10))
    chart_title = raw.get("chart_title")
    github_auto_discovery = raw.get("github_auto_discovery", None)
    return Config(
        output_dir=output_dir,
        cache_dir=cache_dir,
        authors=authors,
        repos=repos,
        since=since,
        until=until,
        session_gap_minutes=session_gap_minutes,
        min_session_minutes=min_session_minutes,
        chart_title=chart_title,
        github_auto_discovery=github_auto_discovery,
    )

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def is_git_url(s: str) -> bool:
    return s.startswith("https://") or s.startswith("http://") or s.endswith(".git")

def clone_or_update(url: str, cache_dir: Path) -> Path:
    # Use a folder name derived from URL
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", url.strip("/"))
    dest = cache_dir / safe
    if dest.exists():
        try:
            subprocess.run(["git", "-C", str(dest), "fetch", "--all", "--prune"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"Warning: fetch failed for {url}: {e}", file=sys.stderr)
    else:
        try:
            subprocess.run(["git", "clone", "--no-tags", "--quiet", url, str(dest)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: clone failed for {url}: {e}", file=sys.stderr)
            return dest
    return dest

def discover_repos_via_github(cfg: Config) -> List[str]:
    gd = cfg.github_auto_discovery or {}
    if not gd or not gd.get("enabled"):
        return []
    user = gd.get("user")
    token_env = gd.get("token_env_var", "GITHUB_TOKEN")
    include_private = bool(gd.get("include_private", True))
    max_repos = int(gd.get("max_repos", 200))
    token = os.getenv(token_env)
    if not user or not token:
        print("GitHub auto discovery skipped: missing user or token env.", file=sys.stderr)
        return []
    # Minimal dependency approach: use curl via subprocess to avoid adding requests
    # Repos API: GET /user/repos for authenticated user, or GET /users/{username}/repos for public
    urls = []
    page = 1
    while len(urls) < max_repos:
        if include_private:
            endpoint = f"https://api.github.com/user/repos?per_page=100&page={page}&affiliation=owner,collaborator,organization_member&sort=updated"
            auth = ["-H", f"Authorization: Bearer {token}"]
        else:
            endpoint = f"https://api.github.com/users/{user}/repos?per_page=100&page={page}&sort=updated"
            auth = []
        try:
            res = subprocess.run(["curl", "-sS", endpoint, *auth], check=True, stdout=subprocess.PIPE)
            data = json.loads(res.stdout.decode("utf-8"))
            if not isinstance(data, list) or not data:
                break
            for repo in data:
                if repo.get("archived"):
                    continue
                urls.append(repo.get("clone_url"))
            page += 1
        except Exception as e:
            print(f"GitHub discovery error: {e}", file=sys.stderr)
            break
    urls = urls[:max_repos]
    return urls

def git_log_timestamps(repo_path: Path, authors: List[str], since: Optional[str], until: Optional[str]) -> List[int]:
    if not (repo_path / ".git").exists():
        # allow bare or worktree-less repos if cloned without .git folder structure
        pass
    cmd = ["git", "-C", str(repo_path), "log", "--pretty=%ct", "--no-merges", "--all"]
    if authors:
        # Build an --author regex that matches any email in authors list.
        # Git matches against "author name <email>"
        auth_regex = "|".join([re.escape(a) for a in authors])
        cmd += ["--extended-regexp", "--regexp-ignore-case", f"--committer=({auth_regex})"]
    if since:
        cmd += [f"--since={since}"]
    if until:
        # Make until inclusive through end of day by adding 23:59:59 if only date provided
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", until):
            cmd += [f"--until={until} 23:59:59"]
        else:
            cmd += [f"--until={until}"]
    try:
        out = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        lines = out.stdout.decode("utf-8").strip().splitlines()
        ts = [int(x) for x in lines if x.strip().isdigit()]
        return sorted(ts)
    except subprocess.CalledProcessError as e:
        print(f"git log failed in {repo_path}: {e}", file=sys.stderr)
        return []

@dataclass
class Session:
    start: int
    end: int
    commits: int
    repos: Tuple[str, ...]

def build_sessions_unified(events: List[Tuple[int, str]], gap_minutes: int, min_minutes: int) -> List[Session]:
    if not events:
        return []
    events_sorted = sorted(events, key=lambda x: x[0])
    sessions: List[Session] = []
    gap = gap_minutes * 60
    cur_start = events_sorted[0][0]
    cur_last = events_sorted[0][0]
    cur_commits = 1
    cur_repos: Set[str] = {events_sorted[0][1]}
    for t, repo in events_sorted[1:]:
        if t - cur_last <= gap:
            cur_last = t
            cur_commits += 1
            cur_repos.add(repo)
        else:
            end_padded = max(cur_last, cur_start + min_minutes * 60)
            sessions.append(Session(start=cur_start, end=end_padded, commits=cur_commits, repos=tuple(sorted(cur_repos))))
            cur_start = t
            cur_last = t
            cur_commits = 1
            cur_repos = {repo}
    end_padded = max(cur_last, cur_start + min_minutes * 60)
    sessions.append(Session(start=cur_start, end=end_padded, commits=cur_commits, repos=tuple(sorted(cur_repos))))
    return sessions

def date_str(ts: int) -> str:
    return dt.datetime.fromtimestamp(ts).strftime(ISO_DATE)

def seconds_to_hours(s: float) -> float:
    return round(s / 3600.0, 2)

def iter_day_slices(start_ts: int, end_ts: int) -> Iterable[Tuple[str, int]]:
    start_dt = dt.datetime.fromtimestamp(start_ts)
    day_start = dt.datetime(start_dt.year, start_dt.month, start_dt.day)
    while True:
        next_day_dt = day_start + dt.timedelta(days=1)
        seg_start = max(start_ts, int(day_start.timestamp()))
        seg_end = min(end_ts, int(next_day_dt.timestamp()))
        if seg_end > seg_start:
            yield day_start.strftime(ISO_DATE), seg_end - seg_start
        if end_ts <= int(next_day_dt.timestamp()):
            break
        day_start = next_day_dt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--csv", help="Override CSV output path")
    ap.add_argument("--png", help="Override PNG chart output path")
    ap.add_argument("--sessions_csv", help="Override sessions CSV output path")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    ensure_dir(cfg.output_dir)
    ensure_dir(cfg.cache_dir)

    repos: List[Tuple[str, Path]] = []  # (label, path)
    # Auto-discovery if requested
    for url in discover_repos_via_github(cfg):
        cfg.repos.append(url)
    # Normalize repos
    for entry in cfg.repos:
        if is_git_url(entry):
            p = clone_or_update(entry, cfg.cache_dir)
            label = Path(entry).stem
            repos.append((label, p))
        else:
            p = Path(entry).expanduser().resolve()
            label = p.name
            repos.append((label, p))

    events: List[Tuple[int, str]] = []
    for label, path in repos:
        ts = git_log_timestamps(path, cfg.authors, cfg.since, cfg.until)
        for t in ts:
            events.append((t, label))
    all_sessions: List[Session] = build_sessions_unified(events, gap_minutes=cfg.session_gap_minutes, min_minutes=cfg.min_session_minutes)
    commit_days: Set[str] = set(date_str(t) for t, _ in events)
    commit_ts_min: Optional[int] = min((t for t, _ in events), default=None)
    commit_ts_max: Optional[int] = max((t for t, _ in events), default=None)

    # Aggregate per day
    per_day_seconds: Dict[str, float] = defaultdict(float)
    per_day_commits: Dict[str, int] = defaultdict(int)
    per_day_sessions: Dict[str, int] = defaultdict(int)

    for s in all_sessions:
        start_day = date_str(s.start)
        per_day_sessions[start_day] += 1
        per_day_commits[start_day] += s.commits
        for day, seg_seconds in iter_day_slices(s.start, s.end):
            per_day_seconds[day] += seg_seconds

    sessions_csv_path = Path(args.sessions_csv) if args.sessions_csv else cfg.output_dir / "git_worktime_sessions.csv"
    with open(sessions_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["start","end","duration_seconds","duration_hours","commits","repos"])
        w.writeheader()
        for s in all_sessions:
            duration_seconds = max(0, s.end - s.start)
            w.writerow({
                "start": dt.datetime.fromtimestamp(s.start).isoformat(sep=" "),
                "end": dt.datetime.fromtimestamp(s.end).isoformat(sep=" "),
                "duration_seconds": duration_seconds,
                "duration_hours": f"{seconds_to_hours(duration_seconds):.2f}",
                "commits": s.commits,
                "repos": ";".join(s.repos),
            })

    # Build complete date range to include zero-hour days
    rows = []
    total_hours = 0.0
    start_date: Optional[dt.date] = None
    end_date: Optional[dt.date] = None
    if cfg.since:
        try:
            start_date = dt.datetime.strptime(cfg.since[:10], ISO_DATE).date()
        except Exception:
            start_date = None
    if cfg.until:
        try:
            end_date = dt.datetime.strptime(cfg.until[:10], ISO_DATE).date()
        except Exception:
            end_date = None
    if start_date is None or end_date is None:
        if per_day_seconds:
            keys_sorted = sorted(per_day_seconds.keys())
            if start_date is None:
                start_date = dt.datetime.strptime(keys_sorted[0], ISO_DATE).date()
            if end_date is None:
                end_date = dt.datetime.strptime(keys_sorted[-1], ISO_DATE).date()
    if start_date is not None and end_date is not None and start_date <= end_date:
        cur = start_date
        one_day = dt.timedelta(days=1)
        while cur <= end_date:
            d = cur.strftime(ISO_DATE)
            secs = per_day_seconds.get(d, 0)
            hours = seconds_to_hours(secs)
            total_hours += hours
            rows.append({
                "date": d,
                "hours": f"{hours:.2f}",
                "sessions": per_day_sessions.get(d, 0),
                "commits": per_day_commits.get(d, 0),
            })
            cur += one_day
    else:
        # Fallback: no range known, keep existing behavior
        dates_sorted = sorted(per_day_seconds.keys())
        for d in dates_sorted:
            hours = seconds_to_hours(per_day_seconds[d])
            total_hours += hours
            rows.append({
                "date": d,
                "hours": f"{hours:.2f}",
                "sessions": per_day_sessions[d],
                "commits": per_day_commits[d],
            })

    csv_path = Path(args.csv) if args.csv else cfg.output_dir / "git_worktime_daily.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date","hours","sessions","commits"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Chart
    png_path = Path(args.png) if args.png else cfg.output_dir / "git_worktime_daily.png"
    if plt and rows:
        # Build x-axis from first to last commit date, include 0-hour days
        if commit_ts_min is not None and commit_ts_max is not None:
            chart_start_date = dt.datetime.fromtimestamp(commit_ts_min).date()
            chart_end_date = dt.datetime.fromtimestamp(commit_ts_max).date()
            cur = chart_start_date
            one_day = dt.timedelta(days=1)
            x = []
            y = []
            while cur <= chart_end_date:
                d = cur.strftime(ISO_DATE)
                x.append(d)
                y.append(float(f"{seconds_to_hours(per_day_seconds.get(d, 0)):.2f}"))
                cur += one_day
        else:
            # Fallback to rows range if no commits were found
            x = [r["date"] for r in rows]
            y = [float(r["hours"]) for r in rows]
        fig = plt.figure(figsize=(12, 4))
        plt.bar(x, y)
        plt.xticks(rotation=90, fontsize=8)
        plt.ylabel("Hours")
        ttl = cfg.chart_title or "Estimated work hours per day (commit-session method)"
        plt.title(ttl)
        plt.tight_layout()
        fig.savefig(png_path, dpi=160)
        plt.close(fig)

    # Summary
    print(json.dumps({
        "csv": str(csv_path),
        "png": str(png_path),
        "sessions_csv": str(sessions_csv_path),
        "days": len(commit_days),
        "total_hours": round(total_hours, 2),
        "sessions": sum(per_day_sessions.values()),
        "commits": sum(per_day_commits.values()),
    }, indent=2))

if __name__ == "__main__":
    main()

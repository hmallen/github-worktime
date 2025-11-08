# github-worktime

Calculates hours worked from GitHub commit history.

Estimate working hours from Git commit history by clustering commits into sessions.

Approach:

- Pull commit timestamps for configured repositories and authors.
- Cluster consecutive commits into a "work session" if the gap <= session_gap_minutes.
- Session duration = max(min_session_minutes, last_commit - first_commit)
- Attribute session time to the calendar day of the session start.
- Export per-day totals as CSV and render a daily bar chart.

Notes and caveats:

- Commits are a proxy for work. This will under/over-estimate time for many workflows.
- Prefer filtering by author email(s) to avoid counting other peoples' commits.
- You can include both local paths and remote Git URLs. Remotes will be cloned to a cache folder.
- Private repos via the GitHub REST API require a token only if you use auto-discovery. Local git logs do not.
- For monorepos with infrequent commits but long coding sessions, consider raising min_session_minutes.

Usage:

  1) Create a config YAML (see example_config.yml) and edit fields.
  2) Run:
       python github_time_from_commits.py --config path/to/config.yml
  3) Outputs:
       - CSV daily report
       - PNG bar chart

Config fields:
  output_dir: where to write CSV and PNG
  cache_dir: where to clone remote repos
  authors: list of email addresses to include
  repos: list of local paths or Git HTTPS(S) URLs
  since: ISO date "YYYY-MM-DD" inclusive filter (optional)
  until: ISO date "YYYY-MM-DD" inclusive filter (optional)
  session_gap_minutes: int, default 90
  min_session_minutes: int, default 10
  chart_title: optional string
  github_auto_discovery:
    enabled: false|true
    user: "github-username"
    token_env_var: "GITHUB_TOKEN"   # env var containing a PAT with repo read perms
    include_private: true|false
    max_repos: 200

Limitations:

- Does not parse unpushed local commits unless you run it on your local clones.
- Day attribution uses session start date. Change if you prefer splitting across days.
- If you use many identities, list all emails under authors.

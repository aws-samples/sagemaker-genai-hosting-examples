# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- **Interactive TUI dashboard** (`python run.py --tui`) — Textual-based terminal UI
  - 5 tabs: Recipes, Results, Endpoints, Benchmark, Deploy
  - Arrow key navigation (left/right: tabs, up/down: rows)
  - Inline command execution (validate, deploy, benchmark) with streaming output
  - Search/filter with `/` key
  - Double-Esc to quit pattern
  - 44 recipes browsable with YAML preview, fixed column widths
  - Results tab: Tree widget with fold/unfold groups (model → config → concurrency levels)
  - 1086 benchmark results with sparkline latency trends
  - Enter → action menu (e: execute, v: validate, b: benchmark, p: preview)

### Changed
- Results tab: DataTable replaced with Tree widget for hierarchical browsing
- All DataTable columns now have fixed widths (no shrinking during search/filter)
- **Prerequisites section** in README — AWS credentials, IAM role, GPU quota setup
- **Benchmark output documentation** in README — CSV format, column descriptions
- **Benchmark data sources** in README — prompts.py use case documentation
- **Roadmap** in README — tracked action items
- `textual>=3.0.0` and `rich>=13.0.0` to requirements.txt
- `--tui` flag to `run.py`

### Changed
- Updated project structure in README to include `tui/` directory
- Recipe count updated from 31 to 44

## [3.0.0] - 2026-03-11
### Added
- Config-driven v3 architecture with YAML recipes
- 22 pre-tested recipes across 5 models, 4 instance types, 5 optimizations
- Unified CLI: `run.py` with declarative (-f recipe.yaml) and imperative (--deploy) modes
- Streaming TTFT measurement support
- EAGLE3 speculative decoding benchmarks
- HyperPod deployment support (WIP)
- Full 5-model benchmark report (FINAL_REPORT_20260311.md)

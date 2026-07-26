# Project Log

## Phase 1: Project Setup

Date: 2026-07-26

Completed the initial repository setup for the NFL predictor project.

- Created the required project folders for raw data, processed data, notebooks, source code, app code, and documentation.
- Confirmed the project uses Poetry metadata in `pyproject.toml`.
- Updated `.gitignore` so generated data and model artifacts stay out of version control while project documentation remains trackable.
- Added starter documentation files for project progress and model results.

## Upcoming Work

- Phase 3: Build the first processed modeling dataset with the `home_win` target.

## Phase 2: Data Loading

Date: 2026-07-26

Implemented `src/data/load_raw_data.py` to load completed NFL seasons 2000-2025 with `nflreadpy`.

- Data source: `nflreadpy`, backed by nflverse data.
- Seasons loaded by default: 2000-2025.
- Raw schedules/games output: `data/raw/games.parquet`.
- Raw weekly team stats output: `data/raw/team_stats.parquet`.
- Verified output from the first successful run:
  - `data/raw/games.parquet`: 7,017 rows and 46 columns.
  - `data/raw/team_stats.parquet`: 14,014 rows and 133 columns.
- Run command: `.venv/bin/python src/data/load_raw_data.py`.

# Project Log

## Phase 1: Project Setup

Date: 2026-07-26

Completed the initial repository setup for the NFL predictor project.

- Created the required project folders for raw data, processed data, notebooks, source code, app code, and documentation.
- Confirmed the project uses Poetry metadata in `pyproject.toml`.
- Updated `.gitignore` so generated data and model artifacts stay out of version control.
- Added starter documentation files for project progress and model results.

## Upcoming Work

- Phase 4: Build matchup and rolling team features.

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

## Phase 3: Dataset Creation

Date: 2026-07-26

Implemented `src/data/make_dataset.py` to create the first cleaned game-level modeling dataset from `data/raw/games.parquet`.

- Output: `data/processed/modeling_dataset.parquet`.
- Target: `home_win`, where `1` means the home team scored more points than the away team and `0` means the away team won.
- Tie handling: removed tied games so the first modeling target stays binary.
- Sorting: ordered games by `season`, `week`, parsed `gameday`, and `game_id`.
- Verified output from the first successful run:
  - Raw schedule rows: 7,017.
  - Removed non-regular-season rows: 298.
  - Removed rows missing final scores: 0.
  - Removed tied games: 15.
  - Final modeling rows: 6,704.
- Run command: `.venv/bin/python src/data/make_dataset.py`.

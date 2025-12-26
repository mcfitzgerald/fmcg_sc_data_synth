# Usage Guide

The generator is controlled via the `generate_data.py` script.

## Basic Usage

Generate the full dataset and write to `postgres/seed.sql`:

```bash
python generate_data.py
```

## Options

### Validation Only
Run generation and validation logic without writing the output file (faster for testing logic changes):

```bash
python generate_data.py --validate-only
```

### Partial Regeneration
If you have modified a specific level (e.g., Level 8 - Demand) and want to regenerate it along with all downstream levels (9-14), use `--from-level`. **Note:** This requires a valid prior run in memory, or it will trigger a full regeneration.

```bash
python generate_data.py --from-level 8
```

### Custom Output
Specify a different output path:

```bash
python generate_data.py --output /tmp/custom_seed.sql
```

### Reproducibility
Set a specific random seed to ensure identical data generation across runs:

```bash
python generate_data.py --seed 12345
```

## Output

The script produces a PostgreSQL `COPY` format SQL file. Load it using `psql`:

```bash
psql -h localhost -U postgres -d fmcg_db -f seed.sql
```

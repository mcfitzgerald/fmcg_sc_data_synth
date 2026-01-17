# Regression Investigation: Service Level Collapse

## Problem Statement

At some point between v0.31.0 and v0.35.5, the simulation achieved all target metrics:
- Service Level: 97.5%
- SLOB: <15%
- Inventory Turns: ~6x
- OEE: ~46%

Current state (v0.35.5, 365-day run):
- Service Level: **80.72%** (target: ≥95%)
- SLOB: **1.8%** (excellent)
- Inventory Turns: **6.06x** (on target)
- OEE: **7.4%** (very low - overcapacity)

The config was restored from pre-v0.35.4 values, but service level is not recovering.

## Key Questions

1. **When did we actually achieve 97.5% service level?** Was it a 30-day run (cold-start artifact) or a true 365-day steady-state result?

2. **What changed in the codebase** between the working version and now? The config was restored, but code changes may have altered behavior.

3. **Why is OEE so low (7.4%)?** With 176 lines and demand at 7.0, we have massive overcapacity. But if capacity is so high, why isn't service level higher?

4. **Inventory drawdown pattern**: Mean inventory dropped from ~2748 to ~1330 over the run. Why is replenishment not keeping up with demand?

## Investigation Tasks

### Phase 1: Historical Validation
```bash
# Check git history for when 97.5% was actually achieved
git log --oneline --all | head -50

# Look for commits mentioning service level
git log --all --grep="service" --oneline

# Check if there are any benchmark results saved
find . -name "*.txt" -path "*/output/*" | xargs grep -l "SERVICE"
```

### Phase 2: Config Archaeology
```bash
# Compare current config to known-good commits
git show fff2f8f:src/prism_sim/config/simulation_config.json > /tmp/v0351_config.json
git show c6e6e1f:src/prism_sim/config/simulation_config.json > /tmp/v0354_config.json
diff /tmp/v0351_config.json src/prism_sim/config/simulation_config.json
```

### Phase 3: Code Diff Analysis
Look for behavioral changes in these files between v0.35.1 and HEAD:
- `simulation/orchestrator.py` - daily loop changes
- `simulation/mrp.py` - production planning logic
- `agents/replenishment.py` - order generation
- `agents/allocation.py` - inventory allocation
- `simulation/demand.py` - demand generation

### Phase 4: Diagnostic Run
```bash
# Run with verbose MRP diagnostics
poetry run python run_simulation.py --days 30 --no-checkpoint 2>&1 | tee diagnostic.log

# Check daily metrics progression
grep "Day " diagnostic.log | head -30
```

### Phase 5: Specific Hypotheses to Test

#### H1: Calibration Script Not Run
The restored config may need recalibration:
```bash
poetry run python scripts/calibrate_config.py --dry-run
```

#### H2: Demand/Capacity Mismatch
Check if demand is actually being served:
- Daily demand: ~7.7M cases
- Daily production: varies 1M-7M cases
- Are we producing enough?

#### H3: Replenishment Signal Broken
Check if replenishment orders are being generated correctly:
- Look at order volumes in logs
- Verify DOS calculations are triggering orders

#### H4: Allocation Bottleneck
Check if inventory exists but isn't being allocated:
- Fair share allocation may be spreading too thin
- ABC prioritization may have changed

## Files to Examine

1. `src/prism_sim/simulation/orchestrator.py` - main loop
2. `src/prism_sim/simulation/mrp.py` - production planning
3. `src/prism_sim/agents/replenishment.py` - order generation
4. `src/prism_sim/agents/allocation.py` - inventory distribution
5. `src/prism_sim/simulation/state.py` - inventory tracking
6. `scripts/calibrate_config.py` - parameter derivation

## Expected Outcome

Identify the specific change (code or config) that caused service level to drop from 97.5% to 80.72%, and propose a fix.

## Context for Fresh Session

Copy this entire file as context, plus:
1. Read `docs/llm_context.md` for architecture overview
2. Read `CLAUDE.md` for project conventions
3. The config has been restored to pre-v0.35.4 values
4. A 365-day simulation was run and produced 80.72% service level
5. The question is: why isn't it 97.5% like it supposedly was before?

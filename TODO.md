Run full 365-day diagnostic to validate v0.45.0 network restructure + echelon/batch fixes

Run all 6 analysis scripts against 365-day output to confirm they work end-to-end

Monitor DC inventory trend (target: stable +-20% vs v0.44.0 +290-360% buildup)

Tune echelon_safety_multiplier (1.0 → 1.1-1.2) if stores starve after echelon IP fix

Update architecture docs and create readme. llm_context has been source of truth, but we might need more detailed explainers and visuals for humans

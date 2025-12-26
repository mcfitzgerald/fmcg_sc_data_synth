# Comparison Report: Current FMCG Implementation vs. Fresh Start Plan

**Date:** December 24, 2025
**Context:** Migration from Sequential Data Generation (`fmcg_example/scripts/data_generation`) to Discrete-Event Simulation (`fresh_start.md`).

## Executive Summary

The "Fresh Start" plan correctly identifies the fundamental architectural flaw of the current system: **Sequential Generation lacks Time-State Continuity.** The current system generates tables one by one (POs -> Production -> Shipments), which necessitates "patching" physics (e.g., fixing mass balance *after* the fact, or ignoring machine capacity). The move to a **Time-Step Simulation (`Sim-Graph`)** is the correct solution to fix "Infinite Capacity" and "Teleportation" issues.

**However**, the current implementation is significantly "richer" than the Fresh Start plan implies. It contains sophisticated logic for volumetrics, behavioral quirks, and specific risk scenarios that must be carefully ported. **The risk is not in the architecture, but in the loss of these tuned business rules.**

---

## 1. Architecture & State Management

| Feature | Current Implementation (`fmcg_example`) | Fresh Start Plan (`Sim-Graph`) | Verdict |
| :--- | :--- | :--- | :--- |
| **Time Model** | **Sequential:** Generates full history for Table A, then Table B. | **Iterative:** `for day in range(365): update_state()` | **Fresh Start is Critical.** The current approach makes feedback loops (Plan -> Source) impossible to model causally. |
| **Capacity** | **Infinite/Random:** `Level5Generator` picks random dates/lines. No check for overlap. | **Finite:** "Gantt Scheduler" explicitly manages `Line_Availability_Vector`. | **Fresh Start is Critical.** This is the main "Potemkin" fix. |
| **Mass Balance** | **Patchwork:** `Level10` attempts to balance supply/demand post-hoc. | **Intrinsic:** `Inventory[t] = Inventory[t-1] + In - Out`. | **Fresh Start is Critical.** Eliminates "Teleportation". |
| **Performance** | **Vectorized Generation:** Uses `numpy` for batch creation. | **Vectorized State:** Uses Tensors for world state. | **Parity.** Both rely on vectorization; new plan just changes *what* is vectorized (State vs. Rows). |

---

## 2. Physics & "Richness" to Preserve

The following features exist in the current codebase and are **at risk** of being lost if the new plan relies only on generic "physics engines."

### A. Behavioral Quirks (`quirks.py`)
The current system implements specific, tuned pathologies that the new "Chaos Monkey" must replicate:

*   **Autoregressive Port Congestion:** Not just random delays, but an **AR(1) model** (`current_delay = 0.7 * prev_delay + noise`). This creates realistic "clumping" of late shipments.
*   **Bullwhip "Whip Crack":** Specific logic to batch small orders into massive ones during promos, amplifying variance.
*   **Human Optimism Bias:** A dedicated logic to over-forecast *new* products by 15% (`apply_optimism_bias`), simulating planner psychology.
*   **Phantom Inventory:** Simulates shrinkage with a **detection lag** (e.g., 14 days) before the system "realizes" the stock is gone.

**Action Item:** The `Sim-Graph` loop needs a `QuirkLayer` that modifies the state tensors (e.g., adding noise to the `Demand` tensor for Bullwhip) *before* the physics engines process them.

### B. Logistics Physics (`level_10_11_fulfillment.py`)
The Fresh Start plan claims the current system has "Cartoon Volumetrics," but the code shows otherwise:
*   **Current Reality:** It *does* use `sku_weight_kg` and `TRUCK_CAPACITY_KG` (20,000kg). It mixes "Dense" and "Bulky" SKUs implicitly.
*   **Fresh Start Upgrade:** The "Tetris Engine" (checking Cube *and* Weight limits explicitly) is an upgrade, but the current system is not zero-fidelity.
*   **Missing in Fresh Start:** The current system has specific **Pallet Cost Tiers** (LTL vs. FTL pricing logic in `benchmark_manifest.json`) and **Carbon logic** (`RSK-ENV-005`). These need to be part of the new `Fulfill_Engine`.

### C. Promotional Logic (`promo_calendar.py`)
*   **Current Reality:** A sophisticated `PromoCalendar` handles **Overlaps** (Max-Lift wins) and **Hangovers** (post-promo dips).
*   **Fresh Start Risk:** The new `Order_Engine` must not simplify this to just "Random Demand Spikes." It needs to ingest the specific 52-week promo plan.

### D. Named Scenarios & Validation (`benchmark_manifest.json`)
The simulation must converge to these specific "Named Entities":
*   **`B-2024-RECALL-001`:** A specific contaminated batch of Sorbitol. The `Transform_Engine` must ensure this exact batch ID is created and flagged.
*   **`DC-NAM-CHI-001`:** The Chicago Hub. The topology must ensure 40% of flow goes here to stress-test the bottleneck.
*   **`SUP-MY-PALM`:** The Single Point of Failure. The `Source_Engine` needs to model the specific lead times (Ocean) for this node.

---

## 3. Implementation Recommendations

1.  **Port, Don't Rewrite, the Config:** The `benchmark_manifest.json` is a gold standard. The new simulation should load this exact file to configure its physics engines (e.g., loading `yield_loss_rate_std` into the `Transform_Engine`).
2.  **Explicit Quirk Handlers:** In the `Sim-Graph` loop:
    ```python
    # 1. Chaos (Global events like Port Strikes)
    chaos_monkey.apply(state, day) 
    
    # 2. Quirks (Behavioral modifiers)
    # MUST PORT: quirk.apply_optimism_bias(state.forecasts)
    quirk_manager.apply(state, day)
    
    # 3. Physics Engines...
    ```
3.  **Hybrid Validation:** Keep the existing `validation.py` scripts. Since the simulation output (tables) will look the same, the validation suite should run unmodified against the new output to prove "Kinetic Fidelity" improved without breaking "Statistical Realism."

## Conclusion

The **Fresh Start** plan is architecturally sound and necessary to fix the core "Infinite Capacity" and "Causality" bugs. However, you must meticulously port the **`quirks.py`** and **`promo_calendar.py`** logic into the new engine's "Pre-Physics" phase, or the simulation will be physically accurate but behaviorally sterile.

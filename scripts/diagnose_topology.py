import numpy as np

from prism_sim.network.core import NodeType
from prism_sim.simulation.orchestrator import Orchestrator


def diagnose() -> None:
    sim = Orchestrator(enable_logging=False)

    print("--- Topology Diagnosis ---")
    rdc_ids = [
        n_id
        for n_id, n in sim.world.nodes.items()
        if n.type == NodeType.DC and n_id.startswith("RDC-")
    ]
    customer_dc_ids = [
        n_id
        for n_id, n in sim.world.nodes.items()
        if n.type == NodeType.DC and not n_id.startswith("RDC-")
    ]

    print(f"RDCs: {len(rdc_ids)}")
    print(f"Customer DCs: {len(customer_dc_ids)}")

    # Check RDC -> DC links
    rdc_downstream_dc: dict[str, list[str]] = {rdc: [] for rdc in rdc_ids}
    rdc_downstream_store: dict[str, list[str]] = {rdc: [] for rdc in rdc_ids}
    dc_parents: dict[str, list[str]] = {dc: [] for dc in customer_dc_ids}

    for link in sim.world.links.values():
        if link.source_id in rdc_ids:
            if link.target_id in customer_dc_ids:
                rdc_downstream_dc[link.source_id].append(link.target_id)
                dc_parents[link.target_id].append(link.source_id)
            elif sim.world.nodes[link.target_id].type == NodeType.STORE:
                rdc_downstream_store[link.source_id].append(link.target_id)

    for rdc in rdc_ids:
        dcs = len(rdc_downstream_dc[rdc])
        stores = len(rdc_downstream_store[rdc])
        print(f"{rdc} has {dcs} downstream DCs and {stores} downstream Stores")

    orphaned_dcs = [dc for dc, parents in dc_parents.items() if not parents]
    if orphaned_dcs:
        print(
            f"WARNING: {len(orphaned_dcs)} Orphaned Customer DCs "
            f"(No RDC parent): {orphaned_dcs[:5]}..."
        )
    else:
        print("All Customer DCs have parents.")

    print("\n--- Inventory Priming Diagnosis ---")
    # Check inventory at Day 0 (before run)
    state = sim.state

    print("RDC Inventory:")
    for rdc in rdc_ids:
        idx = state.node_id_to_idx.get(rdc)
        if idx is not None:
            inv = np.sum(state.actual_inventory[idx])
            print(f"  {rdc}: {inv:,.0f}")

    print("Customer DC Inventory (Sample):")
    for dc in customer_dc_ids[:5]:
        idx = state.node_id_to_idx.get(dc)
        if idx is not None:
            inv = np.sum(state.actual_inventory[idx])
            print(f"  {dc}: {inv:,.0f}")

    print("\n--- Push Logic Check ---")
    # Simulate one push step manually
    print("Simulating Push Step...")
    push_shipments = sim._push_excess_rdc_inventory(day=1)
    print(f"Generated {len(push_shipments)} push shipments")

    total_push_qty = sum(
        sum(line.quantity for line in s.lines) for s in push_shipments
    )
    print(f"Total Pushed Quantity: {total_push_qty:,.0f}")


if __name__ == "__main__":
    diagnose()

#!/usr/bin/env python3
"""
Generate Route Simulation GIF
==============================

Creates an animated GIF showing the cart moving through the optimized route.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from cad_adapter.layout_importer import LayoutImporter
from optimizer.batch import Batch, PickItem, Cart
from optimizer.route_optimizer import RouteOptimizer
from visualization.route_simulation import RouteSimulator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

print("\n" + "="*60)
print("WAREHOUSE ROUTE SIMULATION")
print("="*60 + "\n")

# 1. Load warehouse
print("1. Loading warehouse from DXF...")
warehouse = LayoutImporter().import_from_file('Sample_warehouse_01.dxf')
print(f"   ✓ {len(warehouse.nodes)} locations, {len(warehouse.edges)} paths\n")

# 2. Create order
print("2. Creating order...")
batch = Batch(id="ORDER_001", depot_id="DEPOT")
batch.add_item(PickItem(sku="Snacks-ChipsA", location_id="A02-R-02", quantity=2, weight=0.5))
batch.add_item(PickItem(sku="Beverages-Juice", location_id="A02-L-04", quantity=3, weight=1.2))
batch.add_item(PickItem(sku="Beverages-Juice", location_id="A02-R-04", quantity=3, weight=1.2))
batch.add_item(PickItem(sku="Beverages-Juice", location_id="A01-L-03", quantity=3, weight=1.2))
batch.add_item(PickItem(sku="Beverages-Juice", location_id="A01-R-04", quantity=3, weight=1.2))
batch.add_item(PickItem(sku="Beverages-Juice", location_id="A01-L-04", quantity=3, weight=1.2))

locations = [l for l in batch.get_required_locations() if l != 'DEPOT']
print(f"   ✓ {batch.total_items()} items ({batch.total_weight():.1f}kg) from {len(locations)} locations")
print(f"   ✓ Locations: {', '.join(locations)}\n")

# 3. Assign cart
print("3. Assigning cart...")
cart = Cart(id="CART_001", capacity=1000, weight_capacity=1000.0)
print(f"   ✓ {cart.id} (capacity: {cart.capacity}, weight capacity: {cart.weight_capacity}kg)\n")

# 4. Optimize
print("4. Finding optimal route...")
optimizer = RouteOptimizer(warehouse, time_limit=30)
routes = optimizer.optimize([batch], [cart])

if routes:
    route = routes[list(routes.keys())[0]]
    route_sequence = route.get_node_sequence()
    print(f"   ✓ Route: {' → '.join(route_sequence)}")
    print(f"   ✓ Time: {route.total_time:.2f} units\n")

    # 5. Create simulation
    print("5. Creating animated simulation...")
    import os
    os.makedirs("output", exist_ok=True)

    pick_locations = [item.location_id for item in batch.items]

    simulator = RouteSimulator(
        graph=warehouse,
        route_sequence=route_sequence,
        cart_id=cart.id,
        batch_id=batch.id
    )

    simulator.simulate_route(
        output_path="output/simulation.gif",
        pick_locations=pick_locations,
        duration=15  # 15 second animation
    )

    print(f"   ✓ Saved: output/simulation.gif\n")

    print("="*60)
    print("SIMULATION COMPLETE!")
    print("="*60)
else:
    print("   ✗ No solution found")

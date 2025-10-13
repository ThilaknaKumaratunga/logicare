#!/usr/bin/env python3
"""
Warehouse Route Optimization & Simulation
==========================================

Complete workflow:
1. Load warehouse from DXF
2. Create order and assign cart
3. Optimize route
4. Save static route visualization
5. Create animated simulation
"""

import sys
import logging
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from cad_adapter.layout_importer import LayoutImporter
from optimizer.batch import Batch, PickItem, Cart
from optimizer.route_optimizer import RouteOptimizer
from visualization.route_visualizer import RouteVisualizer
from visualization.route_simulation import RouteSimulator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

print("\n" + "="*60)
print("WAREHOUSE ROUTE OPTIMIZATION & SIMULATION")
print("="*60 + "\n")

# Configuration
DXF_FILE = 'Sample_warehouse_01.dxf'
ANIMATION_DURATION = 15  # seconds
ANIMATION_FPS = 30

# Create output directory
os.makedirs("output", exist_ok=True)

# 1. Load warehouse from DXF
print("1. Loading warehouse from DXF...")
warehouse = LayoutImporter().import_from_file(DXF_FILE)
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
batch.add_item(PickItem(sku="Snacks-ChipsA", location_id="A01-R-02", quantity=2, weight=0.5))
batch.add_item(PickItem(sku="Snacks-ChipsA", location_id="A01-R-02", quantity=2, weight=0.5))
batch.add_item(PickItem(sku="Snacks-ChipsA", location_id="A02-L-02", quantity=2, weight=0.5))
locations = [l for l in batch.get_required_locations() if l != 'DEPOT']
print(f"   ✓ {batch.total_items()} items ({batch.total_weight():.1f}kg) from {len(locations)} locations")
print(f"   ✓ Locations: {', '.join(locations)}\n")

# 3. Assign cart
print("3. Assigning cart...")
cart = Cart(id="CART_001", capacity=1000, weight_capacity=1000.0)
print(f"   ✓ {cart.id} (capacity: {cart.capacity}, weight capacity: {cart.weight_capacity}kg)\n")

# 4. Optimize route
print("4. Finding optimal route...")
optimizer = RouteOptimizer(warehouse, time_limit=30)
routes = optimizer.optimize([batch], [cart])

if routes:
    route = routes[list(routes.keys())[0]]
    route_sequence = route.get_node_sequence()
    print(f"   ✓ Route: {' → '.join(route_sequence)}")
    print(f"   ✓ Distance: {route.total_time:.2f} units\n")

    pick_locations = [item.location_id for item in batch.items]

    cart_info = {
        'capacity': cart.capacity,
        'items': batch.total_items(),
        'weight': batch.total_weight()
    }

    # 5. Save static route visualization
    print("5. Creating static route visualization...")
    viz = RouteVisualizer(warehouse, dxf_file=DXF_FILE)
    viz.plot_route(route_sequence, batch.id, cart.id, cart_info=cart_info, pick_locations=pick_locations)
    viz.save_plot("output/route.png")
    print(f"   ✓ Saved: output/route.png\n")

    # 6. Create animated simulation
    print("6. Creating animated simulation...")
    simulator = RouteSimulator(
        graph=warehouse,
        dxf_file=DXF_FILE,
        route_sequence=route_sequence,
        cart_id=cart.id,
        batch_id=batch.id
    )

    simulator.simulate_route(
        output_path="output/simulation.gif",
        pick_locations=pick_locations,
        duration=ANIMATION_DURATION,
        fps=ANIMATION_FPS
    )

    print(f"   ✓ Animation saved: output/simulation.gif")
    print(f"   ✓ Duration: {ANIMATION_DURATION}s @ {ANIMATION_FPS}fps\n")

    print("="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"📊 Static route: output/route.png")
    print(f"🎬 Animation: output/simulation.gif")
    print("="*60)
else:
    print("   ✗ No solution found")

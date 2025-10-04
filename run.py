#!/usr/bin/env python3
"""
Simple Warehouse Route Optimization
====================================

One warehouse + One batch + One cart = One optimal route
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from cad_adapter.layout_importer import LayoutImporter
from optimizer.batch import Batch, PickItem, Cart
from optimizer.route_optimizer import RouteOptimizer
from visualization.route_visualizer import RouteVisualizer

# Setup simple logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

print("\n" + "="*60)
print("WAREHOUSE ROUTE OPTIMIZATION")
print("="*60 + "\n")

# 1. Load warehouse
print("1. Loading warehouse...")
warehouse = LayoutImporter().load_from_json('data/warehouse_grid_25x4.json')
print(f"   ✓ {len(warehouse.nodes)} locations, {len(warehouse.edges)} paths\n")

# 2. Create order
print("2. Creating order...")
batch = Batch(id="ORDER_001", depot_id="DEPOT")

# Add 100 items distributed across aisles A01-A100
import random
random.seed(42)  # For reproducibility

for i in range(1, 101):
    aisle = random.randint(1, 100)
    side = random.choice(['L', 'R'])
    shelf = random.randint(0, 4)
    location_id = f"A{aisle:02d}-{side}-{shelf:02d}"

    batch.add_item(PickItem(
        sku=f"Item-{i:03d}",
        location_id=location_id,
        quantity=random.randint(1, 3),
        weight=round(random.uniform(0.3, 2.0), 1)
    ))

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
    print(f"   ✓ Route: {' → '.join(route.get_node_sequence())}")
    print(f"   ✓ Time: {route.total_time:.2f} units\n")

    # 5. Visualize
    print("5. Creating visualization...")
    import os
    os.makedirs("output", exist_ok=True)

    # Prepare cart information for visualization
    cart_info = {
        'capacity': cart.capacity,
        'items': batch.total_items(),
        'weight': batch.total_weight()
    }

    pick_locations = [item.location_id for item in batch.items]

    viz = RouteVisualizer(warehouse)
    viz.plot_route(route.get_node_sequence(), batch.id, cart.id, cart_info=cart_info, pick_locations=pick_locations)
    viz.save_plot("output/route.svg")
    print(f"   ✓ Saved: output/route.svg\n")

    print("="*60)
    print("COMPLETE!")
    print("="*60)


    #viz.show()
else:
    print("   ✗ No solution found")

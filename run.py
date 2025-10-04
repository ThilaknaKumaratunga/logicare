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
warehouse = LayoutImporter().load_from_json('data/warehouse_crossaisle.json')
print(f"   ✓ {len(warehouse.nodes)} locations, {len(warehouse.edges)} paths\n")

# 2. Create order
print("2. Creating order...")
batch = Batch(id="ORDER_001", depot_id="DEPOT")
batch.add_item(PickItem(sku="Snacks-ChipsA", location_id="A02-R-08", quantity=2, weight=0.5))
batch.add_item(PickItem(sku="Beverages-Juice", location_id="A05-L-14", quantity=3, weight=1.2))
batch.add_item(PickItem(sku="Dairy-Milk", location_id="A01-L-03", quantity=1, weight=1.0))
batch.add_item(PickItem(sku="Bakery-Bread", location_id="A08-R-15", quantity=2, weight=0.8))
batch.add_item(PickItem(sku="Produce-Apples", location_id="A02-L-02", quantity=4, weight=1.5))
batch.add_item(PickItem(sku="Frozen-IceCream", location_id="A10-R-18", quantity=1, weight=2.0))
batch.add_item(PickItem(sku="Canned-Beans", location_id="A06-R-11", quantity=3, weight=1.3))
batch.add_item(PickItem(sku="Cereal-Oats", location_id="A03-L-09", quantity=2, weight=0.9))
batch.add_item(PickItem(sku="Pasta-Spaghetti", location_id="A07-R-07", quantity=2, weight=1.1))
batch.add_item(PickItem(sku="Condiments-Ketchup", location_id="A04-L-05", quantity=1, weight=0.6))
batch.add_item(PickItem(sku="Snacks-Cookies", location_id="A09-R-12", quantity=3, weight=0.7))
batch.add_item(PickItem(sku="Beverages-Soda", location_id="A05-L-08", quantity=4, weight=2.5))
batch.add_item(PickItem(sku="Cleaning-Detergent", location_id="A01-L-06", quantity=1, weight=1.8))
batch.add_item(PickItem(sku="PersonalCare-Shampoo", location_id="A06-R-03", quantity=2, weight=0.8))
batch.add_item(PickItem(sku="Meat-Chicken", location_id="A10-R-02", quantity=1, weight=2.2))
batch.add_item(PickItem(sku="Bakery-Rolls", location_id="A08-R-16", quantity=2, weight=0.5))
batch.add_item(PickItem(sku="Produce-Bananas", location_id="A03-L-07", quantity=3, weight=1.2))
batch.add_item(PickItem(sku="Dairy-Yogurt", location_id="A01-L-04", quantity=2, weight=1.0))
batch.add_item(PickItem(sku="Frozen-Pizza", location_id="A09-R-19", quantity=1, weight=1.5))
batch.add_item(PickItem(sku="Snacks-Nuts", location_id="A07-R-05", quantity=2, weight=0.9))
batch.add_item(PickItem(sku="Canned-Soup", location_id="A04-L-10", quantity=2, weight=1.1))
batch.add_item(PickItem(sku="Rice-LongGrain", location_id="A02-R-13", quantity=3, weight=2.0))
batch.add_item(PickItem(sku="Oil-Olive", location_id="A05-L-06", quantity=1, weight=1.3))
batch.add_item(PickItem(sku="Spices-Pepper", location_id="A08-R-04", quantity=2, weight=0.3))
batch.add_item(PickItem(sku="Tea-GreenTea", location_id="A03-L-11", quantity=3, weight=0.5))
batch.add_item(PickItem(sku="Coffee-Ground", location_id="A06-R-09", quantity=2, weight=0.9))
batch.add_item(PickItem(sku="Candy-Chocolate", location_id="A09-R-06", quantity=4, weight=0.6))
batch.add_item(PickItem(sku="PetFood-DogFood", location_id="A01-L-17", quantity=1, weight=3.0))
batch.add_item(PickItem(sku="BabyFood-Puree", location_id="A10-R-10", quantity=2, weight=0.7))
batch.add_item(PickItem(sku="Batteries-AA", location_id="A07-R-14", quantity=3, weight=0.4))
batch.add_item(PickItem(sku="Paper-Towels", location_id="A04-L-12", quantity=2, weight=1.2))
batch.add_item(PickItem(sku="Toilet-Paper", location_id="A02-R-17", quantity=3, weight=1.5))
batch.add_item(PickItem(sku="Soap-BarSoap", location_id="A05-L-19", quantity=2, weight=0.6))
batch.add_item(PickItem(sku="Toothpaste-Mint", location_id="A08-R-08", quantity=1, weight=0.3))
batch.add_item(PickItem(sku="Diapers-M", location_id="A03-L-15", quantity=1, weight=2.5))
batch.add_item(PickItem(sku="Vitamins-C", location_id="A06-R-16", quantity=2, weight=0.4))
batch.add_item(PickItem(sku="FirstAid-Bandages", location_id="A09-R-11", quantity=3, weight=0.2))
batch.add_item(PickItem(sku="Sauces-Tomato", location_id="A01-L-13", quantity=2, weight=1.0))
batch.add_item(PickItem(sku="Vinegar-White", location_id="A10-R-07", quantity=1, weight=1.1))
batch.add_item(PickItem(sku="Sugar-White", location_id="A07-R-18", quantity=2, weight=1.8))
batch.add_item(PickItem(sku="Flour-AllPurpose", location_id="A04-L-09", quantity=1, weight=2.2))
batch.add_item(PickItem(sku="Eggs-Dozen", location_id="A02-R-03", quantity=2, weight=1.4))
batch.add_item(PickItem(sku="Butter-Salted", location_id="A05-L-16", quantity=1, weight=0.9))
batch.add_item(PickItem(sku="Cheese-Cheddar", location_id="A08-R-19", quantity=2, weight=1.3))
batch.add_item(PickItem(sku="Fish-Salmon", location_id="A03-L-05", quantity=1, weight=1.9))
batch.add_item(PickItem(sku="Vegetables-Carrots", location_id="A06-R-12", quantity=3, weight=1.1))
batch.add_item(PickItem(sku="Fruit-Oranges", location_id="A09-R-14", quantity=2, weight=1.4))
batch.add_item(PickItem(sku="Nuts-Almonds", location_id="A01-L-08", quantity=2, weight=0.8))
batch.add_item(PickItem(sku="Juice-Orange", location_id="A10-R-15", quantity=3, weight=1.6))
batch.add_item(PickItem(sku="Water-Bottled", location_id="A07-R-10", quantity=4, weight=2.8))

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
    viz.save_plot("output/route.png")
    print(f"   ✓ Saved: output/route.png\n")

    print("="*60)
    print("COMPLETE!")
    print("="*60)


    #viz.show()
else:
    print("   ✗ No solution found")

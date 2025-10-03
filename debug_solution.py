#!/usr/bin/env python3
import logging
logging.basicConfig(level=logging.WARNING)

from cad_adapter.layout_importer import LayoutImporter
from optimizer.batch import Batch, PickItem, Cart
from optimizer.route_optimizer import RouteOptimizer

warehouse = LayoutImporter().load_from_json('data/test_warehouse_small.json')
batch = Batch(id='ORDER_001', depot_id='DEPOT')
batch.add_item(PickItem(sku='test', location_id='A03-05-00', quantity=1, weight=0.5))
cart = Cart(id='CART_001', capacity=100)

optimizer = RouteOptimizer(warehouse, time_limit=30)
routes = optimizer.optimize([batch], [cart])

# Debug: check actual variable values
if optimizer.model and optimizer.model.solution:
    sol = optimizer.model.solution

    print("\n=== DEBUGGING SOLUTION ===\n")

    # Check visit variables
    print("Visit variables:")
    for node_id in ['DEPOT', 'A01-01-00', 'A03-05-00']:
        var_name = f'visit_{node_id}_{batch.id}_{cart.id}'
        try:
            var = optimizer.model.get_var_by_name(var_name)
            value = sol.get_value(var)
            print(f"  {var_name} = {value}")
        except:
            print(f"  {var_name} = NOT FOUND")

    # Check route variables (edges used)
    print("\nRoute variables (active edges):")
    for var in optimizer.model.iter_binary_vars():
        if var.name.startswith('r_') and sol.get_value(var) > 0.5:
            print(f"  {var.name} = 1")

    print("\n=== END DEBUG ===\n")

#!/usr/bin/env python3
"""
Test script to visualize the 20x5 grid warehouse layout.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from cad_adapter.layout_importer import LayoutImporter
from visualization.route_visualizer import RouteVisualizer

print("\n" + "="*60)
print("TESTING 20x5 GRID WAREHOUSE LAYOUT")
print("="*60 + "\n")

# Load warehouse
print("Loading warehouse...")
warehouse = LayoutImporter().load_from_json('data/warehouse_grid_20x5.json')
print(f"✓ {len(warehouse.nodes)} locations, {len(warehouse.edges)} paths")
print(f"✓ Grid: 20 columns × 5 rows = 100 aisles")
print(f"✓ Each aisle: 2 sides × 20 blocks = 40 storage locations")
print(f"✓ Total storage locations: 4000\n")

# Visualize layout
print("Creating visualization...")
viz = RouteVisualizer(warehouse)
viz.plot_layout(figsize=(60, 110), show_edge_labels=False, title="20×5 Grid Warehouse Layout (100 Aisles)")
viz.save_plot("output/warehouse_grid_20x5.png")
print("✓ Saved: output/warehouse_grid_20x5.png\n")

print("="*60)
print("VISUALIZATION COMPLETE!")
print("="*60)

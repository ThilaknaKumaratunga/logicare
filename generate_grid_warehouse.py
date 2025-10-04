#!/usr/bin/env python3
"""
Generate a 20x5 grid warehouse layout with 100 aisles.
Each aisle has 2 sides (L and R) with 20 blocks per side.
"""

import json

def generate_grid_warehouse():
    """
    Generate warehouse with aisles arranged in a 20x5 grid.

    Grid layout (20 columns x 5 rows):
    - 20 aisles per row
    - 5 rows of aisles
    - Each aisle has 2 sides (L and R) with 20 blocks each
    - Aisles numbered A001-A100

    Coordinate system:
    - Each aisle occupies 3 x-units (L side, passage, R side)
    - Each row of aisles occupies 22 y-units (20 blocks + 2 cross-aisles)
    - Horizontal passages at y=1 and y=20 within each row
    - Vertical passages between columns at x positions
    """

    warehouse = {
        "warehouse": {
            "name": "Grid Warehouse 20x5",
            "description": "100 aisles in 20x5 grid, each with 2 sides and 20 blocks per side",
            "dimensions": {
                "grid_rows": 5,
                "grid_cols": 20,
                "total_aisles": 100,
                "blocks_per_side": 20,
                "sides_per_aisle": 2
            },
            "passages": {
                "horizontal": [],  # Will be computed per row
                "description": "Cross-aisle passages at bottom (y=1,23,45,67,89) and top (y=20,42,64,86,108)"
            },
            "start": {
                "x": 1,
                "y": 0,
                "z": 0,
                "id": "DEPOT"
            },
            "locations": []
        }
    }

    # Calculate horizontal passage y-coordinates for all rows
    horizontal_passages = []
    for row in range(5):
        base_y = row * 22  # Each row is 22 units tall (20 blocks + 2 passages)
        horizontal_passages.extend([base_y + 1, base_y + 20])

    warehouse["warehouse"]["passages"]["horizontal"] = horizontal_passages

    aisle_num = 1

    for row in range(5):  # 5 rows
        for col in range(20):  # 20 columns
            aisle_id = f"A{aisle_num:03d}"

            # Calculate base coordinates for this aisle
            # Each aisle is 3 units wide (0: L, 1: passage, 2: R)
            base_x = col * 3
            base_y = row * 22  # Each row separated by 22 units

            aisle_data = {
                "aisle": aisle_id,
                "sides": {
                    "L": [],
                    "R": []
                }
            }

            # Generate 20 blocks for left side
            for block in range(1, 21):
                aisle_data["sides"]["L"].append({
                    "x": base_x,
                    "y": base_y + block,
                    "z": 0,
                    "id": f"{aisle_id}-L-{block:02d}"
                })

            # Generate 20 blocks for right side
            for block in range(1, 21):
                aisle_data["sides"]["R"].append({
                    "x": base_x + 2,
                    "y": base_y + block,
                    "z": 0,
                    "id": f"{aisle_id}-R-{block:02d}"
                })

            warehouse["warehouse"]["locations"].append(aisle_data)
            aisle_num += 1

    return warehouse

if __name__ == "__main__":
    warehouse = generate_grid_warehouse()

    # Save to JSON file
    output_file = "data/warehouse_grid_20x5.json"
    with open(output_file, 'w') as f:
        json.dump(warehouse, f, indent=2)

    print(f"✓ Generated 20x5 grid warehouse")
    print(f"✓ Total aisles: {warehouse['warehouse']['dimensions']['total_aisles']}")
    print(f"✓ Total locations: {100 * 2 * 20} (100 aisles × 2 sides × 20 blocks)")
    print(f"✓ Saved to: {output_file}")

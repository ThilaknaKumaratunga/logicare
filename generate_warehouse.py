#!/usr/bin/env python3
"""
Generate a grid-based warehouse layout with 25x4 (100 aisles total)
Each aisle has L and R sides with 4 storage blocks per side
"""

import json
from pathlib import Path


def generate_warehouse_layout(rows=25, cols=4, blocks_per_side=4):
    """
    Generate a grid-based warehouse layout.

    Args:
        rows: Number of aisle rows (default 25)
        cols: Number of aisle columns (default 4)
        blocks_per_side: Number of storage blocks per aisle side (default 4)

    Returns:
        Dictionary containing warehouse layout
    """

    # Configuration
    block_size = 4  # Each block is 4x4 units
    aisle_spacing = 3  # Space between aisle R and L sides
    passage_width = 2  # Width of passages
    row_spacing = block_size * blocks_per_side  # Vertical space per row
    col_spacing = block_size * 2 + aisle_spacing + passage_width  # Horizontal space per column

    # Calculate depot position (bottom left)
    depot_x = 2
    depot_y = 1
    depot_width = 4
    depot_height = 2

    # Track passage centerlines
    horizontal_passages = []
    vertical_passages = []

    # Bottom horizontal passage (connects to depot)
    horizontal_passages.append({
        "y": depot_y,
        "x_start": 0,
        "x_end": depot_x + col_spacing * cols + passage_width,
        "description": "Bottom passage"
    })

    # Top horizontal passage
    top_y = depot_y + depot_height + row_spacing * rows + passage_width
    horizontal_passages.append({
        "y": top_y,
        "x_start": 0,
        "x_end": depot_x + col_spacing * cols + passage_width,
        "description": "Top passage"
    })

    # Generate vertical passages (one before each column and one at the end)
    for col in range(cols + 1):
        vp_x = depot_x + passage_width + col * col_spacing
        if col == 0:
            vp_x = 1  # Left edge
        vertical_passages.append({
            "x": vp_x,
            "y_start": 0,
            "y_end": top_y + passage_width,
            "description": f"Vertical passage at column {col}"
        })

    # Generate aisles
    aisles = []
    aisle_num = 1

    for row in range(rows):
        for col in range(cols):
            aisle_id = f"A{aisle_num:02d}"

            # Calculate base coordinates for this aisle
            base_x = depot_x + depot_width + col * col_spacing
            base_y = depot_y + depot_height + passage_width + row * row_spacing

            # R side (right side of passage)
            r_x = base_x
            r_blocks = []
            for block_idx in range(blocks_per_side):
                block_y = base_y + block_idx * block_size + block_size / 2
                r_blocks.append({
                    "id": f"{aisle_id}-R-{block_idx+1:02d}",
                    "x": r_x,
                    "y": block_y
                })

            # L side (left side of passage, across the aisle)
            l_x = base_x + block_size + aisle_spacing
            l_blocks = []
            for block_idx in range(blocks_per_side):
                block_y = base_y + block_idx * block_size + block_size / 2
                l_blocks.append({
                    "id": f"{aisle_id}-L-{block_idx+1:02d}",
                    "x": l_x,
                    "y": block_y
                })

            # Add aisle to list
            aisles.append({
                "id": aisle_id,
                "row": row,
                "col": col,
                "sides": {
                    "R": {
                        "x_center": r_x,
                        "blocks": r_blocks
                    },
                    "L": {
                        "x_center": l_x,
                        "blocks": l_blocks
                    }
                }
            })

            aisle_num += 1

    # Create warehouse structure
    warehouse = {
        "warehouse": {
            "name": f"Grid Warehouse {rows}x{cols} ({rows * cols} aisles)",
            "description": f"{rows}x{cols} grid layout with {blocks_per_side} blocks per aisle side",
            "layout": {
                "rows": rows,
                "cols": cols,
                "blocks_per_side": blocks_per_side,
                "total_aisles": rows * cols
            },
            "block_size": {
                "width": block_size,
                "height": block_size,
                "description": f"Each storage block is {block_size}x{block_size} units"
            },
            "aisle_spacing": aisle_spacing,
            "passage_width": passage_width,
            "depot": {
                "x": depot_x,
                "y": depot_y,
                "width": depot_width,
                "height": depot_height,
                "id": "DEPOT",
                "description": f"Depot at ({depot_x},{depot_y})"
            },
            "passages": {
                "horizontal": horizontal_passages,
                "vertical": vertical_passages
            },
            "aisles": aisles
        }
    }

    return warehouse


def main():
    """Generate and save warehouse layout."""
    print("Generating 25x4 (100 aisles) warehouse layout...")

    # Generate layout
    warehouse = generate_warehouse_layout(rows=25, cols=4, blocks_per_side=4)

    # Create output directory
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)

    # Save to JSON
    output_file = output_dir / "warehouse_grid_25x4.json"
    with open(output_file, 'w') as f:
        json.dump(warehouse, f, indent=2)

    print(f"✓ Generated warehouse with {warehouse['warehouse']['layout']['total_aisles']} aisles")
    print(f"✓ Each aisle has L and R sides with {warehouse['warehouse']['layout']['blocks_per_side']} blocks each")
    print(f"✓ Total storage blocks: {warehouse['warehouse']['layout']['total_aisles'] * 2 * warehouse['warehouse']['layout']['blocks_per_side']}")
    print(f"✓ Saved to: {output_file}")

    return warehouse


if __name__ == "__main__":
    main()

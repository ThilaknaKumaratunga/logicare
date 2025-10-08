"""
Warehouse Graph Builder

This script reads a DXF file and creates a NetworkX graph representing:
- Passage lines with Manhattan distance weights
- Aisle polygons (centroids) connected to touching passages
- Depot polygon (centroid) connected to touching passages

Usage:
    python build_warehouse_graph.py <dxf_file>

    Or import as module:
    from build_warehouse_graph import build_warehouse_graph
    G = build_warehouse_graph('warehouse.dxf')
"""

import ezdxf
import networkx as nx
import re
import sys


def point_on_segment(point, seg_start, seg_end, tol=0.01):
    """Check if a point lies on a line segment."""
    x, y = point
    x1, y1 = seg_start
    x2, y2 = seg_end
    if abs(x1 - x2) < tol:  # Vertical
        return abs(x - x1) < tol and min(y1, y2) <= y <= max(y1, y2)
    elif abs(y1 - y2) < tol:  # Horizontal
        return abs(y - y1) < tol and min(x1, x2) <= x <= max(x1, x2)
    return False


def build_warehouse_graph(dxf_file):
    """
    Build a warehouse graph from a DXF file.

    Args:
        dxf_file (str): Path to the DXF file

    Returns:
        networkx.Graph: Graph with nodes and weighted edges
            - Passage nodes have attribute 'pos' (x, y coordinates)
            - Aisle nodes have attributes 'pos' and 'aisle' (layer name)
            - Depot node has attributes 'pos' and 'depot' (True)
            - Edges have attribute 'weight' (Manhattan distance, 0 for connections)
    """
    # Load DXF
    doc = ezdxf.readfile(dxf_file)
    msp = doc.modelspace()

    # 1. Extract passage lines and build graph
    lines = []
    for entity in msp:
        if entity.dxf.layer.lower() == 'passages' and entity.dxftype() == 'LINE':
            start = (round(entity.dxf.start.x, 2), round(entity.dxf.start.y, 2))
            end = (round(entity.dxf.end.x, 2), round(entity.dxf.end.y, 2))
            lines.append((start, end))

    # Collect all unique points
    all_points = set()
    for start, end in lines:
        all_points.add(start)
        all_points.add(end)

    # Create graph
    G = nx.Graph()
    for pt in all_points:
        G.add_node(pt, pos=pt)

    # Create edges between consecutive points on each line
    edges_created = set()
    for line_start, line_end in lines:
        points_on_line = [line_start, line_end]
        for point in all_points:
            if point != line_start and point != line_end:
                if point_on_segment(point, line_start, line_end):
                    points_on_line.append(point)

        # Sort points along the line
        points_on_line.sort(key=lambda p: p[1] if abs(line_start[0] - line_end[0]) < 0.01 else p[0])

        # Create edges between consecutive points
        for i in range(len(points_on_line) - 1):
            p1, p2 = points_on_line[i], points_on_line[i + 1]
            edge = tuple(sorted([p1, p2]))
            if edge not in edges_created:
                weight = abs(p2[0] - p1[0]) + abs(p2[1] - p1[1])
                G.add_edge(p1, p2, weight=weight)
                edges_created.add(edge)

    # 2. Add aisles
    aisle_pattern = re.compile(r'^A\d+-[RL]-\d+$')
    aisle_layers = [layer.dxf.name for layer in doc.layers if aisle_pattern.match(layer.dxf.name)]

    for layer_name in aisle_layers:
        points = []
        for entity in msp:
            if entity.dxf.layer == layer_name and entity.dxftype() == 'LWPOLYLINE':
                points = [(p[0], p[1]) for p in entity.get_points()]
                break

        if points:
            # Calculate centroid
            cx = sum(p[0] for p in points) / len(points)
            cy = sum(p[1] for p in points) / len(points)
            centroid = (round(cx, 2), round(cy, 2))
            G.add_node(centroid, pos=centroid, aisle=layer_name)

            # Connect touching passage nodes
            for node in [n for n in G.nodes() if 'aisle' not in G.nodes[n]]:
                poly_points = [(round(p[0], 2), round(p[1], 2)) for p in points]
                for i in range(len(poly_points)):
                    p1 = poly_points[i]
                    p2 = poly_points[(i + 1) % len(poly_points)]
                    if point_on_segment(node, p1, p2, tol=2.0):
                        G.add_edge(node, centroid, weight=0)
                        break

    # 3. Add depot
    depot_points = []
    for entity in msp:
        if entity.dxf.layer.lower() == 'depot' and entity.dxftype() == 'LWPOLYLINE':
            depot_points = [(round(p[0], 2), round(p[1], 2)) for p in entity.get_points()]
            break

    if depot_points:
        # Calculate depot centroid
        depot_x = sum(p[0] for p in depot_points) / len(depot_points)
        depot_y = sum(p[1] for p in depot_points) / len(depot_points)
        depot_centroid = (round(depot_x, 2), round(depot_y, 2))
        G.add_node(depot_centroid, pos=depot_centroid, depot=True)

        # Connect touching passage nodes
        for node in [n for n in G.nodes() if 'aisle' not in G.nodes[n] and 'depot' not in G.nodes[n]]:
            for i in range(len(depot_points)):
                p1 = depot_points[i]
                p2 = depot_points[(i + 1) % len(depot_points)]
                if point_on_segment(node, p1, p2, tol=5.0):
                    G.add_edge(node, depot_centroid, weight=0)
                    break

    return G


def main():
    """Command line interface."""
    if len(sys.argv) != 2:
        print("Usage: python build_warehouse_graph.py <dxf_file>")
        sys.exit(1)

    dxf_file = sys.argv[1]

    try:
        G = build_warehouse_graph(dxf_file)

        # Print summary
        passages = [n for n in G.nodes() if 'aisle' not in G.nodes[n] and 'depot' not in G.nodes[n]]
        aisles = [n for n in G.nodes() if 'aisle' in G.nodes[n]]
        depot = [n for n in G.nodes() if 'depot' in G.nodes[n]]
        passage_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > 0]
        connection_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] == 0]

        print(f"\nWarehouse Graph Built Successfully!")
        print(f"{'=' * 50}")
        print(f"Nodes: {G.number_of_nodes()}")
        print(f"  - Passage nodes: {len(passages)}")
        print(f"  - Aisle nodes: {len(aisles)}")
        print(f"  - Depot nodes: {len(depot)}")
        print(f"\nEdges: {G.number_of_edges()}")
        print(f"  - Passage edges: {len(passage_edges)}")
        print(f"  - Zero-weight connections: {len(connection_edges)}")
        print(f"\nConnected: {nx.is_connected(G)}")

        return G

    except FileNotFoundError:
        print(f"Error: File '{dxf_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

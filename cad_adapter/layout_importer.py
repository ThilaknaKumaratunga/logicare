"""
DXF Layout Importer for Warehouse Graphs

Uses build_warehouse_graph module to parse DXF files and convert to WarehouseGraph.
"""

from pathlib import Path
from graph.graph import Node, Edge, WarehouseGraph


class DXFAdapter:
    """DXF CAD file import adapter using build_warehouse_graph module."""

    def validate_format(self, file_path: str) -> bool:
        """Check if file is a valid DXF file."""
        return file_path.endswith('.dxf') and Path(file_path).exists()

    def import_layout(self, file_path: str) -> WarehouseGraph:
        """Import from DXF file using build_warehouse_graph."""
        from build_warehouse_graph import build_warehouse_graph

        nx_graph = build_warehouse_graph(file_path)
        return self._convert_networkx_to_warehouse_graph(nx_graph)

    def _convert_networkx_to_warehouse_graph(self, nx_graph) -> WarehouseGraph:
        """Convert NetworkX graph to WarehouseGraph."""
        warehouse = WarehouseGraph()
        coord_to_id = {}

        # Add nodes
        for node_id, node_data in nx_graph.nodes(data=True):
            pos = node_data.get('pos', node_id)
            x, y = pos

            # Determine node type and string ID
            if 'depot' in node_data:
                node_type, str_id, metadata = 'depot', 'DEPOT', {'start': True}
            elif 'aisle' in node_data:
                node_type, str_id = 'product', node_data['aisle']
                metadata = {'aisle': node_data['aisle']}
            else:
                node_type, str_id, metadata = 'junction', f"P_{x}_{y}", {}

            warehouse.add_node(Node(id=str_id, x=x, y=y, z=0.0, node_type=node_type, metadata=metadata))
            coord_to_id[node_id] = str_id

        # Add edges
        for u, v, edge_data in nx_graph.edges(data=True):
            weight = edge_data.get('weight', 1.0)
            warehouse.add_edge(Edge(
                from_node=coord_to_id[u],
                to_node=coord_to_id[v],
                travel_time=weight,
                distance=weight,
                bidirectional=True,
                direction_allowed='both'
            ))

        warehouse.build_adjacency_lists()
        return warehouse


class LayoutImporter:
    """Main importer class for DXF warehouse layouts."""

    def __init__(self):
        self.adapter = DXFAdapter()

    def import_from_file(self, file_path: str) -> WarehouseGraph:
        """
        Import warehouse layout from DXF file.

        Args:
            file_path: Path to the DXF layout file

        Returns:
            WarehouseGraph object
        """
        if not self.adapter.validate_format(file_path):
            raise ValueError(f"Invalid or unsupported file: {file_path}")

        return self.adapter.import_layout(file_path)

"""
Layout importer for warehouse graphs.

Provides interfaces for importing warehouse layouts from various sources
(JSON, CSV, and future CAD formats).
"""

import json
import csv
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from pathlib import Path

from graph.graph import Node, Edge, WarehouseGraph


class CADAdapterInterface(ABC):
    """
    Abstract interface for CAD file importers.
    
    Future implementations should inherit from this class to provide
    CAD-specific import functionality (DXF, DWG, STEP, etc.)
    """
    
    @abstractmethod
    def import_layout(self, file_path: str) -> WarehouseGraph:
        """
        Import a warehouse layout from a CAD file.
        
        Args:
            file_path: Path to the CAD file
            
        Returns:
            WarehouseGraph object representing the layout
        """
        pass
    
    @abstractmethod
    def validate_format(self, file_path: str) -> bool:
        """Check if the file format is supported."""
        pass


class JSONLayoutImporter(CADAdapterInterface):
    """
    Import warehouse layouts from JSON files.
    
    Expected JSON format:
    {
        "nodes": [
            {
                "id": "depot",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "node_type": "depot",
                "metadata": {}
            },
            ...
        ],
        "edges": [
            {
                "from_node": "depot",
                "to_node": "A1",
                "travel_time": 10.0,
                "distance": 5.0,
                "bidirectional": true,
                "direction_allowed": "both",
                "metadata": {}
            },
            ...
        ]
    }
    """
    
    def validate_format(self, file_path: str) -> bool:
        """Check if file is valid JSON."""
        try:
            with open(file_path, 'r') as f:
                json.load(f)
            return True
        except (json.JSONDecodeError, FileNotFoundError):
            return False
    
    def import_layout(self, file_path: str) -> WarehouseGraph:
        """Import warehouse layout from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)

        graph = WarehouseGraph()

        # Check which format is being used
        if 'warehouse' in data:
            warehouse_data = data['warehouse']
            # Check if it's double-sided layout (has 'sides' in locations)
            if warehouse_data.get('locations') and isinstance(warehouse_data['locations'][0].get('sides'), dict):
                return self._import_double_sided_layout(warehouse_data)
            # Format 2: Aisle-based warehouse (single-sided)
            return self._import_aisle_based_layout(warehouse_data)

        # Format 1: Direct nodes/edges
        # Import nodes
        for node_data in data.get('nodes', []):
            # Get node_type from metadata['type'] or direct field
            metadata = node_data.get('metadata', {})
            node_type = node_data.get('node_type') or metadata.get('type', 'junction')
            
            node = Node(
                id=node_data['id'],
                x=node_data.get('x', 0.0),
                y=node_data.get('y', 0.0),
                z=node_data.get('z', 0.0),
                node_type=node_type,
                metadata=metadata
            )
            graph.add_node(node)
        
        # Import edges
        for edge_data in data.get('edges', []):
            # Support both 'from'/'to' and 'from_node'/'to_node' formats
            from_node = edge_data.get('from_node') or edge_data.get('from')
            to_node = edge_data.get('to_node') or edge_data.get('to')
            
            edge = Edge(
                from_node=from_node,
                to_node=to_node,
                travel_time=edge_data.get('travel_time', 1.0),
                distance=edge_data.get('distance', 0.0),
                bidirectional=edge_data.get('bidirectional', True),
                direction_allowed=edge_data.get('direction_allowed', 'both'),
                capacity=edge_data.get('capacity'),
                metadata=edge_data.get('metadata', {})
            )
            graph.add_edge(edge)
        
        # Build adjacency lists
        graph.build_adjacency_lists()

        return graph

    def _import_aisle_based_layout(self, warehouse_data: dict) -> WarehouseGraph:
        """Import warehouse layout from aisle-based format."""
        graph = WarehouseGraph()

        start = warehouse_data.get('start', {'x': 0, 'y': 0, 'z': 0})

        # Add depot/start node
        depot_node = Node(
            id='DEPOT',
            x=start['x'],
            y=start['y'],
            z=start.get('z', 0),
            node_type='depot',
            metadata={'start': True}
        )
        graph.add_node(depot_node)

        # Track all cells for edge creation
        all_cells = []

        # Import locations from aisles
        for location_group in warehouse_data.get('locations', []):
            aisle_id = location_group.get('aisle', 'UNKNOWN')

            for cell in location_group.get('cells', []):
                cell_id = cell.get('id', f"{aisle_id}-{cell['x']}-{cell['y']}")

                node = Node(
                    id=cell_id,
                    x=cell['x'],
                    y=cell['y'],
                    z=cell.get('z', 0),
                    node_type='product',
                    metadata={'aisle': aisle_id}
                )
                graph.add_node(node)
                all_cells.append(node)

        # Auto-generate edges based on proximity
        self._generate_edges_from_cells(graph, depot_node, all_cells)

        # Build adjacency lists
        graph.build_adjacency_lists()

        return graph

    def _import_double_sided_layout(self, warehouse_data: dict) -> WarehouseGraph:
        """Import warehouse layout with double-sided aisles and cross-aisle passages."""
        graph = WarehouseGraph()

        start = warehouse_data.get('start', {'x': 0, 'y': 0, 'z': 0})

        # Add depot node
        depot_node = Node(
            id=start.get('id', 'DEPOT'),
            x=start['x'],
            y=start['y'],
            z=start.get('z', 0),
            node_type='depot',
            metadata={'start': True}
        )
        graph.add_node(depot_node)

        # Get passage levels
        passages = warehouse_data.get('passages', {}).get('horizontal', [0, 12])

        # Track all nodes by coordinates for cross-aisle connections
        nodes_by_coord = {}  # (x, y) -> node

        # Import all storage locations
        for location_group in warehouse_data.get('locations', []):
            aisle_id = location_group.get('aisle', 'UNKNOWN')
            sides = location_group.get('sides', {})

            for side_id, cells in sides.items():
                for cell in cells:
                    node_id = cell.get('id', f"{aisle_id}-{side_id}-{cell['y']}")
                    node = Node(
                        id=node_id,
                        x=cell['x'],
                        y=cell['y'],
                        z=cell.get('z', 0),
                        node_type='product',
                        metadata={'aisle': aisle_id, 'side': side_id}
                    )
                    graph.add_node(node)
                    nodes_by_coord[(cell['x'], cell['y'])] = node

        # Generate edges
        self._generate_double_sided_edges(graph, depot_node, nodes_by_coord, passages)

        # Build adjacency lists
        graph.build_adjacency_lists()

        return graph

    def _generate_double_sided_edges(self, graph: WarehouseGraph, depot: Node,
                                     nodes_by_coord: dict, passages: list) -> None:
        """Generate edges for double-sided warehouse with cross-aisle passages."""

        def add_edge(from_id: str, to_id: str, edge_type: str = 'internal'):
            from_node = graph.nodes.get(from_id)
            to_node = graph.nodes.get(to_id)
            if from_node and to_node:
                distance = abs(from_node.x - to_node.x) + abs(from_node.y - to_node.y)
                edge = Edge(
                    from_node=from_id,
                    to_node=to_id,
                    travel_time=distance,
                    distance=distance,
                    bidirectional=True,
                    direction_allowed='both',
                    metadata={'type': edge_type}
                )
                graph.add_edge(edge)

        # 1. Connect depot to lowest level nodes for all x-coordinates
        # Find minimum y value in the warehouse
        min_y = min(y for x, y in nodes_by_coord.keys())
        passage_nodes_at_bottom = [(x, y, nid) for (x, y), n in nodes_by_coord.items()
                                   if y == min_y for nid in [n.id]]
        for x, y, node_id in passage_nodes_at_bottom:
            add_edge(depot.id, node_id, 'depot_to_passage')

        # 2. Vertical connections within same x-coordinate (along aisle sides)
        by_x = {}
        for (x, y), node in nodes_by_coord.items():
            if x not in by_x:
                by_x[x] = []
            by_x[x].append((y, node.id))

        for x, nodes in by_x.items():
            nodes.sort()  # Sort by y
            for i in range(len(nodes) - 1):
                add_edge(nodes[i][1], nodes[i+1][1], 'vertical')

        # 3. Horizontal cross-aisle connections at passage levels
        for passage_y in passages:
            nodes_at_passage = [(x, nid) for (x, y), n in nodes_by_coord.items()
                               if y == passage_y for nid in [n.id]]
            nodes_at_passage.sort()  # Sort by x

            for i in range(len(nodes_at_passage) - 1):
                add_edge(nodes_at_passage[i][1], nodes_at_passage[i+1][1], 'cross_aisle')

    def _generate_edges_from_cells(self, graph: WarehouseGraph, depot: Node, cells: list) -> None:
        """Generate edges for warehouse with passages."""
        if not cells:
            return

        def manhattan_distance(n1: Node, n2: Node) -> float:
            return abs(n1.x - n2.x) + abs(n1.y - n2.y)

        # Group cells by aisle
        aisles = {}
        for cell in cells:
            aisle = cell.metadata.get('aisle', 'UNKNOWN')
            if aisle not in aisles:
                aisles[aisle] = []
            aisles[aisle].append(cell)

        # Sort cells within each aisle by y-coordinate
        for aisle in aisles:
            aisles[aisle].sort(key=lambda c: c.y)

        # 1. Connect depot to bottom of each aisle
        for aisle, aisle_cells in aisles.items():
            if aisle_cells:
                bottom_cell = aisle_cells[0]
                distance = manhattan_distance(depot, bottom_cell)
                edge = Edge(
                    from_node=depot.id,
                    to_node=bottom_cell.id,
                    travel_time=distance,
                    distance=distance,
                    bidirectional=True,
                    direction_allowed='both',
                    metadata={'type': 'depot_to_aisle', 'aisle': aisle}
                )
                graph.add_edge(edge)

        # 2. Connect blocks within same aisle (vertical movement)
        for aisle, aisle_cells in aisles.items():
            for i in range(len(aisle_cells) - 1):
                cell1 = aisle_cells[i]
                cell2 = aisle_cells[i + 1]

                if cell1.x == cell2.x:
                    distance = abs(cell2.y - cell1.y)
                    edge = Edge(
                        from_node=cell1.id,
                        to_node=cell2.id,
                        travel_time=distance,
                        distance=distance,
                        bidirectional=True,
                        direction_allowed='both',
                        metadata={'type': 'within_aisle', 'aisle': aisle}
                    )
                    graph.add_edge(edge)

        # 3. Connect adjacent aisles at bottom and top
        aisle_list = sorted(aisles.items(), key=lambda x: x[1][0].x if x[1] else 0)
        for i in range(len(aisle_list) - 1):
            aisle1_name, aisle1_cells = aisle_list[i]
            aisle2_name, aisle2_cells = aisle_list[i + 1]

            if aisle1_cells and aisle2_cells:
                # Bottom
                bottom1 = aisle1_cells[0]
                bottom2 = aisle2_cells[0]
                distance = manhattan_distance(bottom1, bottom2)
                edge = Edge(
                    from_node=bottom1.id,
                    to_node=bottom2.id,
                    travel_time=distance,
                    distance=distance,
                    bidirectional=True,
                    direction_allowed='both',
                    metadata={'type': 'cross_aisle', 'location': 'bottom'}
                )
                graph.add_edge(edge)

                # Top
                top1 = aisle1_cells[-1]
                top2 = aisle2_cells[-1]
                distance = manhattan_distance(top1, top2)
                edge = Edge(
                    from_node=top1.id,
                    to_node=top2.id,
                    travel_time=distance,
                    distance=distance,
                    bidirectional=True,
                    direction_allowed='both',
                    metadata={'type': 'cross_aisle', 'location': 'top'}
                )
                graph.add_edge(edge)


class CSVLayoutImporter(CADAdapterInterface):
    """
    Import warehouse layouts from CSV files.
    
    Expects two CSV files:
    - nodes.csv: id, x, y, z, node_type
    - edges.csv: from_node, to_node, travel_time, distance, bidirectional, direction_allowed
    """
    
    def validate_format(self, nodes_file: str, edges_file: str) -> bool:
        """Check if both CSV files exist and are readable."""
        try:
            Path(nodes_file).exists() and Path(edges_file).exists()
            return True
        except Exception:
            return False
    
    def import_layout(self, nodes_file: str, edges_file: Optional[str] = None) -> WarehouseGraph:
        """
        Import warehouse layout from CSV files.
        
        Args:
            nodes_file: Path to nodes CSV file
            edges_file: Path to edges CSV file (optional, will try to infer from nodes_file)
        """
        if edges_file is None:
            # Try to infer edges file path
            nodes_path = Path(nodes_file)
            edges_file = str(nodes_path.parent / f"{nodes_path.stem}_edges{nodes_path.suffix}")
        
        graph = WarehouseGraph()
        
        # Import nodes
        with open(nodes_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                node = Node(
                    id=row['id'],
                    x=float(row.get('x', 0.0)),
                    y=float(row.get('y', 0.0)),
                    z=float(row.get('z', 0.0)),
                    node_type=row.get('node_type', 'junction')
                )
                graph.add_node(node)
        
        # Import edges
        with open(edges_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                edge = Edge(
                    from_node=row['from_node'],
                    to_node=row['to_node'],
                    travel_time=float(row.get('travel_time', 1.0)),
                    distance=float(row.get('distance', 0.0)),
                    bidirectional=row.get('bidirectional', 'true').lower() == 'true',
                    direction_allowed=row.get('direction_allowed', 'both')
                )
                graph.add_edge(edge)
        
        # Build adjacency lists
        graph.build_adjacency_lists()
        
        return graph


class DXFAdapterStub(CADAdapterInterface):
    """
    Placeholder stub for DXF CAD file import.
    
    To be implemented when CAD integration is required.
    Will parse DXF files and extract:
    - Wall/obstacle geometry
    - Aisle paths
    - Product locations
    - Depot locations
    """
    
    def validate_format(self, file_path: str) -> bool:
        """Check if file is a valid DXF file."""
        return file_path.endswith('.dxf')
    
    def import_layout(self, file_path: str) -> WarehouseGraph:
        """Import from DXF file (stub implementation)."""
        raise NotImplementedError(
            "DXF import not yet implemented. "
            "Use JSONLayoutImporter or CSVLayoutImporter for POC."
        )


class LayoutImporter:
    """
    Main importer class that delegates to appropriate adapter based on file type.
    """
    
    def __init__(self):
        self.adapters: Dict[str, CADAdapterInterface] = {
            '.json': JSONLayoutImporter(),
            '.csv': CSVLayoutImporter(),
            '.dxf': DXFAdapterStub(),
        }
    
    def import_from_file(self, file_path: str, **kwargs) -> WarehouseGraph:
        """
        Import warehouse layout from file.
        
        Args:
            file_path: Path to the layout file
            **kwargs: Additional arguments for specific importers
            
        Returns:
            WarehouseGraph object
        """
        file_path_obj = Path(file_path)
        extension = file_path_obj.suffix.lower()
        
        if extension not in self.adapters:
            raise ValueError(
                f"Unsupported file format: {extension}. "
                f"Supported formats: {list(self.adapters.keys())}"
            )
        
        adapter = self.adapters[extension]
        return adapter.import_layout(file_path, **kwargs)
    
    def load_from_json(self, file_path: str) -> WarehouseGraph:
        """
        Convenience method to load from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            WarehouseGraph object
        """
        return self.import_from_file(file_path)
    
    def load_from_csv(self, nodes_file: str, edges_file: Optional[str] = None) -> WarehouseGraph:
        """
        Convenience method to load from CSV files.
        
        Args:
            nodes_file: Path to nodes CSV file
            edges_file: Path to edges CSV file (optional)
            
        Returns:
            WarehouseGraph object
        """
        return self.adapters['.csv'].import_layout(nodes_file, edges_file)
    
    def get_cad_adapter(self, format_type: str) -> CADAdapterInterface:
        """
        Get a specific CAD adapter by format type.
        
        Args:
            format_type: Format type ('json', 'csv', 'dxf', etc.)
            
        Returns:
            CADAdapterInterface instance
        """
        extension = f".{format_type.lower()}"
        if extension not in self.adapters:
            raise ValueError(f"No adapter found for format: {format_type}")
        return self.adapters[extension]
    
    def register_adapter(self, extension: str, adapter: CADAdapterInterface) -> None:
        """Register a custom adapter for a specific file extension."""
        self.adapters[extension] = adapter

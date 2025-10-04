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
            # Check if it's block-based layout (has 'aisles' with block centers)
            if warehouse_data.get('aisles') and warehouse_data.get('passages'):
                return self._import_block_based_layout(warehouse_data)
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

    def _import_block_based_layout(self, warehouse_data: dict) -> WarehouseGraph:
        """
        Import warehouse layout with block-based design and passage centerline routing.

        Blocks are 4×4 units with centers as node positions.
        All routing goes through passage centerlines (middle of 2-unit passages).
        """
        graph = WarehouseGraph()

        # Add depot node
        depot_data = warehouse_data.get('depot', {'x': 2, 'y': 1, 'id': 'DEPOT'})
        depot_node = Node(
            id=depot_data.get('id', 'DEPOT'),
            x=depot_data['x'],
            y=depot_data['y'],
            z=depot_data.get('z', 0),
            node_type='depot',
            metadata={'start': True}
        )
        graph.add_node(depot_node)

        # Add all block nodes (at block centers)
        for aisle in warehouse_data.get('aisles', []):
            aisle_id = aisle.get('id')
            for side_name, side_data in aisle.get('sides', {}).items():
                for block in side_data.get('blocks', []):
                    node = Node(
                        id=block['id'],
                        x=block['x'],
                        y=block['y'],
                        z=0,
                        node_type='product',
                        metadata={'aisle': aisle_id, 'side': side_name}
                    )
                    graph.add_node(node)

        # Generate passage-based routing edges
        self._generate_passage_routing_edges(graph, warehouse_data)

        # Build adjacency lists
        graph.build_adjacency_lists()

        return graph

    def _generate_passage_routing_edges(self, graph: WarehouseGraph, warehouse_data: dict) -> None:
        """
        Generate edges for passage-based routing.

        Routing logic:
        1. Block center → nearest vertical passage centerline (horizontal movement)
        2. Vertical passage centerline → horizontal passage centerline (vertical movement)
        3. Horizontal passage centerlines connect across aisles
        4. Depot connects to bottom horizontal passage
        """

        def manhattan_distance(x1, y1, x2, y2):
            return abs(x2 - x1) + abs(y2 - y1)

        def add_edge(from_id, to_id, distance, edge_type='passage'):
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

        # Get passage data
        h_passages = warehouse_data.get('passages', {}).get('horizontal', [])
        v_passages = warehouse_data.get('passages', {}).get('vertical', [])

        # Extract passage centerlines
        h_passage_y = [p['y'] for p in h_passages]  # y-coordinates of horizontal passages
        v_passage_x = [p['x'] for p in v_passages]  # x-coordinates of vertical passages

        depot = graph.nodes['DEPOT']

        # 1. Connect depot to bottom horizontal passage
        # Depot is at (2,1), bottom passage is at y=1
        # Find closest vertical passage on the bottom horizontal passage
        closest_vp_x = min(v_passage_x, key=lambda x: abs(x - depot.x))
        # Distance from depot to intersection of closest VP and bottom HP
        dist_to_passage = abs(closest_vp_x - depot.x)

        # For now, connect depot directly to all blocks on bottom passage
        # (We'll create passage intersection nodes in a more sophisticated version)

        # 2. Connect each block to other blocks via passages
        # To avoid duplicates, only create edge if node_id < other_id (lexicographically)
        nodes_list = [(nid, n) for nid, n in graph.nodes.items() if n.node_type == 'product']

        for i, (node_id, node) in enumerate(nodes_list):
            nearest_vp_x = min(v_passage_x, key=lambda x: abs(x - node.x))

            # Only connect to nodes that come after this one (avoid duplicates)
            for j in range(i + 1, len(nodes_list)):
                other_id, other_node = nodes_list[j]

                other_nearest_vp = min(v_passage_x, key=lambda x: abs(x - other_node.x))

                # Calculate passage-based distance via bottom passage
                # node → VP1 (horizontal) → HP (along horizontal) → VP2 (along horizontal) → other (horizontal)
                hp_y = h_passage_y[0]  # Use bottom passage

                dist1 = abs(node.x - nearest_vp_x) + abs(node.y - hp_y)  # node to VP1-HP intersection
                dist2 = abs(nearest_vp_x - other_nearest_vp)  # along horizontal passage
                dist3 = abs(other_node.x - other_nearest_vp) + abs(other_node.y - hp_y)  # VP2-HP to other node

                total = dist1 + dist2 + dist3
                add_edge(node_id, other_id, total, 'via_passage')

        # 3. Connect depot to all blocks via bottom passage
        bottom_passage_y = h_passage_y[0]
        for node_id, node in graph.nodes.items():
            if node.node_type != 'product':
                continue

            # Find nearest VP to block
            nearest_vp = min(v_passage_x, key=lambda x: abs(x - node.x))

            # Find nearest VP to depot
            depot_nearest_vp = min(v_passage_x, key=lambda x: abs(x - depot.x))

            # Distance: depot → VP → HP → VP → block
            depot_to_vp = abs(depot.x - depot_nearest_vp)
            depot_to_hp = abs(depot.y - bottom_passage_y)
            hp_distance = abs(depot_nearest_vp - nearest_vp)
            hp_to_block_vp = abs(node.x - nearest_vp)
            block_vp_to_block = abs(node.y - bottom_passage_y)

            total_dist = depot_to_vp + depot_to_hp + hp_distance + hp_to_block_vp + block_vp_to_block
            add_edge(depot.id, node_id, total_dist, 'depot_to_block')

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

        # Get passage levels (handle both old and new format)
        passages_data = warehouse_data.get('passages', {}).get('horizontal', [0, 12])
        if passages_data and isinstance(passages_data[0], dict):
            passages = [p['y'] for p in passages_data]
        else:
            passages = passages_data

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
        """Generate edges for double-sided warehouse with passage-based routing.

        Key design: All vertical movement happens through passage nodes at specific y-levels.
        Storage blocks connect horizontally to passages only.
        """

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

        # Group nodes by x (aisle sides) and y (passage levels)
        by_x = {}
        for (x, y), node in nodes_by_coord.items():
            if x not in by_x:
                by_x[x] = []
            by_x[x].append((y, node.id))

        # Sort each column by y
        for x in by_x:
            by_x[x].sort()

        # 1. Connect depot to passage nodes at min_y
        min_y = min(y for x, y in nodes_by_coord.keys())
        for x in by_x:
            nodes_in_col = by_x[x]
            if nodes_in_col and nodes_in_col[0][0] == min_y:
                add_edge(depot.id, nodes_in_col[0][1], 'depot_to_passage')

        # 2. For each aisle side (x-coordinate), connect blocks to passage nodes
        # Only storage blocks at passage y-levels are accessible
        for x in by_x:
            nodes_in_col = by_x[x]

            # Connect each storage block to nearest passage nodes (up/down)
            for i, (y, node_id) in enumerate(nodes_in_col):
                if y not in passages:
                    # Non-passage node - connect to nearest passage above and below
                    # Find nearest passage below
                    passage_below = None
                    for j in range(i-1, -1, -1):
                        if nodes_in_col[j][0] in passages:
                            passage_below = nodes_in_col[j][1]
                            break

                    # Find nearest passage above
                    passage_above = None
                    for j in range(i+1, len(nodes_in_col)):
                        if nodes_in_col[j][0] in passages:
                            passage_above = nodes_in_col[j][1]
                            break

                    # Connect to passages
                    if passage_below:
                        add_edge(node_id, passage_below, 'to_passage')
                    if passage_above:
                        add_edge(node_id, passage_above, 'to_passage')

        # 3. Vertical connections between passage nodes on same x-coordinate
        for x in by_x:
            passage_nodes_in_col = [(y, nid) for y, nid in by_x[x] if y in passages]
            passage_nodes_in_col.sort()

            for i in range(len(passage_nodes_in_col) - 1):
                add_edge(passage_nodes_in_col[i][1], passage_nodes_in_col[i+1][1], 'vertical_passage')

        # 4. Horizontal connections across aisles at passage levels
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

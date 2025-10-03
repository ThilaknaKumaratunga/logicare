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

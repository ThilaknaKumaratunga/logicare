"""
Graph representation for warehouse layout.

This module defines the core data structures for representing warehouse layouts
as directed graphs with nodes (locations) and edges (paths).
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field


@dataclass
class Node:
    """
    Represents a location in the warehouse.
    
    Attributes:
        id: Unique identifier for the node
        x: X-coordinate in warehouse layout
        y: Y-coordinate in warehouse layout
        z: Z-coordinate (for 3D layouts, optional)
        node_type: Type of node ('depot', 'product', 'junction', etc.)
        metadata: Additional information (e.g., product SKU, capacity)
    """
    id: str
    x: float
    y: float
    z: float = 0.0
    node_type: str = 'junction'
    metadata: Dict = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, Node):
            return self.id == other.id
        return False
    
    def distance_to(self, other: 'Node') -> float:
        """Calculate Euclidean distance to another node."""
        return ((self.x - other.x)**2 + 
                (self.y - other.y)**2 + 
                (self.z - other.z)**2) ** 0.5


@dataclass
class Edge:
    """
    Represents a path between two nodes.
    
    Attributes:
        from_node: Starting node ID
        to_node: Ending node ID
        travel_time: Time to traverse this edge (in seconds)
        distance: Physical distance
        bidirectional: Whether travel is allowed in both directions
        direction_allowed: Direction constraint ('both', 'forward', 'reverse')
        capacity: Maximum number of carts that can use this path simultaneously
        metadata: Additional edge properties (e.g., aisle width, clearance)
    """
    from_node: str
    to_node: str
    travel_time: float
    distance: float = 0.0
    bidirectional: bool = True
    direction_allowed: str = 'both'  # 'both', 'forward', 'reverse'
    capacity: Optional[int] = None
    metadata: Dict = field(default_factory=dict)
    
    def is_valid_direction(self, from_id: str, to_id: str) -> bool:
        """Check if traversal from from_id to to_id is allowed."""
        if from_id == self.from_node and to_id == self.to_node:
            return self.direction_allowed in ['both', 'forward']
        elif from_id == self.to_node and to_id == self.from_node:
            return self.direction_allowed in ['both', 'reverse'] and self.bidirectional
        return False


class WarehouseGraph:
    """
    Represents the complete warehouse layout as a directed graph.
    
    Provides methods for graph construction, validation, and querying.
    """
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.edges_dict: Dict[Tuple[str, str], Edge] = {}  # For O(1) edge lookup
        # Adjacency lists for efficient lookups
        self._outgoing: Dict[str, List[Edge]] = {}
        self._incoming: Dict[str, List[Edge]] = {}
        self._adjacency_built = False
    
    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        if node.id in self.nodes:
            raise ValueError(f"Node {node.id} already exists")
        self.nodes[node.id] = node
        self._adjacency_built = False
    
    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph."""
        if edge.from_node not in self.nodes:
            raise ValueError(f"Node {edge.from_node} not found")
        if edge.to_node not in self.nodes:
            raise ValueError(f"Node {edge.to_node} not found")
        self.edges.append(edge)
        
        # Add to edges_dict for fast lookup
        self.edges_dict[(edge.from_node, edge.to_node)] = edge
        if edge.bidirectional:
            self.edges_dict[(edge.to_node, edge.from_node)] = edge
        
        self._adjacency_built = False
    
    def build_adjacency_lists(self) -> None:
        """Build adjacency lists for efficient lookups."""
        self._outgoing.clear()
        self._incoming.clear()
        
        # Initialize empty lists for all nodes
        for node_id in self.nodes:
            self._outgoing[node_id] = []
            self._incoming[node_id] = []
        
        # Populate adjacency lists
        for edge in self.edges:
            # Forward direction
            if edge.direction_allowed in ['both', 'forward']:
                self._outgoing[edge.from_node].append(edge)
                self._incoming[edge.to_node].append(edge)
            
            # Reverse direction (if bidirectional)
            if edge.bidirectional and edge.direction_allowed in ['both', 'reverse']:
                # Create a reverse view of the edge
                self._outgoing[edge.to_node].append(edge)
                self._incoming[edge.from_node].append(edge)
        
        self._adjacency_built = True
    
    def get_outgoing_edges(self, node_id: str) -> List[Edge]:
        """Get all edges going out from a node."""
        if not self._adjacency_built:
            self.build_adjacency_lists()
        return self._outgoing.get(node_id, [])
    
    def get_incoming_edges(self, node_id: str) -> List[Edge]:
        """Get all edges coming into a node."""
        if not self._adjacency_built:
            self.build_adjacency_lists()
        return self._incoming.get(node_id, [])
    
    def get_neighbors(self, node_id: str) -> List[str]:
        """Get all neighbor node IDs (nodes directly reachable from node_id)."""
        if not self._adjacency_built:
            self.build_adjacency_lists()
        
        neighbors = []
        for edge in self._outgoing.get(node_id, []):
            if edge.from_node == node_id:
                neighbors.append(edge.to_node)
            else:
                neighbors.append(edge.from_node)
        return neighbors
    
    def is_edge_valid(self, from_id: str, to_id: str) -> bool:
        """Check if there's a valid edge from from_id to to_id."""
        if not self._adjacency_built:
            self.build_adjacency_lists()
        
        for edge in self._outgoing.get(from_id, []):
            if edge.is_valid_direction(from_id, to_id):
                if (edge.from_node == from_id and edge.to_node == to_id) or \
                   (edge.from_node == to_id and edge.to_node == from_id):
                    return True
        return False
    
    def get_edge(self, from_id: str, to_id: str) -> Optional[Edge]:
        """Get the edge between two nodes (if it exists and is valid)."""
        if not self._adjacency_built:
            self.build_adjacency_lists()
        
        for edge in self._outgoing.get(from_id, []):
            if edge.is_valid_direction(from_id, to_id):
                if (edge.from_node == from_id and edge.to_node == to_id) or \
                   (edge.from_node == to_id and edge.to_node == from_id):
                    return edge
        return None
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_nodes_by_type(self, node_type: str) -> List[Node]:
        """Get all nodes of a specific type."""
        return [node for node in self.nodes.values() if node.node_type == node_type]
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the graph structure.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check if there's at least one depot
        depots = self.get_nodes_by_type('depot')
        if not depots:
            errors.append("No depot node found")
        
        # Check for isolated nodes
        self.build_adjacency_lists()
        for node_id in self.nodes:
            if not self._outgoing[node_id] and not self._incoming[node_id]:
                errors.append(f"Node {node_id} is isolated (no edges)")
        
        # Check for invalid edge references
        for edge in self.edges:
            if edge.from_node not in self.nodes:
                errors.append(f"Edge references non-existent node: {edge.from_node}")
            if edge.to_node not in self.nodes:
                errors.append(f"Edge references non-existent node: {edge.to_node}")
            if edge.travel_time < 0:
                errors.append(f"Edge ({edge.from_node}, {edge.to_node}) has negative travel time")
        
        return len(errors) == 0, errors
    
    def get_statistics(self) -> Dict:
        """Get statistics about the graph."""
        if not self._adjacency_built:
            self.build_adjacency_lists()
        
        return {
            'num_nodes': len(self.nodes),
            'num_edges': len(self.edges),
            'num_depots': len(self.get_nodes_by_type('depot')),
            'num_products': len(self.get_nodes_by_type('product')),
            'num_junctions': len(self.get_nodes_by_type('junction')),
            'avg_degree': sum(len(edges) for edges in self._outgoing.values()) / len(self.nodes) if self.nodes else 0
        }
    
    def __repr__(self):
        return f"WarehouseGraph(nodes={len(self.nodes)}, edges={len(self.edges)})"


# Alias for backward compatibility
Graph = WarehouseGraph

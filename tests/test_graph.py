"""
Unit tests for Graph module
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.graph import Node, Edge, Graph


class TestNode(unittest.TestCase):
    """Test Node class."""

    def test_node_creation(self):
        """Test creating a node."""
        node = Node(id="A1", x=10.0, y=20.0, metadata={"type": "location"})
        self.assertEqual(node.id, "A1")
        self.assertEqual(node.x, 10.0)
        self.assertEqual(node.y, 20.0)
        self.assertEqual(node.metadata["type"], "location")
    
    def test_node_distance(self):
        """Test distance calculation between nodes."""
        node1 = Node(id="A", x=0, y=0)
        node2 = Node(id="B", x=3, y=4)
        distance = node1.distance_to(node2)
        self.assertEqual(distance, 5.0)  # 3-4-5 triangle


class TestEdge(unittest.TestCase):
    """Test Edge class."""
    
    def test_edge_creation(self):
        """Test creating an edge."""
        edge = Edge(from_node="A", to_node="B", travel_time=5.0, bidirectional=True)
        self.assertEqual(edge.from_node, "A")
        self.assertEqual(edge.to_node, "B")
        self.assertEqual(edge.travel_time, 5.0)
        self.assertTrue(edge.bidirectional)

    def test_unidirectional_edge(self):
        """Test one-way edge."""
        edge = Edge(from_node="A", to_node="B", travel_time=3.0, bidirectional=False)
        self.assertFalse(edge.bidirectional)


class TestGraph(unittest.TestCase):
    """Test Graph class."""
    
    def setUp(self):
        """Set up a simple graph for testing."""
        self.graph = Graph()

        # Add nodes
        self.graph.add_node(Node(id="DEPOT", x=0, y=0, node_type="depot"))
        self.graph.add_node(Node(id="A1", x=5, y=0, node_type="product"))
        self.graph.add_node(Node(id="A2", x=10, y=0, node_type="product"))

        # Add edges
        self.graph.add_edge(Edge(from_node="DEPOT", to_node="A1", travel_time=2.0, bidirectional=True))
        self.graph.add_edge(Edge(from_node="A1", to_node="A2", travel_time=1.5, bidirectional=False))
    
    def test_node_addition(self):
        """Test adding nodes to graph."""
        self.assertEqual(len(self.graph.nodes), 3)
        self.assertIn("DEPOT", self.graph.nodes)
        self.assertIn("A1", self.graph.nodes)
    
    def test_edge_addition(self):
        """Test adding edges to graph."""
        # Should have 3 edges: DEPOT<->A1 (2) + A1->A2 (1)
        self.assertGreaterEqual(len(self.graph.edges), 2)
    
    def test_get_node(self):
        """Test retrieving a node."""
        node = self.graph.get_node("A1")
        self.assertIsNotNone(node)
        self.assertEqual(node.id, "A1")
        self.assertEqual(node.x, 5)
        self.assertEqual(node.y, 0)
    
    def test_get_nonexistent_node(self):
        """Test retrieving a non-existent node."""
        node = self.graph.get_node("NONEXISTENT")
        self.assertIsNone(node)
    
    def test_outgoing_edges(self):
        """Test getting outgoing edges."""
        outgoing = self.graph.get_outgoing_edges("A1")
        self.assertGreater(len(outgoing), 0)
        
        # Check that A1->A2 is in outgoing
        destinations = [edge.to_node for edge in outgoing]
        self.assertIn("A2", destinations)
    
    def test_incoming_edges(self):
        """Test getting incoming edges."""
        incoming = self.graph.get_incoming_edges("A1")
        self.assertGreater(len(incoming), 0)
        
        # Check that DEPOT->A1 is in incoming
        sources = [edge.from_node for edge in incoming]
        self.assertIn("DEPOT", sources)
    
    def test_edge_validation(self):
        """Test edge validity checking."""
        # Valid edges
        self.assertTrue(self.graph.is_edge_valid("DEPOT", "A1"))
        self.assertTrue(self.graph.is_edge_valid("A1", "DEPOT"))  # Bidirectional
        self.assertTrue(self.graph.is_edge_valid("A1", "A2"))
        
        # Invalid edges
        self.assertFalse(self.graph.is_edge_valid("A2", "A1"))  # One-way only
        self.assertFalse(self.graph.is_edge_valid("DEPOT", "A2"))  # Doesn't exist
    
    def test_neighbors(self):
        """Test getting node neighbors."""
        neighbors = self.graph.get_neighbors("A1")
        self.assertIn("DEPOT", neighbors)  # Bidirectional connection
        self.assertIn("A2", neighbors)  # Outgoing connection


class TestGraphIntegrity(unittest.TestCase):
    """Test graph integrity and edge cases."""
    
    def test_empty_graph(self):
        """Test operations on empty graph."""
        graph = Graph()
        self.assertEqual(len(graph.nodes), 0)
        self.assertEqual(len(graph.edges), 0)
        self.assertIsNone(graph.get_node("ANY"))
    
    def test_duplicate_node(self):
        """Test adding duplicate nodes."""
        graph = Graph()
        graph.add_node(Node(id="A", x=0, y=0))
        with self.assertRaises(ValueError):
            graph.add_node(Node(id="A", x=5, y=5))  # Should raise error
    
    def test_self_loop(self):
        """Test edge from node to itself."""
        graph = Graph()
        graph.add_node(Node(id="A", x=0, y=0))
        graph.add_edge(Edge(from_node="A", to_node="A", travel_time=0.0, bidirectional=False))
        self.assertTrue(graph.is_edge_valid("A", "A"))


if __name__ == "__main__":
    unittest.main()

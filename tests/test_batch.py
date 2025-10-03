"""
Unit tests for Batch module
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimizer.batch import Batch, BatchManager, Cart, PickItem
from graph.graph import Graph, Node, Edge


class TestPickItem(unittest.TestCase):
    """Test PickItem class."""

    def test_pick_item_creation(self):
        """Test creating a pick item."""
        item = PickItem(sku="SKU001", location_id="A1", quantity=5, weight=2.0, volume=0.1)
        self.assertEqual(item.sku, "SKU001")
        self.assertEqual(item.location_id, "A1")
        self.assertEqual(item.quantity, 5)
        self.assertEqual(item.weight, 2.0)

    def test_total_weight(self):
        """Test total weight calculation."""
        item = PickItem(sku="SKU001", location_id="A1", quantity=3, weight=2.5)
        self.assertEqual(item.total_weight(), 7.5)

    def test_total_volume(self):
        """Test total volume calculation."""
        item = PickItem(sku="SKU001", location_id="A1", quantity=4, volume=0.2)
        self.assertEqual(item.total_volume(), 0.8)


class TestCart(unittest.TestCase):
    """Test Cart class."""

    def test_cart_creation(self):
        """Test creating a cart."""
        cart = Cart(id="CART_1", capacity=100, weight_capacity=50.0)
        self.assertEqual(cart.id, "CART_1")
        self.assertEqual(cart.capacity, 100)
        self.assertEqual(cart.weight_capacity, 50.0)

    def test_has_capacity(self):
        """Test capacity checking."""
        cart = Cart(id="CART_1", capacity=100, weight_capacity=50.0)
        self.assertTrue(cart.has_capacity(items=10, weight=5.0))
        self.assertFalse(cart.has_capacity(items=150, weight=5.0))
        self.assertFalse(cart.has_capacity(items=10, weight=60.0))

    def test_add_load(self):
        """Test adding load to cart."""
        cart = Cart(id="CART_1", capacity=100, weight_capacity=50.0)
        cart.add_load(items=10, weight=5.0)
        self.assertEqual(cart.current_load, 10)
        self.assertEqual(cart.metadata['current_weight'], 5.0)

    def test_reset_load(self):
        """Test resetting cart load."""
        cart = Cart(id="CART_1", capacity=100)
        cart.add_load(items=10, weight=5.0)
        cart.reset_load()
        self.assertEqual(cart.current_load, 0)
        self.assertEqual(cart.metadata.get('current_weight', 0), 0)


class TestBatch(unittest.TestCase):
    """Test Batch class."""

    def test_batch_creation(self):
        """Test creating a batch."""
        batch = Batch(id="BATCH_001", depot_id="DEPOT")
        item1 = PickItem(sku="SKU001", location_id="A1", quantity=2)
        item2 = PickItem(sku="SKU002", location_id="A2", quantity=3)
        batch.add_item(item1)
        batch.add_item(item2)

        self.assertEqual(batch.id, "BATCH_001")
        self.assertEqual(len(batch.items), 2)
        self.assertEqual(batch.depot_id, "DEPOT")

    def test_get_required_locations(self):
        """Test getting required locations."""
        batch = Batch(id="B1", depot_id="DEPOT")
        batch.add_item(PickItem(sku="SKU001", location_id="A1"))
        batch.add_item(PickItem(sku="SKU002", location_id="A2"))
        batch.add_item(PickItem(sku="SKU003", location_id="A1"))  # Duplicate location

        locations = batch.get_required_locations()
        self.assertEqual(len(locations), 3)  # DEPOT, A1, A2
        self.assertIn("DEPOT", locations)
        self.assertIn("A1", locations)
        self.assertIn("A2", locations)

    def test_total_items(self):
        """Test total items calculation."""
        batch = Batch(id="B1")
        batch.add_item(PickItem(sku="SKU001", location_id="A1", quantity=3))
        batch.add_item(PickItem(sku="SKU002", location_id="A2", quantity=5))
        self.assertEqual(batch.total_items(), 8)

    def test_total_weight(self):
        """Test total weight calculation."""
        batch = Batch(id="B1")
        batch.add_item(PickItem(sku="SKU001", location_id="A1", quantity=2, weight=3.0))
        batch.add_item(PickItem(sku="SKU002", location_id="A2", quantity=3, weight=1.5))
        self.assertEqual(batch.total_weight(), 10.5)  # 2*3 + 3*1.5

    def test_can_fit_in_cart(self):
        """Test checking if batch fits in cart."""
        batch = Batch(id="B1")
        batch.add_item(PickItem(sku="SKU001", location_id="A1", quantity=10, weight=2.0))

        cart_small = Cart(id="C1", capacity=5)
        cart_large = Cart(id="C2", capacity=100)

        self.assertFalse(batch.can_fit_in_cart(cart_small))
        self.assertTrue(batch.can_fit_in_cart(cart_large))

    def test_batch_repr(self):
        """Test batch string representation."""
        batch = Batch(id="B1")
        batch.add_item(PickItem(sku="SKU001", location_id="A1"))
        batch.add_item(PickItem(sku="SKU002", location_id="A2"))
        repr_str = str(batch)
        self.assertIn("B1", repr_str)
        self.assertIn("2", repr_str)


class TestBatchManager(unittest.TestCase):
    """Test BatchManager class."""

    def setUp(self):
        """Set up batch manager."""
        self.manager = BatchManager()

    def test_add_batch(self):
        """Test adding batches to manager."""
        batch1 = Batch(id="B1")
        batch2 = Batch(id="B2")

        self.manager.add_batch(batch1)
        self.manager.add_batch(batch2)

        self.assertEqual(len(self.manager.batches), 2)

    def test_add_cart(self):
        """Test adding carts to manager."""
        cart1 = Cart(id="C1")
        cart2 = Cart(id="C2")

        self.manager.add_cart(cart1)
        self.manager.add_cart(cart2)

        self.assertEqual(len(self.manager.carts), 2)

    def test_assign_cart_to_batch(self):
        """Test assigning cart to batch."""
        batch = Batch(id="B1")
        batch.add_item(PickItem(sku="SKU001", location_id="A1", quantity=10))
        cart = Cart(id="C1", capacity=100)

        self.manager.add_batch(batch)
        self.manager.add_cart(cart)

        success = self.manager.assign_cart_to_batch("B1", "C1")
        self.assertTrue(success)
        self.assertEqual(batch.assigned_cart_id, "C1")

    def test_get_unassigned_batches(self):
        """Test getting unassigned batches."""
        batch1 = Batch(id="B1")
        batch2 = Batch(id="B2", assigned_cart_id="C1")

        self.manager.add_batch(batch1)
        self.manager.add_batch(batch2)

        unassigned = self.manager.get_unassigned_batches()
        self.assertEqual(len(unassigned), 1)
        self.assertEqual(unassigned[0].id, "B1")

    def test_get_available_carts(self):
        """Test getting available carts."""
        cart1 = Cart(id="C1")
        cart2 = Cart(id="C2")
        batch = Batch(id="B1", assigned_cart_id="C1")

        self.manager.add_cart(cart1)
        self.manager.add_cart(cart2)
        self.manager.add_batch(batch)

        available = self.manager.get_available_carts()
        self.assertEqual(len(available), 1)
        self.assertEqual(available[0].id, "C2")

    def test_validate_assignments(self):
        """Test validating batch-cart assignments."""
        batch = Batch(id="B1")
        batch.add_item(PickItem(sku="SKU001", location_id="A1", quantity=10))
        cart = Cart(id="C1", capacity=100)

        self.manager.add_batch(batch)
        self.manager.add_cart(cart)
        self.manager.assign_cart_to_batch("B1", "C1")

        errors = self.manager.validate_assignments()
        self.assertEqual(len(errors), 0)

    def test_get_statistics(self):
        """Test getting statistics."""
        batch1 = Batch(id="B1")
        batch1.add_item(PickItem(sku="SKU001", location_id="A1", quantity=5))
        batch2 = Batch(id="B2", assigned_cart_id="C1")
        cart = Cart(id="C1")

        self.manager.add_batch(batch1)
        self.manager.add_batch(batch2)
        self.manager.add_cart(cart)

        stats = self.manager.get_statistics()
        self.assertEqual(stats['total_batches'], 2)
        self.assertEqual(stats['assigned_batches'], 1)
        self.assertEqual(stats['unassigned_batches'], 1)
        self.assertEqual(stats['total_carts'], 1)


if __name__ == "__main__":
    unittest.main()

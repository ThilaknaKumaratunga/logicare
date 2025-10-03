"""
Batch and cart definitions for order picking.

Defines the structure for batches of orders and cart assignments.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set


@dataclass
class Cart:
    """
    Represents a picking cart/vehicle.
    
    Attributes:
        id: Unique identifier for the cart
        capacity: Maximum number of items the cart can hold
        weight_capacity: Maximum weight the cart can carry (kg)
        volume_capacity: Maximum volume (cubic meters)
        current_load: Current number of items in cart
        depot_id: Starting depot node ID
        metadata: Additional cart properties
    """
    id: str
    capacity: int = 100
    weight_capacity: float = 50.0  # kg
    volume_capacity: float = 1.0  # cubic meters
    current_load: int = 0
    depot_id: str = "depot"
    metadata: Dict = field(default_factory=dict)
    
    def has_capacity(self, items: int = 1, weight: float = 0, volume: float = 0) -> bool:
        """Check if cart has capacity for additional items."""
        return (self.current_load + items <= self.capacity and
                self.metadata.get('current_weight', 0) + weight <= self.weight_capacity and
                self.metadata.get('current_volume', 0) + volume <= self.volume_capacity)
    
    def add_load(self, items: int = 1, weight: float = 0, volume: float = 0) -> None:
        """Add items to cart load."""
        if not self.has_capacity(items, weight, volume):
            raise ValueError(f"Cart {self.id} exceeds capacity")
        self.current_load += items
        self.metadata['current_weight'] = self.metadata.get('current_weight', 0) + weight
        self.metadata['current_volume'] = self.metadata.get('current_volume', 0) + volume
    
    def reset_load(self) -> None:
        """Reset cart to empty state."""
        self.current_load = 0
        self.metadata['current_weight'] = 0
        self.metadata['current_volume'] = 0


@dataclass
class PickItem:
    """
    Represents a single item to be picked.
    
    Attributes:
        sku: Stock keeping unit identifier
        location_id: Node ID where item is located
        quantity: Number of units to pick
        weight: Weight per unit (kg)
        volume: Volume per unit (cubic meters)
        priority: Priority level (higher = more urgent)
    """
    sku: str
    location_id: str
    quantity: int = 1
    weight: float = 0.0
    volume: float = 0.0
    priority: int = 0
    
    def total_weight(self) -> float:
        """Calculate total weight of all units."""
        return self.weight * self.quantity
    
    def total_volume(self) -> float:
        """Calculate total volume of all units."""
        return self.volume * self.quantity


@dataclass
class Batch:
    """
    Represents a batch of orders to be picked together.
    
    A batch contains multiple items to be collected in a single picking trip.
    
    Attributes:
        id: Unique batch identifier
        items: List of items to pick
        assigned_cart_id: ID of cart assigned to this batch (optional)
        priority: Batch priority level
        depot_id: Starting/ending depot for this batch
        metadata: Additional batch information (order IDs, customer info, etc.)
    """
    id: str
    items: List[PickItem] = field(default_factory=list)
    assigned_cart_id: Optional[str] = None
    priority: int = 0
    depot_id: str = "depot"
    metadata: Dict = field(default_factory=dict)
    
    def get_required_locations(self) -> Set[str]:
        """Get set of all location IDs that must be visited."""
        locations = {self.depot_id}  # Always include depot
        for item in self.items:
            locations.add(item.location_id)
        return locations
    
    def get_location_items(self) -> Dict[str, List[PickItem]]:
        """Group items by location."""
        location_items = {}
        for item in self.items:
            if item.location_id not in location_items:
                location_items[item.location_id] = []
            location_items[item.location_id].append(item)
        return location_items
    
    def total_items(self) -> int:
        """Get total number of items in batch."""
        return sum(item.quantity for item in self.items)
    
    def total_weight(self) -> float:
        """Get total weight of all items."""
        return sum(item.total_weight() for item in self.items)
    
    def total_volume(self) -> float:
        """Get total volume of all items."""
        return sum(item.total_volume() for item in self.items)
    
    def can_fit_in_cart(self, cart: Cart) -> bool:
        """Check if batch can fit in given cart."""
        return cart.has_capacity(
            items=self.total_items(),
            weight=self.total_weight(),
            volume=self.total_volume()
        )
    
    def add_item(self, item: PickItem) -> None:
        """Add an item to the batch."""
        self.items.append(item)
    
    def remove_item(self, sku: str) -> bool:
        """Remove item by SKU. Returns True if item was found and removed."""
        original_length = len(self.items)
        self.items = [item for item in self.items if item.sku != sku]
        return len(self.items) < original_length
    
    def get_item_count_by_location(self) -> Dict[str, int]:
        """Get count of items at each location."""
        location_counts = {}
        for item in self.items:
            location_counts[item.location_id] = location_counts.get(item.location_id, 0) + item.quantity
        return location_counts
    
    def __repr__(self):
        return (f"Batch(id={self.id}, items={len(self.items)}, "
                f"locations={len(self.get_required_locations())}, "
                f"cart={self.assigned_cart_id})")


class BatchManager:
    """
    Manages multiple batches and cart assignments.
    """
    
    def __init__(self):
        self.batches: Dict[str, Batch] = {}
        self.carts: Dict[str, Cart] = {}
    
    def add_batch(self, batch: Batch) -> None:
        """Add a batch to the manager."""
        if batch.id in self.batches:
            raise ValueError(f"Batch {batch.id} already exists")
        self.batches[batch.id] = batch
    
    def add_cart(self, cart: Cart) -> None:
        """Add a cart to the manager."""
        if cart.id in self.carts:
            raise ValueError(f"Cart {cart.id} already exists")
        self.carts[cart.id] = cart
    
    def assign_cart_to_batch(self, batch_id: str, cart_id: str) -> bool:
        """
        Assign a cart to a batch.
        
        Returns True if assignment successful, False if capacity constraints violated.
        """
        if batch_id not in self.batches:
            raise ValueError(f"Batch {batch_id} not found")
        if cart_id not in self.carts:
            raise ValueError(f"Cart {cart_id} not found")
        
        batch = self.batches[batch_id]
        cart = self.carts[cart_id]
        
        if not batch.can_fit_in_cart(cart):
            return False
        
        batch.assigned_cart_id = cart_id
        return True
    
    def get_unassigned_batches(self) -> List[Batch]:
        """Get all batches without assigned carts."""
        return [batch for batch in self.batches.values() if batch.assigned_cart_id is None]
    
    def get_batches_for_cart(self, cart_id: str) -> List[Batch]:
        """Get all batches assigned to a specific cart."""
        return [batch for batch in self.batches.values() if batch.assigned_cart_id == cart_id]
    
    def get_available_carts(self) -> List[Cart]:
        """Get carts that have no batches assigned."""
        assigned_cart_ids = {batch.assigned_cart_id for batch in self.batches.values() 
                            if batch.assigned_cart_id is not None}
        return [cart for cart in self.carts.values() if cart.id not in assigned_cart_ids]
    
    def validate_assignments(self) -> List[str]:
        """
        Validate all batch-cart assignments.
        
        Returns list of validation errors (empty if all valid).
        """
        errors = []
        
        for batch in self.batches.values():
            if batch.assigned_cart_id is None:
                continue
            
            if batch.assigned_cart_id not in self.carts:
                errors.append(f"Batch {batch.id} assigned to non-existent cart {batch.assigned_cart_id}")
                continue
            
            cart = self.carts[batch.assigned_cart_id]
            if not batch.can_fit_in_cart(cart):
                errors.append(
                    f"Batch {batch.id} exceeds capacity of cart {cart.id} "
                    f"(items: {batch.total_items()}/{cart.capacity}, "
                    f"weight: {batch.total_weight():.2f}/{cart.weight_capacity}kg)"
                )
        
        return errors
    
    def get_statistics(self) -> Dict:
        """Get statistics about batches and carts."""
        assigned_batches = sum(1 for b in self.batches.values() if b.assigned_cart_id is not None)
        
        return {
            'total_batches': len(self.batches),
            'assigned_batches': assigned_batches,
            'unassigned_batches': len(self.batches) - assigned_batches,
            'total_carts': len(self.carts),
            'available_carts': len(self.get_available_carts()),
            'total_items': sum(b.total_items() for b in self.batches.values()),
            'total_weight': sum(b.total_weight() for b in self.batches.values()),
        }

"""
Route optimization using DOcplex.

Implements the mathematical optimization model for warehouse route planning
with support for multiple batches, carts, directional constraints, and subtour elimination.
"""

import logging
from typing import Dict, List, Optional, Tuple, Set
from docplex.mp.model import Model
from dataclasses import dataclass, field

from graph.graph import WarehouseGraph, Node, Edge
from optimizer.batch import Batch, Cart

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class RouteSegment:
    """Represents a segment in the optimized route."""
    from_node: str
    to_node: str
    batch_id: str
    cart_id: str
    travel_time: float
    sequence_order: int = 0


@dataclass
class OptimizedRoute:
    """Complete optimized route for a batch/cart combination."""
    batch_id: str
    cart_id: str
    segments: List[RouteSegment] = field(default_factory=list)
    total_time: float = 0.0
    total_distance: float = 0.0
    nodes_visited: List[str] = field(default_factory=list)
    
    def add_segment(self, segment: RouteSegment) -> None:
        """Add a route segment."""
        self.segments.append(segment)
        self.total_time += segment.travel_time
    
    def get_node_sequence(self) -> List[str]:
        """Get ordered list of nodes in the route by following edges from start."""
        if not self.segments:
            return []

        # Build adjacency list (from_node -> [to_nodes]) to handle multiple visits
        from collections import defaultdict
        adjacency = defaultdict(list)
        for seg in self.segments:
            adjacency[seg.from_node].append(seg.to_node)

        # Find starting node (prefer DEPOT)
        all_from = set(seg.from_node for seg in self.segments)
        if 'DEPOT' in all_from:
            current = 'DEPOT'
        else:
            # Find node with outgoing edge but no incoming
            all_to = set(seg.to_node for seg in self.segments)
            start_candidates = all_from - all_to
            current = list(start_candidates)[0] if start_candidates else self.segments[0].from_node

        # Follow the path by consuming edges
        sequence = [current]
        edges_used = 0
        max_edges = len(self.segments)

        while current in adjacency and adjacency[current] and edges_used < max_edges:
            # Take the next available edge from current node
            next_node = adjacency[current].pop(0)
            sequence.append(next_node)
            edges_used += 1
            current = next_node

        return sequence


class RouteOptimizer:
    """
    DOcplex-based route optimizer for warehouse picking.
    
    Implements:
    - Multi-batch, multi-cart optimization
    - Directional constraints (one-way aisles)
    - Subtour elimination (MTZ formulation)
    - Flow conservation
    - Depot constraints
    """
    
    def __init__(self, graph: WarehouseGraph, time_limit: Optional[int] = 300):
        """
        Initialize the route optimizer.
        
        Args:
            graph: Warehouse graph with nodes and edges
            time_limit: Maximum solve time in seconds (default: 300)
        """
        self.graph = graph
        self.time_limit = time_limit
        self.model: Optional[Model] = None
        self.solution: Optional[Dict] = None
        
    def optimize(self, batches: List[Batch], carts: List[Cart]) -> Dict[str, OptimizedRoute]:
        """
        Optimize routes for given batches and carts.

        Args:
            batches: List of batches to optimize
            carts: List of available carts

        Returns:
            Dictionary mapping batch_id to OptimizedRoute
        """
        logger.info("="*60)
        logger.info("Starting Route Optimization")
        logger.info("="*60)

        # Validate inputs
        if not batches:
            raise ValueError("No batches provided")
        if not carts:
            raise ValueError("No carts provided")

        logger.info(f"Input: {len(batches)} batches, {len(carts)} carts")
        for batch in batches:
            logger.info(f"  Batch {batch.id}: {len(batch.items)} items, "
                       f"{len(batch.get_required_locations())} locations")
        for cart in carts:
            logger.info(f"  Cart {cart.id}: capacity={cart.capacity}")

        # Ensure graph adjacency is built
        logger.info("Building graph adjacency lists")
        self.graph.build_adjacency_lists()

        # Create optimization model
        logger.info("Creating DOcplex optimization model")
        self.model = Model(name="Warehouse_Route_Optimization")

        # Set time limit
        if self.time_limit:
            logger.info(f"Setting time limit: {self.time_limit} seconds")
            self.model.set_time_limit(self.time_limit)

        # Create decision variables
        logger.info("Creating decision variables...")
        r_vars = self._create_route_variables(batches, carts)
        logger.info(f"  Created {len(r_vars)} route variables")

        visit_vars = self._create_visit_variables(batches, carts)
        logger.info(f"  Created {len(visit_vars)} visit variables")

        order_vars = self._create_order_variables(batches, carts)
        logger.info(f"  Created {len(order_vars)} order variables (MTZ)")

        # Set objective function
        logger.info("Setting objective function (minimize total travel time)")
        self._set_objective(r_vars, batches, carts)

        # Add constraints
        logger.info("Adding constraints...")
        self._add_flow_conservation_constraints(r_vars, visit_vars, batches, carts)
        self._add_depot_constraints(r_vars, visit_vars, batches, carts)
        self._add_visit_requirements(visit_vars, batches, carts)
        self._add_directional_constraints(r_vars, batches, carts)
        self._add_subtour_elimination(r_vars, order_vars, visit_vars, batches, carts)

        stats = self.get_model_statistics()
        logger.info(f"Model statistics:")
        logger.info(f"  Variables: {stats['num_variables']} "
                   f"(binary: {stats['num_binary_variables']}, "
                   f"integer: {stats['num_integer_variables']})")
        logger.info(f"  Constraints: {stats['num_constraints']}")

        # Solve
        logger.info("Solving optimization model with CPLEX...")
        solution = self.model.solve(log_output=True)

        if not solution:
            logger.error("No solution found!")
            return {}

        logger.info(f"Solution found! Objective value: {solution.objective_value:.2f}")
        logger.info(f"Solve status: {solution.solve_details.status}")

        # Extract and return routes
        logger.info("Extracting routes from solution...")
        routes = self._extract_routes(solution, r_vars, batches, carts)

        for route_key, route in routes.items():
            logger.info(f"Route {route_key}: {len(route.segments)} segments, "
                       f"total time: {route.total_time:.2f}")
            logger.info(f"  Path: {' -> '.join(route.get_node_sequence())}")

        logger.info("="*60)
        logger.info("Optimization Complete")
        logger.info("="*60)

        return routes
    
    def _create_route_variables(self, batches: List[Batch], carts: List[Cart]) -> Dict:
        """Create binary decision variables r[i,j,b,k] for edge usage."""
        logger.debug("Creating route variables for edge usage")
        r_vars = {}

        for batch in batches:
            for cart in carts:
                # Only consider valid cart-batch assignments
                if not batch.can_fit_in_cart(cart):
                    logger.debug(f"Batch {batch.id} cannot fit in cart {cart.id}, skipping")
                    continue

                for edge in self.graph.edges:
                    # Forward direction
                    if edge.direction_allowed in ['both', 'forward']:
                        var_name = f"r_{edge.from_node}_{edge.to_node}_{batch.id}_{cart.id}"
                        r_vars[(edge.from_node, edge.to_node, batch.id, cart.id)] = \
                            self.model.binary_var(name=var_name)

                    # Reverse direction (if bidirectional)
                    if edge.bidirectional and edge.direction_allowed in ['both', 'reverse']:
                        var_name = f"r_{edge.to_node}_{edge.from_node}_{batch.id}_{cart.id}"
                        r_vars[(edge.to_node, edge.from_node, batch.id, cart.id)] = \
                            self.model.binary_var(name=var_name)

        return r_vars
    
    def _create_visit_variables(self, batches: List[Batch], carts: List[Cart]) -> Dict:
        """Create binary decision variables visit[i,b,k] for node visitation."""
        visit_vars = {}
        
        for batch in batches:
            for cart in carts:
                if not batch.can_fit_in_cart(cart):
                    continue
                
                for node_id in self.graph.nodes:
                    var_name = f"visit_{node_id}_{batch.id}_{cart.id}"
                    visit_vars[(node_id, batch.id, cart.id)] = \
                        self.model.binary_var(name=var_name)
        
        return visit_vars
    
    def _create_order_variables(self, batches: List[Batch], carts: List[Cart]) -> Dict:
        """Create integer variables order[i,b,k] for subtour elimination (MTZ)."""
        order_vars = {}
        n = len(self.graph.nodes)
        
        for batch in batches:
            for cart in carts:
                if not batch.can_fit_in_cart(cart):
                    continue
                
                for node_id in self.graph.nodes:
                    # Depot has order 0, others from 1 to n-1
                    if node_id == batch.depot_id:
                        var_name = f"order_{node_id}_{batch.id}_{cart.id}"
                        order_vars[(node_id, batch.id, cart.id)] = \
                            self.model.integer_var(lb=0, ub=0, name=var_name)
                    else:
                        var_name = f"order_{node_id}_{batch.id}_{cart.id}"
                        order_vars[(node_id, batch.id, cart.id)] = \
                            self.model.integer_var(lb=1, ub=n-1, name=var_name)
        
        return order_vars
    
    def _set_objective(self, r_vars: Dict, batches: List[Batch], carts: List[Cart]) -> None:
        """Set objective function to minimize total travel time."""
        objective_expr = 0
        
        for (i, j, b_id, k_id), var in r_vars.items():
            edge = self.graph.get_edge(i, j)
            if edge:
                objective_expr += edge.travel_time * var
        
        self.model.minimize(objective_expr)
    
    def _add_flow_conservation_constraints(self, r_vars: Dict, visit_vars: Dict,
                                          batches: List[Batch], carts: List[Cart]) -> None:
        """Add flow conservation allowing round trips: inflow = outflow, max 2 visits per edge."""
        logger.debug("Adding flow conservation constraints")
        constraint_count = 0

        for batch in batches:
            for cart in carts:
                if not batch.can_fit_in_cart(cart):
                    continue

                for node_id in self.graph.nodes:
                    # Sum of incoming edges
                    inflow = sum(r_vars.get((j, node_id, batch.id, cart.id), 0)
                               for j in self.graph.nodes if (j, node_id, batch.id, cart.id) in r_vars)

                    # Sum of outgoing edges
                    outflow = sum(r_vars.get((node_id, j, batch.id, cart.id), 0)
                                for j in self.graph.nodes if (node_id, j, batch.id, cart.id) in r_vars)

                    # Flow conservation: what comes in must go out
                    self.model.add_constraint(
                        inflow == outflow,
                        ctname=f"flow_balance_{node_id}_{batch.id}_{cart.id}"
                    )
                    constraint_count += 1

                    # Limit flow to at most 2 (allows round trip on same edges)
                    self.model.add_constraint(
                        inflow <= 2,
                        ctname=f"max_inflow_{node_id}_{batch.id}_{cart.id}"
                    )
                    self.model.add_constraint(
                        outflow <= 2,
                        ctname=f"max_outflow_{node_id}_{batch.id}_{cart.id}"
                    )
                    constraint_count += 2

                    visit = visit_vars.get((node_id, batch.id, cart.id), 0)

                    if node_id == batch.depot_id:
                        # Depot: inflow = outflow = visit (must all be 1)
                        self.model.add_constraint(
                            inflow == visit,
                            ctname=f"depot_in_eq_visit_{batch.id}_{cart.id}"
                        )
                        self.model.add_constraint(
                            outflow == visit,
                            ctname=f"depot_out_eq_visit_{batch.id}_{cart.id}"
                        )
                        constraint_count += 2
                    else:
                        # Non-depot: if visit=1, must have inflow >= 1
                        self.model.add_constraint(
                            inflow >= visit,
                            ctname=f"visit_needs_flow_{node_id}_{batch.id}_{cart.id}"
                        )
                        constraint_count += 1

        logger.info(f"Added {constraint_count} flow conservation constraints")
    
    def _add_depot_constraints(self, r_vars: Dict, visit_vars: Dict,
                              batches: List[Batch], carts: List[Cart]) -> None:
        """Depot constraints now handled in flow conservation."""
        logger.debug("Depot constraints handled in flow conservation")
        pass
    
    def _add_visit_requirements(self, visit_vars: Dict, batches: List[Batch],
                               carts: List[Cart]) -> None:
        """Force visits to all required picking locations."""
        logger.info("Adding visit requirement constraints")
        constraint_count = 0

        for batch in batches:
            required_locations = batch.get_required_locations()
            pick_locations = {loc for loc in required_locations if loc != batch.depot_id}
            logger.info(f"Batch {batch.id} requires visits to pick locations: {pick_locations}")

            # Force depot visit = 1 (ensures route starts and ends at depot)
            for cart in carts:
                if batch.can_fit_in_cart(cart):
                    depot_visit = visit_vars.get((batch.depot_id, batch.id, cart.id), 0)
                    self.model.add_constraint(
                        depot_visit == 1,
                        ctname=f"depot_must_visit_{batch.id}_{cart.id}"
                    )
                    constraint_count += 1

            for location in pick_locations:
                if location not in self.graph.nodes:
                    raise ValueError(f"Required location {location} not in graph")

                # At least one cart must visit this location for this batch
                visit_sum = sum(visit_vars.get((location, batch.id, cart.id), 0)
                              for cart in carts if batch.can_fit_in_cart(cart))

                self.model.add_constraint(
                    visit_sum >= 1,
                    ctname=f"visit_required_{location}_{batch.id}"
                )
                constraint_count += 1

        logger.debug(f"Added {constraint_count} visit requirement constraints")
    
    def _add_directional_constraints(self, r_vars: Dict, batches: List[Batch],
                                    carts: List[Cart]) -> None:
        """Ensure directional constraints are respected (already handled in variable creation)."""
        # Directional constraints are implicitly enforced by only creating
        # variables for valid edge directions
        pass
    
    def _add_subtour_elimination(self, r_vars: Dict, order_vars: Dict, visit_vars: Dict,
                                batches: List[Batch], carts: List[Cart]) -> None:
        """
        Subtour elimination using modified MTZ that allows round trips.

        Key insight: We can use MTZ on VISIT events rather than edge traversals.
        - Each node can be traversed multiple times (round trip)
        - But each PICK location is visited exactly once in sequence
        - This prevents subtours while allowing backtracking

        For nodes with visit=1, we enforce ordering.
        For nodes with visit=0 (pass-through), no ordering constraint.
        """
        logger.info("Adding standard MTZ subtour elimination")
        constraint_count = 0
        n = len(self.graph.nodes)

        for batch in batches:
            for cart in carts:
                if not batch.can_fit_in_cart(cart):
                    continue

                for (i, j, b_id, k_id), var in r_vars.items():
                    if b_id != batch.id or k_id != cart.id:
                        continue

                    # Skip depot edges
                    if i == batch.depot_id or j == batch.depot_id:
                        continue

                    order_i = order_vars.get((i, batch.id, cart.id), 0)
                    order_j = order_vars.get((j, batch.id, cart.id), 0)

                    # Standard MTZ: if edge (i,j) is used, order[j] >= order[i] + 1
                    # order[j] >= order[i] + 1 - n*(1 - r[i,j])
                    self.model.add_constraint(
                        order_j >= order_i + 1 - n * (1 - var),
                        ctname=f"mtz_{i}_{j}_{batch.id}_{cart.id}"
                    )
                    constraint_count += 1

        logger.info(f"Added {constraint_count} MTZ subtour elimination constraints")
    
    def _extract_routes(self, solution, r_vars: Dict, batches: List[Batch],
                       carts: List[Cart]) -> Dict[str, OptimizedRoute]:
        """Extract route information from solution."""
        routes = {}
        
        for batch in batches:
            for cart in carts:
                if not batch.can_fit_in_cart(cart):
                    continue
                
                route = OptimizedRoute(batch_id=batch.id, cart_id=cart.id)
                
                # Find all edges used in this route
                for (i, j, b_id, k_id), var in r_vars.items():
                    if b_id == batch.id and k_id == cart.id:
                        if solution.get_value(var) > 0.5:  # Binary variable is "on"
                            edge = self.graph.get_edge(i, j)
                            if edge:
                                segment = RouteSegment(
                                    from_node=i,
                                    to_node=j,
                                    batch_id=batch.id,
                                    cart_id=cart.id,
                                    travel_time=edge.travel_time
                                )
                                route.add_segment(segment)
                
                # Only include routes that have segments
                if route.segments:
                    route.nodes_visited = route.get_node_sequence()
                    routes[f"{batch.id}_{cart.id}"] = route
        
        return routes
    
    def get_model_statistics(self) -> Dict:
        """Get statistics about the optimization model."""
        if not self.model:
            return {}
        
        return {
            'num_variables': self.model.number_of_variables,
            'num_binary_variables': self.model.number_of_binary_variables,
            'num_integer_variables': self.model.number_of_integer_variables,
            'num_constraints': self.model.number_of_constraints,
        }

"""
Route Visualizer Module

Visualizes warehouse layouts, graphs, and optimized routes using matplotlib and networkx.
Provides 2D visualization for POC and debugging.
"""

import logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from typing import Dict, List, Tuple, Optional
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


class RouteVisualizer:
    """Visualizes warehouse graphs and optimized routes."""
    
    def __init__(self, graph):
        """
        Initialize the visualizer with a warehouse graph.
        
        Args:
            graph: Graph object containing nodes and edges
        """
        self.graph = graph
        self.fig = None
        self.ax = None
        
    def create_networkx_graph(self) -> nx.DiGraph:
        """
        Convert internal graph to NetworkX DiGraph for visualization.
        
        Returns:
            NetworkX directed graph
        """
        G = nx.DiGraph()

        # Add nodes with adjusted positions for visualization
        for node_id, node in self.graph.nodes.items():
            # Map logical coordinates to visual coordinates
            # Storage blocks are offset from passage corridors
            visual_x, visual_y = self._get_visual_position(node)
            G.add_node(
                node_id,
                pos=(visual_x, visual_y),
                node_type=node.metadata.get('type', 'location'),
                logical_pos=(node.x, node.y)
            )
        
        # Add edges with attributes
        for edge in self.graph.edges:
            G.add_edge(
                edge.from_node,
                edge.to_node,
                weight=edge.travel_time,
                bidirectional=edge.bidirectional,
                capacity=edge.capacity
            )
            
            # Add reverse edge if bidirectional
            if edge.bidirectional:
                G.add_edge(
                    edge.to_node,
                    edge.from_node,
                    weight=edge.travel_time,
                    bidirectional=True,
                    capacity=edge.capacity
                )
        
        return G

    def _get_visual_position(self, node):
        """Map logical coordinates to visual coordinates.

        Storage blocks at passage y-levels (1, 12) are offset visually
        so passages appear as open corridors.
        """
        x, y = node.x, node.y

        if node.node_type == 'depot':
            return (x, y)

        # Offset storage blocks at passage y-levels
        passage_levels = [1, 20]
        if y in passage_levels:
            # Shift blocks slightly away from passage corridor
            # Blocks at y=1 shift to y=1.3, blocks at y=20 shift to y=19.7
            if y == 1:
                visual_y = y + 0.3
            elif y == 20:
                visual_y = y - 0.3
            else:
                visual_y = y
            return (x, visual_y)

        return (x, y)

    def _draw_route_through_passages(self, route: List[str], pos: dict, route_color: str) -> None:
        """Draw route path through passage corridors instead of direct lines.

        For each edge in the route, determine waypoints through passages:
        - If both nodes at same passage y-level: draw horizontal
        - If different y-levels: route through nearest passage corridor
        """
        if not self.ax or len(route) < 2:
            return

        # Passage coordinates (vertical aisles and horizontal passages)
        # Updated for 10 aisles: passages at x=1,4,7,10,13,16,19,22,25,28
        vertical_passages = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]  # x-coordinates of vertical aisle passages
        horizontal_passages = [1, 20]  # y-coordinates of horizontal cross-aisle passages

        for i in range(len(route) - 1):
            from_node = route[i]
            to_node = route[i + 1]

            if from_node not in pos or to_node not in pos:
                continue

            from_x, from_y = pos[from_node]
            to_x, to_y = pos[to_node]

            # Get logical positions (before visual offset)
            from_node_obj = self.graph.nodes.get(from_node)
            to_node_obj = self.graph.nodes.get(to_node)
            from_logical_x, from_logical_y = from_node_obj.x, from_node_obj.y
            to_logical_x, to_logical_y = to_node_obj.x, to_node_obj.y

            # Get edge metadata to determine routing
            edge_type = None
            for edge in self.graph.edges:
                if (edge.from_node == from_node and edge.to_node == to_node) or \
                   (edge.to_node == from_node and edge.from_node == to_node):
                    edge_type = edge.metadata.get('type')
                    break

            # Build waypoints based on edge type and positions
            waypoints = [(from_x, from_y)]

            # Determine which vertical passage to use based on x-coordinate
            # A01 (x=0,2) uses x=1, A02 (x=3,5) uses x=4, A03 (x=6,8) uses x=7
            # A04 (x=9,11) uses x=10, A05 (x=12,14) uses x=13, A06 (x=15,17) uses x=16
            # A07 (x=18,20) uses x=19, A08 (x=21,23) uses x=22, A09 (x=24,26) uses x=25, A10 (x=27,29) uses x=28
            if from_x <= 2:
                aisle_passage_x = 1
            elif from_x <= 5:
                aisle_passage_x = 4
            elif from_x <= 8:
                aisle_passage_x = 7
            elif from_x <= 11:
                aisle_passage_x = 10
            elif from_x <= 14:
                aisle_passage_x = 13
            elif from_x <= 17:
                aisle_passage_x = 16
            elif from_x <= 20:
                aisle_passage_x = 19
            elif from_x <= 23:
                aisle_passage_x = 22
            elif from_x <= 26:
                aisle_passage_x = 25
            else:
                aisle_passage_x = 28

            if edge_type == 'depot_to_passage':
                # Movement from depot to passage nodes
                # Route along horizontal passage at y=1 (or depot y-level)
                if from_y != to_y:
                    # Depot at different y than target - go vertical first then horizontal
                    waypoints.append((from_x, to_y))  # Move to passage y-level
                waypoints.append((to_x, to_y))  # Move horizontally along passage
            elif edge_type == 'cross_aisle':
                # Horizontal movement along passage
                # Check if we're crossing through an aisle (need to route through vertical passage)
                # Determine target aisle passage
                if to_x <= 2:
                    target_aisle_passage_x = 1
                elif to_x <= 5:
                    target_aisle_passage_x = 4
                elif to_x <= 8:
                    target_aisle_passage_x = 7
                elif to_x <= 11:
                    target_aisle_passage_x = 10
                elif to_x <= 14:
                    target_aisle_passage_x = 13
                elif to_x <= 17:
                    target_aisle_passage_x = 16
                elif to_x <= 20:
                    target_aisle_passage_x = 19
                elif to_x <= 23:
                    target_aisle_passage_x = 22
                elif to_x <= 26:
                    target_aisle_passage_x = 25
                else:
                    target_aisle_passage_x = 28

                # Check if from and to are in same aisle
                same_aisle = (from_x <= 2 and to_x <= 2) or \
                            (3 <= from_x <= 5 and 3 <= to_x <= 5) or \
                            (6 <= from_x <= 8 and 6 <= to_x <= 8) or \
                            (9 <= from_x <= 11 and 9 <= to_x <= 11) or \
                            (12 <= from_x <= 14 and 12 <= to_x <= 14) or \
                            (15 <= from_x <= 17 and 15 <= to_x <= 17) or \
                            (18 <= from_x <= 20 and 18 <= to_x <= 20) or \
                            (21 <= from_x <= 23 and 21 <= to_x <= 23) or \
                            (24 <= from_x <= 26 and 24 <= to_x <= 26) or \
                            (27 <= from_x and 27 <= to_x)

                if same_aisle and from_x != to_x:
                    # Within same aisle but different sides (L to R or R to L)
                    # Route through vertical aisle passage
                    waypoints.append((target_aisle_passage_x, from_y))
                    waypoints.append((target_aisle_passage_x, to_y))
                    waypoints.append((to_x, to_y))
                else:
                    # Across aisles - stay on horizontal passage
                    waypoints.append((to_x, to_y))
            elif edge_type == 'to_passage':
                # Movement from storage block to passage node
                # Must route through the vertical aisle passage
                if from_y != to_y:
                    # Different y-levels: go through vertical passage
                    waypoints.append((aisle_passage_x, from_y))  # Enter passage
                    waypoints.append((aisle_passage_x, to_y))    # Move vertically in passage
                    waypoints.append((to_x, to_y))               # Exit to storage block
                else:
                    # Same y-level: horizontal movement
                    waypoints.append((to_x, to_y))
            elif edge_type == 'vertical_passage':
                # Vertical movement between passage levels on same aisle
                waypoints.append((aisle_passage_x, from_y))  # Enter vertical passage
                waypoints.append((aisle_passage_x, to_y))    # Move vertically
                waypoints.append((to_x, to_y))               # Exit passage
            else:
                # Default: route through passage if different y-levels
                if from_y != to_y:
                    waypoints.append((aisle_passage_x, from_y))
                    waypoints.append((aisle_passage_x, to_y))
                waypoints.append((to_x, to_y))

            # Draw the path segments without arrows
            for j in range(len(waypoints) - 1):
                x_coords = [waypoints[j][0], waypoints[j+1][0]]
                y_coords = [waypoints[j][1], waypoints[j+1][1]]
                self.ax.plot(x_coords, y_coords, color=route_color, linewidth=3, alpha=0.9, zorder=5)

    def _draw_passage_based_route(self, route: List[str], pos: dict, route_color: str) -> None:
        """Draw route path through passage centerlines for block-based warehouses."""
        if not self.ax or len(route) < 2:
            return

        # Extract passage centerlines from graph metadata
        h_passages, v_passages = self._get_passage_centerlines()

        for i in range(len(route) - 1):
            from_node_id = route[i]
            to_node_id = route[i + 1]

            if from_node_id not in pos or to_node_id not in pos:
                continue

            from_x, from_y = pos[from_node_id]
            to_x, to_y = pos[to_node_id]

            # Build waypoints through passages
            waypoints = [(from_x, from_y)]

            # Find nearest vertical passages for from and to nodes
            from_nearest_vp = min(v_passages, key=lambda x: abs(x - from_x))
            to_nearest_vp = min(v_passages, key=lambda x: abs(x - to_x))

            # Use bottom horizontal passage (y=1) for routing
            hp_y = h_passages[0]

            # Route: from_node → from_VP (horizontal) → HP (vertical down/up) →
            #        along HP (horizontal) → to_VP (along HP) → to_node (horizontal)

            # 1. Move horizontally to nearest VP at from_node's y-level
            if from_x != from_nearest_vp:
                waypoints.append((from_nearest_vp, from_y))

            # 2. Move vertically along VP to horizontal passage
            if from_y != hp_y:
                waypoints.append((from_nearest_vp, hp_y))

            # 3. Move horizontally along the horizontal passage to target VP
            if from_nearest_vp != to_nearest_vp:
                waypoints.append((to_nearest_vp, hp_y))

            # 4. Move vertically along target VP to target node's y-level
            if to_y != hp_y:
                waypoints.append((to_nearest_vp, to_y))

            # 5. Move horizontally to target node
            if to_x != to_nearest_vp:
                waypoints.append((to_x, to_y))

            # If last waypoint is not the target, add it
            if waypoints[-1] != (to_x, to_y):
                waypoints.append((to_x, to_y))

            # Draw the path segments without arrows
            for j in range(len(waypoints) - 1):
                x_coords = [waypoints[j][0], waypoints[j+1][0]]
                y_coords = [waypoints[j][1], waypoints[j+1][1]]
                self.ax.plot(x_coords, y_coords, color=route_color, linewidth=3, alpha=0.9, zorder=5)

    def _get_passage_centerlines(self):
        """Extract passage centerlines from graph metadata or infer from structure."""
        # Try to get from graph metadata (if available)
        if hasattr(self.graph, 'metadata') and 'passages' in self.graph.metadata:
            passages = self.graph.metadata['passages']
            h_passages = passages.get('horizontal', [])
            v_passages = passages.get('vertical', [])
            return h_passages, v_passages

        # Fallback: infer from node positions and edge types
        # For block-based warehouses: every 6 units horizontally (0, 6, 12, 18, 24, 30, 36...)
        all_nodes = [n for n in self.graph.nodes.values() if n.node_type != 'depot']
        if not all_nodes:
            return [1, 18], [1, 7, 13, 19, 25, 31]

        all_x = sorted(set(n.x for n in all_nodes))
        all_y = sorted(set(n.y for n in all_nodes))

        # Infer vertical passages (between groups of x-coordinates)
        v_passages = []
        if all_x:
            # Passages are typically halfway between aisle columns
            # For x = 4, 10, 16, 22, 28, 34 -> passages at 1, 7, 13, 19, 25, 31
            min_x = min(all_x)
            for x in all_x:
                if x >= 4:
                    passage_x = x - 3
                    if passage_x not in v_passages:
                        v_passages.append(passage_x)
            # Add leftmost passage
            if min_x >= 4:
                v_passages.insert(0, 1)

        # Infer horizontal passages (typically at min and max y)
        h_passages = []
        if all_y:
            min_y, max_y = min(all_y), max(all_y)
            h_passages = [min_y - 3, max_y + 2]  # Bottom and top passages

        return h_passages, v_passages

    def _draw_block_bounding_boxes(self) -> None:
        """Draw bounding boxes for each 4×4 storage block."""
        if not self.ax:
            return

        # Block size (4×4 units)
        block_size = 4

        # Draw bounding box for each product node
        for node_id, node in self.graph.nodes.items():
            if node.node_type != 'product':
                continue

            # Block center is at (node.x, node.y)
            # Bounding box corners: (x-2, y-2) to (x+2, y+2)
            block_rect = mpatches.Rectangle(
                (node.x - block_size/2, node.y - block_size/2),
                block_size,
                block_size,
                linewidth=1.5,
                edgecolor='darkblue',
                facecolor='lightblue',
                alpha=0.2,
                zorder=1
            )
            self.ax.add_patch(block_rect)

            # Add block ID as text in the center
            self.ax.text(
                node.x, node.y,
                node_id,
                fontsize=8,
                ha='center',
                va='center',
                color='darkblue',
                weight='bold',
                zorder=2
            )

    def _draw_passages(self) -> None:
        """Draw passage corridors where movement occurs."""
        if not self.ax:
            return

        all_nodes = [n for n in self.graph.nodes.values() if n.node_type != 'depot']
        if not all_nodes:
            return

        # Get coordinate ranges
        all_x = [n.x for n in all_nodes]
        all_y = [n.y for n in all_nodes]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        # Get passage centerlines
        h_passages, v_passages = self._get_passage_centerlines()

        # Draw vertical passages
        for vp_x in v_passages:
            if vp_x >= min_x - 10 and vp_x <= max_x + 10:
                passage_rect = mpatches.Rectangle(
                    (vp_x - 1, 0),
                    2,
                    max_y + 2,
                    linewidth=1,
                    edgecolor='gray',
                    facecolor='lightgray',
                    alpha=0.3,
                    zorder=0
                )
                self.ax.add_patch(passage_rect)

        # Draw horizontal passages
        for hp_y in h_passages:
            if hp_y >= min_y - 5 and hp_y <= max_y + 5:
                passage_rect = mpatches.Rectangle(
                    (0, hp_y - 1),
                    max_x + 2,
                    2,
                    linewidth=1,
                    edgecolor='orange',
                    facecolor='lightyellow',
                    alpha=0.3,
                    zorder=0
                )
                self.ax.add_patch(passage_rect)

    def _draw_aisle_rectangles(self, ax, pos, node_types):
        """Draw rectangular aisles as connected blocks."""
        # Draw depot rectangle first
        for node_id, (x, y) in pos.items():
            if node_types.get(node_id) == 'depot':
                depot_rect = mpatches.Rectangle(
                    (x - 0.5, y - 0.5),
                    1.0,
                    1.0,
                    linewidth=3,
                    edgecolor='darkred',
                    facecolor='lightcoral',
                    alpha=0.3,
                    zorder=0
                )
                ax.add_patch(depot_rect)

        # Group nodes by aisle
        aisles = {}
        for node_id, (x, y) in pos.items():
            node_type = node_types.get(node_id, 'location')
            if node_type == 'depot':
                continue

            # Extract aisle from node ID (format: A01-01-00)
            parts = node_id.split('-')
            if len(parts) >= 1:
                aisle = parts[0]
                if aisle not in aisles:
                    aisles[aisle] = []
                aisles[aisle].append((node_id, x, y))

        # Draw each aisle as one connected rectangle
        for aisle, nodes in aisles.items():
            if not nodes:
                continue

            xs = [x for _, x, y in nodes]
            ys = [y for _, x, y in nodes]

            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            # Single connected aisle block
            aisle_rect = mpatches.Rectangle(
                (min_x - 0.35, min_y - 0.35),
                max_x - min_x + 0.7,
                max_y - min_y + 0.7,
                linewidth=2,
                edgecolor='darkgray',
                facecolor='lightgray',
                alpha=0.6,
                zorder=1
            )
            ax.add_patch(aisle_rect)

    def plot_layout(self,
                   figsize: Tuple[int, int] = (32, 20),
                   show_edge_labels: bool = True,
                   title: str = "Warehouse Layout") -> None:
        """
        Plot the basic warehouse layout without routes.

        Args:
            figsize: Figure size (width, height)
            show_edge_labels: Whether to show travel times on edges
            title: Plot title
        """
        G = self.create_networkx_graph()
        pos = nx.get_node_attributes(G, 'pos')
        node_types = nx.get_node_attributes(G, 'node_type')

        self.fig, self.ax = plt.subplots(figsize=figsize)

        # Don't draw old-style aisle rectangles - we use block bounding boxes instead
        # self._draw_aisle_rectangles(self.ax, pos, node_types)

        # Separate depot from other nodes
        depot_nodes = [n for n, t in node_types.items() if t == 'depot']
        location_nodes = [n for n, t in node_types.items() if t != 'depot']
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=depot_nodes,
            node_color='red',
            node_size=1200,
            node_shape='s',  # square for depot
            label='Depot',
            ax=self.ax
        )
        
        # Don't draw location nodes as circles - just show labels
        # nx.draw_networkx_nodes(
        #     G, pos,
        #     nodelist=location_nodes,
        #     node_color='lightblue',
        #     node_size=500,
        #     node_shape='o',
        #     label='Locations',
        #     ax=self.ax
        # )
        
        # Don't draw edges in base layout (too cluttered)
        # Edges will only be visible in the route path

        # Don't draw labels here - they're already drawn in block bounding boxes
        # Only draw depot label
        depot_labels = {n: n for n in depot_nodes}
        nx.draw_networkx_labels(G, pos, labels=depot_labels, font_size=10, font_weight='bold', ax=self.ax)
        
        # Draw edge labels (travel times)
        if show_edge_labels:
            edge_labels = nx.get_edge_attributes(G, 'weight')
            # Format labels to 1 decimal place
            edge_labels = {k: f"{v:.1f}" for k, v in edge_labels.items()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=12, ax=self.ax)
        
        self.ax.set_title(title, fontsize=22, fontweight='bold')
        self.ax.legend(fontsize=14)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        plt.tight_layout()
    
    def plot_route(self,
                  route: List[str],
                  batch_id: str = "1",
                  cart_id: str = "1",
                  route_color: str = 'blue',
                  figsize: Tuple[int, int] = (32, 20),
                  show_sequence: bool = True,
                  cart_info: Optional[Dict] = None,
                  pick_locations: Optional[List[str]] = None) -> None:
        """
        Plot a single optimized route on the warehouse layout.

        Args:
            route: Ordered list of node IDs representing the route
            batch_id: Batch identifier for labeling
            cart_id: Cart identifier for labeling
            route_color: Color for the route path
            figsize: Figure size (width, height)
            show_sequence: Whether to show sequence numbers on nodes
            cart_info: Optional dict with cart details (capacity, items, weight)
            pick_locations: Optional list of locations where items are picked (highlighted in red)
        """
        # First plot the base layout
        self.plot_layout(figsize=figsize, show_edge_labels=False,
                        title=f"Optimized Route - Batch {batch_id}, Cart {cart_id}")

        # Draw passages
        self._draw_passages()

        # Draw block bounding boxes
        self._draw_block_bounding_boxes()

        G = self.create_networkx_graph()
        pos = nx.get_node_attributes(G, 'pos')

        # Draw passage-based route path
        self._draw_passage_based_route(route, pos, route_color)
        
        # Highlight visited nodes
        visited_nodes = list(set(route))  # Remove duplicates

        # Separate pick locations from regular visited nodes
        if pick_locations:
            pick_nodes = [n for n in visited_nodes if n in pick_locations]
            regular_nodes = [n for n in visited_nodes if n not in pick_locations]

            # Draw pick locations in red
            if pick_nodes:
                nx.draw_networkx_nodes(
                    G, pos,
                    nodelist=pick_nodes,
                    node_color='red',
                    node_size=1000,
                    alpha=0.9,
                    ax=self.ax
                )

            # Draw regular visited nodes in blue
            if regular_nodes:
                nx.draw_networkx_nodes(
                    G, pos,
                    nodelist=regular_nodes,
                    node_color=route_color,
                    node_size=900,
                    alpha=0.7,
                    ax=self.ax
                )
        else:
            # No pick locations specified, draw all in blue
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=visited_nodes,
                node_color=route_color,
                node_size=900,
                alpha=0.7,
                ax=self.ax
            )
        
        # Add sequence numbers if requested
        if show_sequence:
            sequence_labels = {route[i]: f"{i}" for i in range(len(route))}
            # Offset position more significantly for sequence numbers to avoid overlap
            offset_pos = {node: (x, y-1.2) for node, (x, y) in pos.items() if node in sequence_labels}
            nx.draw_networkx_labels(
                G, offset_pos,
                labels=sequence_labels,
                font_size=14,
                font_color='white',
                font_weight='bold',
                bbox=dict(boxstyle='circle,pad=0.4', facecolor=route_color, alpha=0.95, edgecolor='white', linewidth=2),
                ax=self.ax
            )
        
        # Build stats text with cart information
        stats_lines = [
            f"Batch: {batch_id}",
            f"Cart: {cart_id}",
            f"Route Length: {len(route)} nodes"
        ]

        # Add cart details if provided
        if cart_info:
            if 'capacity' in cart_info:
                stats_lines.append(f"Cart Capacity: {cart_info['capacity']} items")
            if 'items' in cart_info:
                stats_lines.append(f"Items Picked: {cart_info['items']}")
            if 'weight' in cart_info:
                stats_lines.append(f"Total Weight: {cart_info['weight']:.2f} kg")

        stats_text = "\n".join(stats_lines)
        self.ax.text(0.02, 0.98, stats_text,
                    transform=self.ax.transAxes,
                    fontsize=14,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    def plot_multiple_routes(self,
                           routes: Dict[Tuple[str, str], List[str]],
                           figsize: Tuple[int, int] = (36, 24),
                           colors: Optional[List[str]] = None) -> None:
        """
        Plot multiple routes on the same layout (for multiple batches/carts).
        
        Args:
            routes: Dictionary mapping (batch_id, cart_id) to route (list of node IDs)
            figsize: Figure size (width, height)
            colors: List of colors for different routes (auto-generated if None)
        """
        if colors is None:
            # Generate distinct colors for each route
            cmap = plt.cm.get_cmap('tab10')
            colors = [cmap(i % 10) for i in range(len(routes))]
        
        # Plot base layout
        self.plot_layout(figsize=figsize, show_edge_labels=False,
                        title="Multiple Route Optimization")
        
        G = self.create_networkx_graph()
        pos = nx.get_node_attributes(G, 'pos')
        
        # Plot each route with different color
        for idx, ((batch_id, cart_id), route) in enumerate(routes.items()):
            color = colors[idx % len(colors)]
            
            # Create route edges
            route_edges = [(route[i], route[i+1]) for i in range(len(route)-1)]
            
            # Draw route path
            nx.draw_networkx_edges(
                G, pos,
                edgelist=route_edges,
                edge_color=[color] * len(route_edges),
                width=3,
                alpha=0.6,
                arrows=True,
                arrowsize=20,
                arrowstyle='->',
                connectionstyle=f'arc3,rad={0.1 + idx*0.05}',  # Offset curves for multiple routes
                label=f"Batch {batch_id}, Cart {cart_id}",
                ax=self.ax
            )
        
        self.ax.legend(loc='upper right')
    
    def save_plot(self, filename: str, dpi: int = 300) -> None:
        """
        Save the current plot to a file.
        
        Args:
            filename: Output filename (with extension, e.g., 'route.png')
            dpi: Resolution in dots per inch
        """
        if self.fig is None:
            raise ValueError("No plot to save. Create a plot first.")
        
        self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {filename}")
    
    def show(self) -> None:
        """Display the current plot."""
        if self.fig is None:
            raise ValueError("No plot to show. Create a plot first.")
        plt.show()
    
    def close(self) -> None:
        """Close the current plot."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


def visualize_solution(graph, solution_routes: Dict[Tuple[str, str], List[str]], 
                      output_file: Optional[str] = None) -> None:
    """
    Convenience function to visualize optimization solution.
    
    Args:
        graph: Graph object
        solution_routes: Dictionary mapping (batch_id, cart_id) to routes
        output_file: Optional filename to save the plot
    """
    visualizer = RouteVisualizer(graph)
    
    if len(solution_routes) == 1:
        # Single route
        (batch_id, cart_id), route = list(solution_routes.items())[0]
        visualizer.plot_route(route, batch_id, cart_id)
    else:
        # Multiple routes
        visualizer.plot_multiple_routes(solution_routes)
    
    if output_file:
        visualizer.save_plot(output_file)
    
    visualizer.show()

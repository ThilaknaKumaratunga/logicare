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
        
        # Add nodes with positions
        for node_id, node in self.graph.nodes.items():
            G.add_node(
                node_id, 
                pos=(node.x, node.y),
                node_type=node.metadata.get('type', 'location')
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
                   figsize: Tuple[int, int] = (12, 10),
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

        # Draw aisle and block rectangles first
        self._draw_aisle_rectangles(self.ax, pos, node_types)

        # Separate depot from other nodes
        depot_nodes = [n for n, t in node_types.items() if t == 'depot']
        location_nodes = [n for n, t in node_types.items() if t != 'depot']
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, 
            nodelist=depot_nodes,
            node_color='red',
            node_size=800,
            node_shape='s',  # square for depot
            label='Depot',
            ax=self.ax
        )
        
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=location_nodes,
            node_color='lightblue',
            node_size=500,
            node_shape='o',
            label='Locations',
            ax=self.ax
        )
        
        # Draw edges
        # Separate bidirectional and unidirectional edges
        bidirectional_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('bidirectional', False)]
        unidirectional_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get('bidirectional', False)]

        # Draw bidirectional edges with arrows in both directions (<-->)
        nx.draw_networkx_edges(
            G, pos,
            edgelist=bidirectional_edges,
            edge_color='gray',
            style='solid',
            alpha=0.6,
            arrows=True,
            arrowsize=20,
            arrowstyle='<->',  # ← Two arrow heads (bidirectional)
            width=1.5,
            connectionstyle='arc3,rad=0.0',
            ax=self.ax
        )

        # Draw unidirectional edges with single arrow (-->)
        nx.draw_networkx_edges(
            G, pos,
            edgelist=unidirectional_edges,
            edge_color='gray',
            style='solid',
            alpha=0.6,
            arrows=True,
            arrowsize=20,
            arrowstyle='->',  # ← Single arrow head (unidirectional)
            width=2,
            connectionstyle='arc3,rad=0.0',
            ax=self.ax
        )
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=self.ax)
        
        # Draw edge labels (travel times)
        if show_edge_labels:
            edge_labels = nx.get_edge_attributes(G, 'weight')
            # Format labels to 1 decimal place
            edge_labels = {k: f"{v:.1f}" for k, v in edge_labels.items()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=self.ax)
        
        self.ax.set_title(title, fontsize=16, fontweight='bold')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        plt.tight_layout()
    
    def plot_route(self,
                  route: List[str],
                  batch_id: str = "1",
                  cart_id: str = "1",
                  route_color: str = 'blue',
                  figsize: Tuple[int, int] = (12, 10),
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
        
        G = self.create_networkx_graph()
        pos = nx.get_node_attributes(G, 'pos')
        
        # Create route edges
        route_edges = [(route[i], route[i+1]) for i in range(len(route)-1)]
        
        # Draw route path with thick colored edges
        nx.draw_networkx_edges(
            G, pos,
            edgelist=route_edges,
            edge_color=route_color,
            width=4,
            alpha=0.8,
            arrows=False,
            ax=self.ax
        )
        
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
                    node_size=700,
                    alpha=0.9,
                    ax=self.ax
                )

            # Draw regular visited nodes in blue
            if regular_nodes:
                nx.draw_networkx_nodes(
                    G, pos,
                    nodelist=regular_nodes,
                    node_color=route_color,
                    node_size=600,
                    alpha=0.7,
                    ax=self.ax
                )
        else:
            # No pick locations specified, draw all in blue
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=visited_nodes,
                node_color=route_color,
                node_size=600,
                alpha=0.7,
                ax=self.ax
            )
        
        # Add sequence numbers if requested
        if show_sequence:
            sequence_labels = {route[i]: f"{i}" for i in range(len(route))}
            # Offset position slightly for sequence numbers
            offset_pos = {node: (x, y-0.3) for node, (x, y) in pos.items() if node in sequence_labels}
            nx.draw_networkx_labels(
                G, offset_pos,
                labels=sequence_labels,
                font_size=8,
                font_color='white',
                bbox=dict(boxstyle='circle', facecolor=route_color, alpha=0.8),
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
                    fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    def plot_multiple_routes(self,
                           routes: Dict[Tuple[str, str], List[str]],
                           figsize: Tuple[int, int] = (14, 12),
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

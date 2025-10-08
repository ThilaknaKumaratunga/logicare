"""
Route Visualizer for DXF-based Warehouse Graphs

Visualizes warehouse layouts from DXF files showing:
- Passage corridors (gray lines with distances)
- Aisle storage blocks (yellow squares at centroids)
- Depot (orange diamond at centroid)
- Optimized routes through the warehouse
"""

import logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class RouteVisualizer:
    """Visualizes DXF-based warehouse graphs and optimized routes."""

    def __init__(self, graph):
        """
        Initialize the visualizer with a warehouse graph.

        Args:
            graph: WarehouseGraph object with nodes and edges
        """
        self.graph = graph
        self.fig = None
        self.ax = None

    def create_networkx_graph(self) -> nx.Graph:
        """Convert internal graph to NetworkX Graph for visualization."""
        G = nx.Graph()

        # Add nodes with positions
        for node_id, node in self.graph.nodes.items():
            G.add_node(
                node_id,
                pos=(node.x, node.y),
                node_type=node.node_type,
                metadata=node.metadata
            )

        # Add edges
        for edge in self.graph.edges:
            G.add_edge(
                edge.from_node,
                edge.to_node,
                weight=edge.travel_time,
                distance=edge.distance
            )

        return G

    def plot_layout(self, figsize: Tuple[int, int] = (16, 12), title: str = "Warehouse Layout") -> None:
        """
        Plot the basic warehouse layout.

        Args:
            figsize: Figure size (width, height)
            title: Plot title
        """
        G = self.create_networkx_graph()
        pos = nx.get_node_attributes(G, 'pos')

        self.fig, self.ax = plt.subplots(figsize=figsize)

        # Separate nodes by type
        passage_nodes = []
        aisle_nodes = []
        depot_nodes = []

        for node_id, data in G.nodes(data=True):
            if data['node_type'] == 'depot':
                depot_nodes.append(node_id)
            elif data['node_type'] == 'product':
                aisle_nodes.append(node_id)
            else:
                passage_nodes.append(node_id)

        # Draw passage edges (gray solid lines with weights)
        passage_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > 0]
        if passage_edges:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=passage_edges,
                edge_color='gray',
                width=2,
                alpha=0.6,
                ax=self.ax
            )

        # Draw connection edges (green dashed lines, weight=0)
        connection_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] == 0]
        if connection_edges:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=connection_edges,
                edge_color='green',
                width=1.5,
                alpha=0.4,
                style='dashed',
                ax=self.ax
            )

        # Draw nodes
        if passage_nodes:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=passage_nodes,
                node_color='lightblue',
                node_size=100,
                node_shape='o',
                label='Passages',
                ax=self.ax
            )

        if aisle_nodes:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=aisle_nodes,
                node_color='yellow',
                node_size=800,
                node_shape='s',
                label='Aisles',
                edgecolors='black',
                linewidths=2,
                ax=self.ax
            )

        if depot_nodes:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=depot_nodes,
                node_color='orange',
                node_size=1200,
                node_shape='D',
                label='Depot',
                edgecolors='darkred',
                linewidths=3,
                ax=self.ax
            )

        # Draw labels for aisles and depot only
        aisle_labels = {n: n for n in aisle_nodes}
        depot_labels = {n: n for n in depot_nodes}
        all_labels = {**aisle_labels, **depot_labels}

        nx.draw_networkx_labels(
            G, pos,
            labels=all_labels,
            font_size=9,
            font_weight='bold',
            ax=self.ax
        )

        self.ax.set_title(title, fontsize=16, fontweight='bold')
        self.ax.legend(fontsize=10, loc='upper right')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        plt.tight_layout()

    def plot_route(self,
                  route: List[str],
                  batch_id: str = "1",
                  cart_id: str = "1",
                  route_color: str = 'blue',
                  figsize: Tuple[int, int] = (16, 12),
                  show_sequence: bool = True,
                  cart_info: Optional[Dict] = None,
                  pick_locations: Optional[List[str]] = None) -> None:
        """
        Plot an optimized route on the warehouse layout.

        Args:
            route: Ordered list of node IDs representing the route
            batch_id: Batch identifier for labeling
            cart_id: Cart identifier for labeling
            route_color: Color for the route path
            figsize: Figure size (width, height)
            show_sequence: Whether to show sequence numbers on nodes
            cart_info: Optional dict with cart details (capacity, items, weight)
            pick_locations: Optional list of locations where items are picked
        """
        # First plot the base layout
        self.plot_layout(figsize=figsize, title=f"Optimized Route - Batch {batch_id}, Cart {cart_id}")

        G = self.create_networkx_graph()
        pos = nx.get_node_attributes(G, 'pos')

        # Draw route path
        if len(route) > 1:
            route_edges = [(route[i], route[i+1]) for i in range(len(route)-1)]
            nx.draw_networkx_edges(
                G, pos,
                edgelist=route_edges,
                edge_color=route_color,
                width=4,
                alpha=0.8,
                arrows=True,
                arrowsize=20,
                arrowstyle='->',
                ax=self.ax
            )

        # Highlight visited nodes
        visited_nodes = list(set(route))

        # Separate pick locations from regular visited nodes
        if pick_locations:
            pick_nodes = [n for n in visited_nodes if n in pick_locations and n != 'DEPOT']
            regular_nodes = [n for n in visited_nodes if n not in pick_locations and n != 'DEPOT']

            # Draw pick locations in red
            if pick_nodes:
                nx.draw_networkx_nodes(
                    G, pos,
                    nodelist=pick_nodes,
                    node_color='red',
                    node_size=1000,
                    node_shape='s',
                    alpha=0.9,
                    edgecolors='darkred',
                    linewidths=3,
                    ax=self.ax
                )

            # Draw regular visited nodes
            if regular_nodes:
                nx.draw_networkx_nodes(
                    G, pos,
                    nodelist=regular_nodes,
                    node_color=route_color,
                    node_size=300,
                    alpha=0.7,
                    ax=self.ax
                )
        else:
            # No pick locations specified
            non_depot = [n for n in visited_nodes if n != 'DEPOT']
            if non_depot:
                nx.draw_networkx_nodes(
                    G, pos,
                    nodelist=non_depot,
                    node_color=route_color,
                    node_size=600,
                    alpha=0.7,
                    ax=self.ax
                )

        # Add sequence numbers if requested
        if show_sequence:
            sequence_labels = {route[i]: f"{i}" for i in range(len(route))}
            nx.draw_networkx_labels(
                G, pos,
                labels=sequence_labels,
                font_size=10,
                font_color='white',
                font_weight='bold',
                bbox=dict(boxstyle='circle,pad=0.3', facecolor=route_color, alpha=0.9, edgecolor='white', linewidth=2),
                ax=self.ax
            )

        # Build stats text
        stats_lines = [
            f"Batch: {batch_id}",
            f"Cart: {cart_id}",
            f"Route Length: {len(route)} stops"
        ]

        if cart_info:
            if 'items' in cart_info:
                stats_lines.append(f"Items: {cart_info['items']}")
            if 'weight' in cart_info:
                stats_lines.append(f"Weight: {cart_info['weight']:.1f} kg")

        stats_text = "\n".join(stats_lines)
        self.ax.text(
            0.02, 0.98, stats_text,
            transform=self.ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9)
        )

    def save_plot(self, filename: str, dpi: int = 300) -> None:
        """
        Save the current plot to a file.

        Args:
            filename: Output filename (with extension)
            dpi: Resolution in dots per inch
        """
        if self.fig is None:
            raise ValueError("No plot to save. Create a plot first.")

        self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        logger.info(f"Plot saved to {filename}")

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

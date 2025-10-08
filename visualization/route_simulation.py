"""
Route Simulation Module

Animates warehouse cart movement along optimized routes using SimPy and Matplotlib.
Creates animated GIF or video showing cart moving through warehouse with DXF polylines.
"""

import logging
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon, Circle
import networkx as nx
import ezdxf
import re
from typing import List, Dict, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class RouteSimulator:
    """Simulates and animates cart movement through warehouse routes."""

    def __init__(self, graph, dxf_file: str, route_sequence: List[str], cart_id: str = "CART_001", batch_id: str = "ORDER_001"):
        """
        Initialize the route simulator.

        Args:
            graph: WarehouseGraph object
            dxf_file: Path to DXF file for polygons
            route_sequence: Ordered list of node IDs for the route
            cart_id: Cart identifier
            batch_id: Batch identifier
        """
        self.graph = graph
        self.dxf_file = dxf_file
        self.route_sequence = route_sequence
        self.cart_id = cart_id
        self.batch_id = batch_id
        self.dxf_polygons = None

        # Load DXF polygons
        self._load_dxf_polygons()

        # Build node positions
        self.node_positions = {node_id: (node.x, node.y) for node_id, node in graph.nodes.items()}

    def _load_dxf_polygons(self):
        """Load polylines from DXF file."""
        try:
            doc = ezdxf.readfile(self.dxf_file)
            msp = doc.modelspace()

            self.dxf_polygons = {
                'aisles': {},
                'depot': None,
                'boundary': None
            }

            aisle_pattern = re.compile(r'^A\d+-[RL]-\d+$')

            for entity in msp:
                if entity.dxftype() == 'LWPOLYLINE':
                    layer_name = entity.dxf.layer
                    points = [(p[0], p[1]) for p in entity.get_points()]

                    if aisle_pattern.match(layer_name):
                        self.dxf_polygons['aisles'][layer_name] = points
                    elif layer_name.lower() == 'depot':
                        self.dxf_polygons['depot'] = points
                    elif layer_name.lower() == 'boundary':
                        self.dxf_polygons['boundary'] = points

        except Exception as e:
            logger.warning(f"Could not load DXF polygons: {e}")

    def _interpolate_path(self, num_frames: int) -> List[Tuple[float, float]]:
        """
        Interpolate smooth path between route nodes.

        Args:
            num_frames: Total number of animation frames

        Returns:
            List of (x, y) coordinates for each frame
        """
        if len(self.route_sequence) < 2:
            return [self.node_positions[self.route_sequence[0]]] * num_frames

        # Get positions for each node in route
        route_positions = [self.node_positions[node_id] for node_id in self.route_sequence]

        # Calculate segment lengths
        segment_lengths = []
        for i in range(len(route_positions) - 1):
            x1, y1 = route_positions[i]
            x2, y2 = route_positions[i + 1]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            segment_lengths.append(length)

        total_length = sum(segment_lengths)

        # Interpolate positions
        interpolated_positions = []
        frames_per_segment = [int(num_frames * length / total_length) for length in segment_lengths]

        # Adjust to ensure total frames
        frames_per_segment[-1] += num_frames - sum(frames_per_segment)

        for i in range(len(route_positions) - 1):
            x1, y1 = route_positions[i]
            x2, y2 = route_positions[i + 1]
            n_frames = frames_per_segment[i]

            for j in range(n_frames):
                t = j / max(n_frames, 1)
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)
                interpolated_positions.append((x, y))

        # Add final position
        interpolated_positions.append(route_positions[-1])

        return interpolated_positions

    def simulate_route(self, output_path: str = "output/simulation.gif",
                      pick_locations: Optional[List[str]] = None,
                      duration: float = 10.0, fps: int = 30) -> None:
        """
        Simulate cart movement and save as animated GIF.

        Args:
            output_path: Path to save animation
            pick_locations: List of aisle IDs where items are picked
            duration: Animation duration in seconds
            fps: Frames per second
        """
        num_frames = int(duration * fps)
        path_positions = self._interpolate_path(num_frames)

        # Setup figure
        fig, ax = plt.subplots(figsize=(16, 12))

        # Draw static elements
        self._draw_warehouse_base(ax, pick_locations)

        # Create cart marker
        cart_marker = Circle((0, 0), radius=20, color='blue', zorder=10, alpha=0.8)
        ax.add_patch(cart_marker)

        # Create trail line
        trail_line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.5, zorder=5)

        # Title and info text
        title_text = ax.text(0.5, 0.98, '', transform=ax.transAxes,
                            fontsize=14, ha='center', va='top', fontweight='bold')
        info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                           fontsize=10, va='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

        # Store trail points
        trail_x, trail_y = [], []

        def init():
            """Initialize animation."""
            cart_marker.center = path_positions[0]
            trail_line.set_data([], [])
            return cart_marker, trail_line, title_text, info_text

        def animate(frame):
            """Update animation frame."""
            # Update cart position
            x, y = path_positions[frame]
            cart_marker.center = (x, y)

            # Update trail
            trail_x.append(x)
            trail_y.append(y)
            trail_line.set_data(trail_x, trail_y)

            # Find current node
            current_node = self._find_nearest_node(x, y)
            progress = (frame / num_frames) * 100

            # Update title
            title_text.set_text(f'Cart {self.cart_id} - Batch {self.batch_id}')

            # Update info
            info_text.set_text(f'Progress: {progress:.0f}%\nCurrent: {current_node}\nFrame: {frame}/{num_frames}')

            return cart_marker, trail_line, title_text, info_text

        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, init_func=init, frames=num_frames,
            interval=1000/fps, blit=True, repeat=True
        )

        # Save animation
        try:
            anim.save(output_path, writer='pillow', fps=fps, dpi=100)
            logger.info(f"Animation saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save animation: {e}")
            # Try saving as MP4 instead
            try:
                mp4_path = output_path.replace('.gif', '.mp4')
                anim.save(mp4_path, writer='ffmpeg', fps=fps, dpi=100)
                logger.info(f"Animation saved to {mp4_path}")
            except:
                logger.error("Could not save animation in any format")

        plt.close(fig)

    def _draw_warehouse_base(self, ax, pick_locations: Optional[List[str]] = None):
        """Draw static warehouse elements."""
        # Draw boundary
        if self.dxf_polygons and self.dxf_polygons['boundary']:
            boundary_poly = Polygon(
                self.dxf_polygons['boundary'],
                fill=False, edgecolor='black', linewidth=2, zorder=1
            )
            ax.add_patch(boundary_poly)

        # Draw depot
        if self.dxf_polygons and self.dxf_polygons['depot']:
            depot_poly = Polygon(
                self.dxf_polygons['depot'],
                fill=True, facecolor='orange', edgecolor='darkred',
                alpha=0.3, linewidth=2, zorder=2
            )
            ax.add_patch(depot_poly)

        # Draw aisles
        if self.dxf_polygons:
            for aisle_name, points in self.dxf_polygons['aisles'].items():
                # Determine if this is a pick location
                is_pick = pick_locations and aisle_name in pick_locations
                facecolor = 'red' if is_pick else 'yellow'
                alpha = 0.5 if is_pick else 0.3

                aisle_poly = Polygon(
                    points,
                    fill=True, facecolor=facecolor, edgecolor='black',
                    alpha=alpha, linewidth=1.5, zorder=2
                )
                ax.add_patch(aisle_poly)

                # Add label
                centroid_x = sum(p[0] for p in points) / len(points)
                centroid_y = sum(p[1] for p in points) / len(points)
                ax.text(centroid_x, centroid_y, aisle_name,
                       fontsize=8, ha='center', va='center', fontweight='bold')

        # Draw passage nodes
        for node_id, (x, y) in self.node_positions.items():
            node = self.graph.nodes[node_id]
            if node.node_type == 'junction':
                ax.plot(x, y, 'o', color='lightblue', markersize=3, zorder=3)

        # Setup axes
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('Warehouse Route Simulation', fontsize=16, fontweight='bold')

    def _find_nearest_node(self, x: float, y: float) -> str:
        """Find nearest node ID to given coordinates."""
        min_dist = float('inf')
        nearest_node = self.route_sequence[0]

        for node_id in self.route_sequence:
            nx, ny = self.node_positions[node_id]
            dist = np.sqrt((x - nx)**2 + (y - ny)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node_id

        return nearest_node

import math
import xml.etree.ElementTree as ET

import numpy as np
import networkx as nx
from scipy.interpolate import CubicSpline


class GlobalHDMapPlanner:
    """
    A global planner that:
      1. Parses a simplified OpenDRIVE (.xodr) file.
      2. Builds a lane-center graph by sampling the parametric road geometry.
      3. Uses A* to find a path from start to goal.
      4. Returns a smooth path and a simple velocity profile.
    """

    def __init__(self, xodr_file, sampling_resolution=1.0):
        """
        :param xodr_file: path to the OpenDRIVE HD map (.xodr).
        :param sampling_resolution: distance (meters) between samples when converting arcs/lines to discrete points.
        """
        self.xodr_file = xodr_file
        self.sampling_resolution = sampling_resolution

        # Directed graph of (x, y) nodes, edges store distance
        self.graph = nx.DiGraph()

        # Parse the roads & build the graph
        self._parse_opendrive()

    def _parse_opendrive(self):
        """
        Minimalistic parser that reads <road> <geometry> elements (arc/line),
        samples them, and builds a directed graph with edges between consecutive samples.
        """
        tree = ET.parse(self.xodr_file)
        root = tree.getroot()

        # We iterate over each <road> in the OpenDRIVE
        for road_elem in root.findall("road"):
            road_id = road_elem.get("id", "unknown")

            plan_view = road_elem.find("planView")
            if plan_view is None:
                continue

            # We'll collect all sampled (x, y) points from each geometry (arc/line)
            road_points = []

            for geom in plan_view.findall("geometry"):
                try:
                    # Starting reference along this geometry
                    sx = float(geom.get("x", 0.0))   # start X
                    sy = float(geom.get("y", 0.0))   # start Y
                    shdg = float(geom.get("hdg", 0.0))  # start heading (radians)
                    length = float(geom.get("length", 0.0))

                    # Child elements: <line/>, <arc curvature="..."/>, <spiral> ...
                    line_elem = geom.find("line")
                    arc_elem = geom.find("arc")
                    
                    if line_elem is not None:
                        # Sample points along a straight line
                        road_points.extend(
                            self._sample_line(sx, sy, shdg, length, self.sampling_resolution)
                        )
                    elif arc_elem is not None:
                        # Sample points along an arc
                        curvature = float(arc_elem.get("curvature", 0.0))
                        road_points.extend(
                            self._sample_arc(sx, sy, shdg, length, curvature, self.sampling_resolution)
                        )
                    else:
                        # For simplicity, ignore <spiral> or other geometry in this example
                        pass

                except Exception as e:
                    print(f"[WARN] Error parsing geometry in road {road_id}: {e}")

            # Now add edges between consecutive points in road_points
            for i in range(len(road_points) - 1):
                p1 = road_points[i]
                p2 = road_points[i + 1]
                dist = self._euclidean_distance(p1, p2)

                # Add a directed edge in both directions if you want 2-way lanes.
                # If your road is strictly one-way, add only p1->p2 or p2->p1.
                self.graph.add_edge(p1, p2, weight=dist, road=road_id)
                self.graph.add_edge(p2, p1, weight=dist, road=road_id)

    def _sample_line(self, sx, sy, shdg, length, ds):
        """
        Sample points along a straight line starting at (sx, sy) with heading shdg, length 'length'.
        :param ds: sampling interval (e.g. 1.0m)
        """
        points = []
        n_samples = int(math.floor(length / ds))

        for i in range(n_samples + 1):
            s = min(i * ds, length)
            # x(s) = sx + s*cos(shdg)
            # y(s) = sy + s*sin(shdg)
            x = sx + s * math.cos(shdg)
            y = sy + s * math.sin(shdg)
            points.append((x, y))
        return points

    def _sample_arc(self, sx, sy, shdg, length, curvature, ds):
        """
        Sample points along an arc.
         - curvature = 1/radius, if curvature>0 => arc goes "left", if <0 => "right"
         - heading changes along the arc by (curvature * distance).
        """
        points = []
        n_samples = int(math.floor(length / ds))
        R = 1.0 / curvature if abs(curvature) > 1e-8 else 1e10  # avoid div by zero

        # sign of curvature => left or right turning
        for i in range(n_samples + 1):
            s = min(i * ds, length)
            # Δθ = curvature * s
            dtheta = curvature * s

            # Using arc geometry from OpenDRIVE reference:
            # x(s) = sx - R*sin(shdg) + R*sin(shdg + dθ)
            # y(s) = sy + R*cos(shdg) - R*cos(shdg + dθ)
            # (This formula might differ if your sign conventions vary.)
            x = sx - R * math.sin(shdg) + R * math.sin(shdg + dtheta)
            y = sy + R * math.cos(shdg) - R * math.cos(shdg + dtheta)

            points.append((x, y))
        return points

    @staticmethod
    def _euclidean_distance(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def plan_route(self, start_xy, goal_xy):
        """
        Finds a path (list of (x, y)) from start_xy to goal_xy using A* on the built graph.
        """
        # If the exact start_xy or goal_xy isn't in the graph, find the nearest node
        start_node = self._find_nearest_node(start_xy)
        goal_node = self._find_nearest_node(goal_xy)

        try:
            path = nx.astar_path(self.graph, start_node, goal_node, weight='weight')
            return path
        except nx.NetworkXNoPath:
            print("[ERROR] No path found!")
            return None

    def _find_nearest_node(self, query_xy, radius=5.0):
        """
        Finds the nearest node in the graph to the query_xy within 'radius' tolerance.
        If none is within radius, it returns the node with the overall min distance.
        """
        best_node = None
        best_dist = float('inf')

        for node in self.graph.nodes:
            dist = self._euclidean_distance(node, query_xy)
            if dist < best_dist:
                best_dist = dist
                best_node = node

        # If you want to strictly require the node to be within a certain radius, you can check that here.
        # We'll just return the best node for simplicity.
        return best_node

    def smooth_path_cubic_spline(self, path_xy, spline_ds=1.0):
        """
        Takes a discrete path (list of (x,y)) and returns a smoothed path with heading.
        Uses cubic spline interpolation of x and y over the path distance.
        """
        if len(path_xy) < 3:
            # Not enough points to meaningfully smooth
            return [(p[0], p[1], 0.0) for p in path_xy]

        # Separate out x and y
        xs = [p[0] for p in path_xy]
        ys = [p[1] for p in path_xy]

        # Compute cumulative distances for param (t)
        cum_dist = [0.0]
        for i in range(1, len(path_xy)):
            cum_dist.append(cum_dist[-1] + self._euclidean_distance(path_xy[i - 1], path_xy[i]))

        # Spline each dimension w.r.t. distance
        spline_x = CubicSpline(cum_dist, xs)
        spline_y = CubicSpline(cum_dist, ys)

        # Sample at uniform intervals
        t_new = np.arange(0, cum_dist[-1], spline_ds)
        x_new = spline_x(t_new)
        y_new = spline_y(t_new)

        # Yaw from derivative
        dx_dt = spline_x.derivative()(t_new)
        dy_dt = spline_y.derivative()(t_new)

        smoothed = []
        for (x, y, dx, dy) in zip(x_new, y_new, dx_dt, dy_dt):
            heading = math.atan2(dy, dx)
            smoothed.append((x, y, heading))

        return smoothed

    def velocity_planner(self, path_xyz, max_speed=10.0, max_accel=2.0, max_decel=3.0, max_lat_acc=2.0):
        """
        Given a path of (x, y, heading) points, plan velocities.
         - max_speed: [m/s]
         - max_accel: [m/s^2]
         - max_decel: [m/s^2]
         - max_lat_acc: [m/s^2] used to limit speed around curves: v <= sqrt(a_lat / curvature).
        
        Returns a list of (x, y, heading, speed).
        """
        if not path_xyz:
            return []

        # 1) Precompute curvature for each segment
        # curvature ~ |dθ/ds|. We can approximate from finite differences or the heading changes.
        # For a small 2D segment, curvature kappa = dHeading / dArcLength
        # We'll do a rough discrete estimate.
        curvatures = self._estimate_curvature(path_xyz)

        # 2) First pass: limit speed by curvature => v <= sqrt(max_lat_acc / curvature)
        speeds_curvature = []
        for c in curvatures:
            if abs(c) < 1e-6:
                # near-zero curvature => no limit from lateral acceleration
                speeds_curvature.append(max_speed)
            else:
                # v_max = sqrt(a_lat_max / curvature)
                v_curv = math.sqrt(max_lat_acc / abs(c))
                speeds_curvature.append(min(v_curv, max_speed))

        # 3) Forward pass for acceleration constraint
        speeds_fwd = [speeds_curvature[0]]
        for i in range(1, len(path_xyz)):
            ds = self._euclidean_distance(path_xyz[i], path_xyz[i - 1])
            # v_possible = sqrt(v_{i-1}^2 + 2 * a_max * ds)
            v_possible = math.sqrt(speeds_fwd[-1] ** 2 + 2 * max_accel * ds)
            v_limited = min(v_possible, speeds_curvature[i])
            speeds_fwd.append(v_limited)

        # 4) Backward pass for deceleration constraint
        speeds_bwd = [speeds_fwd[-1]] * len(path_xyz)
        for i in range(len(path_xyz) - 2, -1, -1):
            ds = self._euclidean_distance(path_xyz[i], path_xyz[i + 1])
            # v_possible = sqrt(v_{i+1}^2 + 2 * decel * ds)
            v_possible = math.sqrt(speeds_bwd[i + 1] ** 2 + 2 * max_decel * ds)
            v_limited = min(speeds_fwd[i], v_possible)
            speeds_bwd[i] = v_limited

        # Speeds are final from backward pass
        speeds_final = speeds_bwd

        # Construct final trajectory with speeds
        final_traj = []
        for i, (x, y, hdg) in enumerate(path_xyz):
            final_traj.append((x, y, hdg, speeds_final[i]))

        return final_traj

    def _estimate_curvature(self, path_xyz):
        """
        Estimates discrete curvature kappa = dHeading/dArcLength for each path point.
        We'll keep it simple: kappa[i] ~ (heading[i+1] - heading[i]) / distance(i -> i+1).
        """
        curvatures = []
        for i in range(len(path_xyz) - 1):
            x0, y0, h0 = path_xyz[i]
            x1, y1, h1 = path_xyz[i + 1]

            ds = self._euclidean_distance((x0, y0), (x1, y1))
            dh = self._angle_diff(h1, h0)  # handle angle wrap properly
            if ds < 1e-6:
                curvatures.append(0.0)
            else:
                curvatures.append(dh / ds)

        # We have one less curvature than points, just replicate the last or append 0
        curvatures.append(curvatures[-1] if curvatures else 0.0)
        return curvatures

    @staticmethod
    def _angle_diff(a, b):
        """
        Computes the difference between angles a and b, in [-pi, pi].
        """
        d = a - b
        while d > math.pi:
            d -= 2.0 * math.pi
        while d < -math.pi:
            d += 2.0 * math.pi
        return d

    def get_global_plan(
        self,
        start_xy,
        goal_xy,
        spline_ds=1.0,
        max_speed=10.0,
        max_accel=2.0,
        max_decel=3.0,
        max_lat_acc=2.0,
    ):
        """
        High-level method that returns a final global path with speeds.
         1) Plan route via A*.
         2) Smooth path with cubic spline.
         3) Plan velocities with constraints.
        
        Returns a list of (x, y, heading, speed).
        """
        # 1) Plan discrete route
        route = self.plan_route(start_xy, goal_xy)
        if not route:
            return []

        # 2) Spline-smooth the path => (x, y, heading)
        smooth_xyz = self.smooth_path_cubic_spline(route, spline_ds)

        # 3) Velocity planning => (x, y, heading, speed)
        final_trajectory = self.velocity_planner(
            smooth_xyz,
            max_speed=max_speed,
            max_accel=max_accel,
            max_decel=max_decel,
            max_lat_acc=max_lat_acc,
        )

        return final_trajectory


import numpy as np
import math
from numpy.polynomial.polynomial import Polynomial

import numpy as np


class trajectory:
    """
    A class representing a trajectory.

    Attributes:
        reference_path (numpy.ndarray): The reference path for the trajectory.
        reference_x (numpy.ndarray): The x-coordinates of the reference path.
        reference_y (numpy.ndarray): The y-coordinates of the reference path.
        cumulative_distances (list): The cumulative distances along the reference path.
        reference_s (numpy.ndarray): The s-coordinates of the reference path in the Frenet frame.
        reference_d (numpy.ndarray): The d-coordinates of the reference path in the Frenet frame.
        reference_sd_path (numpy.ndarray): The reference path in the Frenet frame.
        next_wp (int): The index of the next waypoint on the reference path.
        prev_wp (int): The index of the previous waypoint on the reference path.
    """
    def __init__(self, reference_path):
        self.reference_path = np.array(reference_path)
        # [point[0] for point in reference_path]
        self.reference_x = self.reference_path[:, 0]
        # [point[1] for point in reference_path]
        self.reference_y = self.reference_path[:, 1]
        self.cumulative_distances = self._precompute_cumulative_distances()
        self.reference_s, self.reference_d = self.convert_to_frenet(
            self.reference_path)
        self.reference_sd_path = np.array(list(zip(self.reference_s, self.reference_d)))
        self.next_wp = 1
        self.prev_wp = 0


    def update_waypoint(self, x_current, y_current):
        # not efficient 
        diffs = self.reference_path - np.array((x_current, y_current))
        dists = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
        closest_wp = np.argmin(dists)
        s_, d_ = self.convert_to_frenet([(x_current, y_current)])

        if self.reference_s[closest_wp] <= s_[0] and closest_wp < len(self.reference_path) - 1:
            self.prev_wp = closest_wp
            self.next_wp = closest_wp + 1 % len(self.reference_path)
        elif self.reference_s[closest_wp] > s_[0] and closest_wp > 0:
            self.next_wp = closest_wp
            self.prev_wp = closest_wp - 1

        if self.next_wp == 0:
            self.prev_wp = len(self.reference_path) - 1
        else:
            self.prev_wp = self.next_wp - 1

    # TODO recursively update the waypoints
    def quick_waypoint_update(self, x_current, y_current, search_window = 20):
        diffs = self.reference_path[-search_window:-1,:] - np.array((x_current, y_current))
        dists = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
        closest_wp = np.argmin(dists)
        s_ = self.cumulative_distances[-search_window + closest_wp]  

        if self.reference_s[closest_wp] <= s_ and closest_wp < len(self.reference_path) - 1:
            self.prev_wp = closest_wp
            self.next_wp = closest_wp + 1 % len(self.reference_path)
        elif self.reference_s[closest_wp] > s_ and closest_wp > 0:
            self.next_wp = closest_wp
            self.prev_wp = closest_wp - 1

        if self.next_wp == 0:
            self.prev_wp = len(self.reference_path) - 1
        else:
            self.prev_wp = self.next_wp - 1

    # TODO recusively update the waypoints
    def get_current_frenet(self):
        diff = self.reference_path[self.prev_wp] - np.array((self.x_current, self.y_current))
        dist = np.sqrt(diff[0]**2 + diff[1]**2)
        total_dist = self.comulative_distances[self.prev_wp] + dist
        pass

    def generate_local_edge_trajectory(self, s_start, s_end, d_start, d_end,  num_points=10):
        # Generate a list of s values from s_start to s_end
        s_values = np.linspace(s_start, s_end, num_points)

        # Fit a 5th degree polynomial to the start and end points
        poly = Polynomial.fit([s_start, s_end], [d_start, d_end], 5)

        # Calculate the d values for the trajectory
        d_values = poly(s_values)
        tx, ty = zip(*[self.getXY(s, d) for s, d in zip(s_values, d_values)])

        return s_values, d_values, tx, ty

    def _precompute_cumulative_distances(self):
        reference_path = np.array(self.reference_path)
        cumulative_distances = [0]
        for i in range(1, len(reference_path)):
            cumulative_distances.append(
                cumulative_distances[i-1] + np.linalg.norm(reference_path[i] - reference_path[i-1]))
        return cumulative_distances

    def convert_to_frenet(self, points):

        reference_path = self.reference_path
        frenet_coords = []
        cumulative_distances = self.cumulative_distances
        for point in points:

            closest_wp = self._get_closest_waypoint(point[0], point[1]) 

            if closest_wp == 0:
                next_wp = 1
                prev_wp = 0
            else:
                next_wp = closest_wp
                prev_wp = next_wp - 1

            n_x = reference_path[next_wp, 0] - reference_path[prev_wp, 0]
            n_y = reference_path[next_wp, 1] - reference_path[prev_wp, 1]
            x_x = point[0] - reference_path[prev_wp, 0]
            x_y = point[1] - reference_path[prev_wp, 1]

            # Compute the projection of the point onto the reference path
            if (n_x*n_x + n_y*n_y) == 0:
                proj_x = 0
                proj_y = 0
            else:
                proj_norm = (x_x*n_x + x_y*n_y) / (n_x*n_x + n_y*n_y) # normalized projection
                proj_x = proj_norm * n_x
                proj_y = proj_norm * n_y

            # Compute the Frenet s coordinate based on the longitudinal position along the reference path
            s = cumulative_distances[prev_wp] + np.sqrt(proj_x**2 + proj_y**2)
            # Compute the Frenet d coordinate based on the lateral distance from the reference path
            # The sign of the d coordinate is determined by the cross product of the vectors to the point and along the reference path
            d = -np.sign(x_x*n_y - x_y*n_x) * np.sqrt((x_x - proj_x)**2 + (x_y - proj_y)**2)

            frenet_coords.append((s, d))

        return zip(*frenet_coords)

    # TODO: this inefficient! need to look into a window only not the whole track
    def _get_closest_waypoint(self, x, y):
        diffs = self.reference_path - np.array((x, y))
        dists = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
        closest_wp = np.argmin(dists)
        return closest_wp
    
    def _get_closest_sd_waypoint(self, s, d):
        diffs = self.reference_sd_path - np.array((s, d))
        dists = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
        closest_wp = np.argmin(dists)
        return closest_wp

    
    # s,d need to be current
    def getXY(self, s, d):
        closest_wp = self._get_closest_sd_waypoint(s, d)    

        if closest_wp == 0:
            next_wp = 1
            prev_wp = 0
        else:
            next_wp = closest_wp
            prev_wp = next_wp - 1

        # Calculate the heading of the track at the previous waypoint
        heading = math.atan2(self.reference_y[next_wp] - self.reference_y[prev_wp],
                             self.reference_x[next_wp] - self.reference_x[prev_wp])
        # Calculate the x and y coordinates on the reference path
        if 0 <= prev_wp < len(self.reference_x) and 0 <= self.reference_s[prev_wp] <= s:
            x = self.reference_x[prev_wp] + \
                (s - self.reference_s[prev_wp]) * math.cos(heading)
            y = self.reference_y[prev_wp] + \
                (s - self.reference_s[prev_wp]) * math.sin(heading)

        # Calculate the perpendicular heading
        perp_heading = heading - math.pi / 2

        # Calculate the final x and y coordinates
        x_final = x - d * math.cos(perp_heading)
        y_final = y - d * math.sin(perp_heading)

        # TODO: need to fix the issue when prev is the last point in the track and we come back to the biginning
        return x_final, y_final

    def getXY_path(self, s_values, d_values):
        # return [self.getXY(s, d) for s, d in zip(s_values, d_values)]
        x_values = []
        y_values = []

        for prev_wp in range(len(s_values)-2): # just reading s and d for next_wp of given trajectory
            next_wp = prev_wp + 1
            s = s_values[next_wp]
            d = d_values[next_wp]
            # Calculate the heading of the track at the previous waypoint
            heading = math.atan2(self.reference_y[next_wp] - self.reference_y[prev_wp],
                                 self.reference_x[next_wp] - self.reference_x[prev_wp])
            

            # Calculate the x and y coordinates on the reference path
            if 0 <= prev_wp < len(self.reference_x) and 0 <= self.reference_s[prev_wp] <= s:
                x = self.reference_x[prev_wp] + \
                    (s - self.reference_s[prev_wp]) * math.cos(heading)
                y = self.reference_y[prev_wp] + \
                    (s - self.reference_s[prev_wp]) * math.sin(heading)

            # Calculate the perpendicular heading
            perp_heading = heading - math.pi / 2

            # Calculate the final x and y coordinates
            x_final = x  - d * math.cos(perp_heading) # negative seems make sense
            y_final = y  - d * math.sin(perp_heading)

            x_values.append(x_final)
            y_values.append(y_final)

        return x_values, y_values

if __name__ == '__main__':
    import dummy_planners
    dummy_planners.main()

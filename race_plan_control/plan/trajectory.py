import numpy as np
import math
from numpy.polynomial.polynomial import Polynomial
import numpy as np
import logging
log = logging.getLogger(__name__)


class Trajectory:
    def __init__(self, reference_xy_path, name="Global Trajectory"):
        self.__reference_path = np.array(reference_xy_path)
        self.path_x = self.__reference_path[:, 0]
        self.path_y = self.__reference_path[:, 1]
        self.__cumulative_distances = self.__precompute_cumulative_distances()
        self.path_s, self.path_d = self.convert_xy_path_to_sd_path(self.__reference_path) # this should be with respect to parent trajectory
        self.__reference_sd_path = np.array(list(zip(self.path_s, self.path_d)))

        self.poly = None # for local trajectory
        self.parent_trajectory = None
        self.path_s_with_respect_to_parent = None
        self.path_d_with_respect_to_parent = None
        
        
        self.next_wp = 1
        self.current_wp = 0
        
        self.name = name # needed to distinguish between global and local trajectory
    
    def get_current_xy(self):
        return self.path_x[self.current_wp], self.path_y[self.current_wp]
    def get_current_sd(self):
        return self.path_s[self.current_wp], self.path_d[self.current_wp]

    def get_xy_by_waypoint(self, wp:int):
        return self.path_x[wp], self.path_y[wp]

    def get_sd_by_waypoint(self, wp:int):
        return self.path_s[wp], self.path_d[wp]
    
    def update_waypoint_by_xy(self, x_current, y_current):
        # not efficient 
        diffs = self.__reference_path - np.array((x_current, y_current))
        dists = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
        closest_wp = np.argmin(dists)
        s_, d_ = self.convert_xy_path_to_sd_path([(x_current, y_current)])

        if self.path_s[closest_wp] <= s_[0]:
            if closest_wp < len(self.__reference_path) - 1:
                self.current_wp = closest_wp
                self.next_wp = closest_wp + 1 % len(self.__reference_path)
            elif closest_wp == len(self.__reference_path) - 1:
                self.current_wp = closest_wp
                self.next_wp = closest_wp
        elif self.path_s[closest_wp] > s_[0] and closest_wp > 0:
            self.next_wp = closest_wp
            self.current_wp = closest_wp - 1
    
    def update_waypoint_by_wp(self, current_wp): 
        self.current_wp = current_wp % len(self.__reference_path) 
        self.next_wp = current_wp + 1 % len(self.__reference_path)
    
    def update_to_next_waypoint(self):
        self.update_waypoint_by_wp(self.current_wp + 1)

    def is_traversed(self):
        return self.current_wp == len(self.__reference_path) - 1

    def create_trajectory_in_sd_coordinate(self, s_start, d_start, s_end, d_end, s_start_derv = None, d_start_derv = None,
                                         num_points=10):
        # Generate a list of s values from s_start to s_end
        s_values = np.linspace(s_start, s_end, num_points)

            # Fit a 5th degree polynomial to the start and end points
        if s_start_derv is not None and d_start_derv is not None:
            # Set up the system of equations
            A = np.array([
                [s_start**5, s_start**4, s_start**3, s_start**2, s_start, 1],
                [s_end**5, s_end**4, s_end**3, s_end**2, s_end, 1],
                [5*s_start**4, 4*s_start**3, 3*s_start**2, 2*s_start, 1, 0],
                [5*s_end**4, 4*s_end**3, 3*s_end**2, 2*s_end, 1, 0]
            ])

            b = np.array([d_start, d_end, s_start_derv, d_start_derv])

            # Solve for the polynomial coefficients
            coefficients = np.linalg.solve(A, b)

            # Create the polynomial
            self.poly = Polynomial(coefficients[::-1])  # Reverse coefficients for Polynomial
        else:
            self.poly = Polynomial.fit([s_start, s_end], [d_start, d_end], 5)

        # Calculate the d values for the trajectory
        d_values = self.poly(s_values)

        tx, ty = zip(*[self.convert_sd_to_xy(s, d) for s, d in zip(s_values, d_values)])

        path = list(zip(tx, ty))
        local_trajectory = Trajectory(path, name="Local Trajectory")
        
        local_trajectory.parent_trajectory = self
        local_trajectory.path_s_with_respect_to_parent = s_values
        local_trajectory.path_d_with_respect_to_parent = d_values

        return local_trajectory
    
    def convert_xy_to_sd(self, x, y):
        s,d = self.convert_xy_path_to_sd_path([(x, y)])
        _s = s[0]
        _d = d[0]

        return _s,_d
    
    # s,d need to be current
    def convert_sd_to_xy(self, s, d):
        closest_wp = self.__get_closest_sd_waypoint(s, d)    

        if closest_wp == 0:
            next_wp = 1
            prev_wp = 0
        else:
            next_wp = closest_wp
            prev_wp = next_wp - 1

        # Calculate the heading of the track at the previous waypoint
        heading = math.atan2(self.path_y[next_wp] - self.path_y[prev_wp],
                             self.path_x[next_wp] - self.path_x[prev_wp])
        # Calculate the x and y coordinates on the reference path
        if 0 <= prev_wp < len(self.path_x) and 0 <= self.path_s[prev_wp] <= s:
            x = self.path_x[prev_wp] + \
                (s - self.path_s[prev_wp]) * math.cos(heading)
            y = self.path_y[prev_wp] + \
                (s - self.path_s[prev_wp]) * math.sin(heading)

        # Calculate the perpendicular heading
        perp_heading = heading - math.pi / 2

        # Calculate the final x and y coordinates
        x_final = x - d * math.cos(perp_heading)
        y_final = y - d * math.sin(perp_heading)

        # TODO: need to fix the issue when prev is the last point in the track and we come back to the biginning
        return x_final, y_final


    def convert_xy_path_to_sd_path(self, points):

        reference_path = self.__reference_path
        frenet_coords = []
        cumulative_distances = self.__cumulative_distances
        for point in points:

            closest_wp = self.__get_closest_waypoint(point[0], point[1]) 

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

    def convert_sd_path_to_xy_path(self, s_values, d_values):
        # return [self.getXY(s, d) for s, d in zip(s_values, d_values)]
        x_values = []
        y_values = []

        for prev_wp in range(len(s_values)-2): # just reading s and d for next_wp of given trajectory
            next_wp = prev_wp + 1
            s = s_values[next_wp]
            d = d_values[next_wp]
            # Calculate the heading of the track at the previous waypoint
            heading = math.atan2(self.path_y[next_wp] - self.path_y[prev_wp],
                                 self.path_x[next_wp] - self.path_x[prev_wp])
            

            # Calculate the x and y coordinates on the reference path
            if 0 <= prev_wp < len(self.path_x) and 0 <= self.path_s[prev_wp] <= s:
                x = self.path_x[prev_wp] + \
                    (s - self.path_s[prev_wp]) * math.cos(heading)
                y = self.path_y[prev_wp] + \
                    (s - self.path_s[prev_wp]) * math.sin(heading)

            # Calculate the perpendicular heading
            perp_heading = heading - math.pi / 2

            # Calculate the final x and y coordinates
            x_final = x  - d * math.cos(perp_heading) # negative seems make sense
            y_final = y  - d * math.sin(perp_heading)

            x_values.append(x_final)
            y_values.append(y_final)

        return x_values, y_values

    def __precompute_cumulative_distances(self):
        reference_path = np.array(self.__reference_path)
        cumulative_distances = [0]
        for i in range(1, len(reference_path)):
            cumulative_distances.append(
                cumulative_distances[i-1] + np.linalg.norm(reference_path[i] - reference_path[i-1]))
        return cumulative_distances



    # TODO: this inefficient! need to look into a window only not the whole track
    def __get_closest_waypoint(self, x, y):
        diffs = self.__reference_path - np.array((x, y))
        dists = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
        closest_wp = np.argmin(dists)
        return closest_wp
    
    def __get_closest_sd_waypoint(self, s, d):
        diffs = self.__reference_sd_path - np.array((s, d))
        dists = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
        closest_wp = np.argmin(dists)
        return closest_wp

    


if __name__ == "__main__":
    import race_plan_control.main as main
    main.run()
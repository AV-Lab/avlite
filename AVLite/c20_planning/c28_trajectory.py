import logging
from typing import Optional
import numpy as np
import math
from numpy.polynomial.polynomial import Polynomial
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

@dataclass
class Trajectory:
    """
    A class to represent a trajectory (path + velocity with aligned resolution).
    """
    path: list[tuple[float,float]] = field(default_factory=list)
    path_x: np.ndarray = np.ndarray(0)
    path_y: np.ndarray = np.ndarray(0)
    path_s: list[float] = field(default_factory=list) # progress along the path
    path_d: list[float] = field(default_factory=list) # always zero, for debugging
    velocity: list[float] = field(default_factory=list)
    current_wp: int = 0
    next_wp: int = 1

    name: str = "Global Trajectory"
    is_initialized:bool = False
    

    poly_d:Optional[Polynomial] = None # for sub trajectory (used by local planner)
    poly_x: Optional[Polynomial] = None
    poly_y: Optional[Polynomial] = None
    parent_trajectory: Optional["Trajectory"] = None
    path_s_from_parent: Optional[list[float]]  = None
    path_d_from_parent: Optional[list[float]] = None


    def __post_init__(self):
        self.initialize_trajectory(self.path, self.velocity)


    def initialize_trajectory(self, reference_xy_path: list[tuple[float, float]], velocity: list[float]):
        self.path = reference_xy_path
        if reference_xy_path is None or len(reference_xy_path) == 0:
            return

        self.is_initialized = True  
        self.__reference_path = np.array(reference_xy_path)
        self.velocity = velocity
        self.path_x = self.__reference_path[:, 0]
        self.path_y = self.__reference_path[:, 1]
        self.__cumulative_distances = self.__precompute_cumulative_distances()
        self.path_s, self.path_d = self.convert_xy_path_to_sd_path(
            self.__reference_path
        )  


        # this should be with respect to parent trajectory
        self.__reference_sd_path = np.array(list(zip(self.path_s, self.path_d)))


    
    def get_current_xy(self) -> tuple[float, float]:
        """
        Returns the current X and Y coordinates.
        """
        return self.path_x[self.current_wp], self.path_y[self.current_wp]

    def get_current_sd(self) -> tuple[float, float]:
        """
        Returns the current S and D coordinates.
        """
        if not self.is_initialized:
            raise ValueError("Trajectory not initialized")

        return self.path_s[self.current_wp], self.path_d[self.current_wp]

    def get_xy_by_waypoint(self, wp: int) -> tuple[float, float]:
        """
        Returns the X and Y coordinates for a given waypoint.

        Parameters:
        wp : int
            The waypoint index.
        """
        if not self.is_initialized:
            raise ValueError("Trajectory not initialized")

        return self.path_x[wp], self.path_y[wp]

    def get_sd_by_waypoint(self, wp: int) -> tuple[float, float]:
        """
        Returns the S and D coordinates for a given waypoint.

        Parameters:
        wp : int
            The waypoint index.
        """
        return self.path_s[wp], self.path_d[wp]

    def update_waypoint_by_xy(self, x_current: float, y_current: float) -> None:
        """
        Updates the current and next waypoints based on the current x and y coordinates.

        This method calculates the differences between the current position (x_current, y_current)
        and all points in the reference path, and identifies the closest waypoint. It then updates
        the current and next waypoints accordingly.

        Parameters:
        x_current (float): The current x-coordinate.
        y_current (float): The current y-coordinate.

        Returns:
        None

        Note:
        This method modifies the instance variables `self.current_wp` and `self.next_wp`.
        """
        # not efficient
        diffs = self.__reference_path - np.array((x_current, y_current))
        dists = np.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2)
        closest_wp = int(np.argmin(dists))
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

    def update_waypoint_by_wp(self, current_wp: int) -> None:
        self.current_wp = current_wp % len(self.__reference_path)
        self.next_wp = current_wp + 1 % len(self.__reference_path)

    def update_to_next_waypoint(self) -> None:
        self.update_waypoint_by_wp(self.current_wp + 1)

    def is_traversed(self) -> bool:
        """
        Check if the trajectory has been fully traversed.
        """
        return self.current_wp >= len(self.__reference_path) - 1

    def create_quintic_trajectory_sd(
        self,
        s_start: float,
        d_start: float,
        s_end: float,
        d_end: float,
        start_d_1st_derv: float = 0.0,
        start_d_2nd_derv: float = 0.0,
        end_d_1st_derv: float = 0.0,
        end_d_2nd_derv: float = 0.0,
        num_points=10,
    ) -> "Trajectory":
        """
        Create a quintic polynomial trajectory in the s-d plane with C2 continuity with respect to the current trajectory.
        By default, speed profile is taken 
        """

        A = np.array(
            [
                [
                    s_start**5,
                    s_start**4,
                    s_start**3,
                    s_start**2,
                    s_start,
                    1,
                ],  # Polynomial at s_start
                [
                    s_end**5,
                    s_end**4,
                    s_end**3,
                    s_end**2,
                    s_end,
                    1,
                ],  # Polynomial at s_end
                [
                    5 * s_start**4,
                    4 * s_start**3,
                    3 * s_start**2,
                    2 * s_start,
                    1,
                    0,
                ],  # 1st derivative at s_start
                [
                    5 * s_end**4,
                    4 * s_end**3,
                    3 * s_end**2,
                    2 * s_end,
                    1,
                    0,
                ],  # 1st derivative at s_end
                [
                    20 * s_start**3,
                    12 * s_start**2,
                    6 * s_start,
                    2,
                    0,
                    0,
                ],  # 2nd derivative at s_start
                [
                    20 * s_end**3,
                    12 * s_end**2,
                    6 * s_end,
                    2,
                    0,
                    0,
                ],  # 2nd derivative at s_end
            ]
        )

        b = np.array([d_start, d_end, start_d_1st_derv, start_d_2nd_derv, end_d_2nd_derv, end_d_2nd_derv])

        # Solve for the polynomial coefficients
        coefficients = np.linalg.solve(A, b)

        # Create the polynomial
        poly = Polynomial(coefficients[::-1])  # Reverse coefficients for Polynomial

        # Generate a list of s values from s_start to s_end
        s_values = np.linspace(s_start, s_end, num_points)
        return self.__decorate_trajectory_sd(poly, s_values)

    def create_default_trajectory_sd(
        self, s_start: float, d_start: float, s_end: float, d_end: float, num_points=10
    ) -> "Trajectory":
        poly = Polynomial.fit([s_start, s_end], [d_start, d_end], 3)
        # log.debug(f"Poly Coefficients: {poly.coef}")

        # Generate a list of s values from s_start to s_end
        s_values = np.linspace(s_start, s_end, num_points)
        return self.__decorate_trajectory_sd(poly, s_values)

    def create_cubic_trajectory_sd(
        self,
        s_start: float,
        d_start: float,
        s_end: float,
        d_end: float,
        d_start_1st_derv: float,
        d_start_2nd_derv: float,
        num_points=10,
    ) -> "Trajectory":

        A = np.array(
            [
                [s_start**3, s_start**2, s_start, 1],  # Polynomial at s_start
                [s_end**3, s_end**2, s_end, 1],  # Polynomial at s_end
                [3 * s_start**2, 2 * s_start, 1, 0],  # 1st derivative at s_start
                [6 * s_start, 2, 0, 0],  # 2nd derivative at s_start
            ]
        )
        # A = np.array(
        #     [
        #         [0, 0, 0, 1],  # Polynomial at s_start = 0
        #         [1, 1, 1, 1],  # Polynomial at s_end = 1
        #         [0, 0, 1, 0],  # 1st derivative at s_start
        #         [0, 2, 0, 0],  # 2nd derivative at s_start
        #     ]
        # )

        b = np.array([d_start, d_end, d_start_1st_derv, d_start_2nd_derv])

        # Solve for the polynomial coefficients
        coefficients = np.linalg.solve(A, b)

        # Create the polynomial
        poly = Polynomial(coefficients[::-1])  # Reverse coefficients for Polynomial
        log.info(f"Poly Coefficients (C2 Continuity): {poly.coef}")

        # Generate a list of s values from s_start to s_end
        s_values = np.linspace(s_start, s_end, num_points)
        return self.__decorate_trajectory_sd(poly, s_values)

    def __decorate_trajectory_sd(self, poly: Polynomial, s_values: list[float]) -> "Trajectory":
        """
        Decorate the trajectory with calculated d values and convert to (x, y) coordinates.

        Args:
            poly (Polynomial): The polynomial used to calculate d values.
            s_values (iterable): The s values along the trajectory.

        Returns:
            Trajectory: The decorated trajectory with (x, y) coordinates.
        """

        # Calculate the d values for the trajectory
        d_values = poly(s_values)

        tx, ty = zip(*[self.convert_sd_to_xy(s, d) for s, d in zip(s_values, d_values)])

        # finding velocities
        start_v = self.get_closest_waypoint_frm_sd(s_values[0], d_values[0])
        end_v = self.get_closest_waypoint_frm_sd(s_values[-1], d_values[-1])

        vel = self.velocity[start_v:end_v+1]

        # increasing the resolution of the velocity array
        current_size = len(vel)
        if current_size < 2:
            raise ValueError("Need at least two velocity values to interpolate.")
        x_old = np.linspace(0, 1, current_size)
        x_new = np.linspace(0, 1, len(s_values))
        velocity_high_res = np.interp(x_new, x_old, vel)

        path = list(zip(tx, ty))

        local_trajectory = Trajectory(path, name="Local Trajectory", velocity=velocity_high_res)

        local_trajectory.poly_d = poly
        local_trajectory.parent_trajectory = self
        local_trajectory.path_s_from_parent = s_values
        local_trajectory.path_d_from_parent = d_values

        return local_trajectory

    def create_cubic_trajectory_xy(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        start_x_1st_derv: float,
        start_y_1st_derv: float,
        start_x_2nd_derv: float,
        start_y_2nd_derv: float,
        num_points=10,
    ) -> "Trajectory":
        A = np.array(
            [
                [0, 0, 0, 1],  # Polynomial at t=0
                [1, 1, 1, 1],  # Polynomial at t=1
                [0, 0, 1, 0],  # 1st derivative at t=0
                [3, 2, 1, 0],  # 1st derivative at t=1
            ]
        )

        b_x = np.array([start_x, end_x, start_x_1st_derv, start_x_1st_derv])
        b_y = np.array([start_y, end_y, start_y_1st_derv, start_y_1st_derv])
        coefficients_x = np.linalg.solve(A, b_x)
        coefficients_y = np.linalg.solve(A, b_y)

        poly_x = Polynomial(coefficients_x[::-1])  # Reverse coefficients for Polynomial
        poly_y = Polynomial(coefficients_y[::-1])  # Reverse coefficients for Polynomial

        # Create the polynomial
        log.debug(f"start_x {start_x:.2f} start_y {start_y:.2f} end_x {end_x:.2f} end_y {end_y:.2f}")
        # log.debug(f"Poly Coefficients (C2 Continuity): X {poly_x.coef} Y {poly_y.coef}")

        t_values = np.linspace(0, 1, num_points)

        return self.__decoreate_trajectory_xy(poly_x, poly_y, t_values)

    def create_quintic_trajectory_xy(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        start_x_1st_derv: float,
        start_y_1st_derv: float,
        start_x_2nd_derv: float,
        start_y_2nd_derv: float,
        end_x_1st_derv: float,
        end_y_1st_derv: float,
        end_y_2nd_derv: float,
        end_x_2nd_derv: float,
        num_points=10,
    ) -> "Trajectory":
        A = np.array(
            [
                [0, 0, 0, 0, 0, 1],  # Polynomial at t=0
                [1, 1, 1, 1, 1, 1],  # Polynomial at t=1
                [0, 0, 0, 0, 1, 0],  # 1st derivative at t=0
                [5, 4, 3, 2, 1, 0],  # 1st derivative at t=1
                [0, 0, 0, 2, 0, 0],  # 2nd derivative at t=0
                [20, 12, 6, 2, 0, 0],  # 2nd derivative at t=1
            ]
        )

        b_x = np.array(
            [
                start_x,
                end_x,
                start_x_1st_derv,
                end_x_1st_derv,
                start_x_2nd_derv,
                end_x_2nd_derv,
            ]
        )
        b_y = np.array(
            [
                start_y,
                end_y,
                start_y_1st_derv,
                end_y_1st_derv,
                start_y_2nd_derv,
                end_y_2nd_derv,
            ]
        )

        coefficients_x = np.linalg.solve(A, b_x)
        coefficients_y = np.linalg.solve(A, b_y)

        poly_x = Polynomial(coefficients_x[::-1])  # Reverse coefficients for Polynomial
        poly_y = Polynomial(coefficients_y[::-1])  # Reverse coefficients for Polynomial

        # Create the polynomial
        log.debug(f"start_x {start_x} start_y {start_y} end_x {end_x} end_y {end_y}")
        # log.debug(f"Poly Coefficients (C2 Continuity): X {poly_x.coef} Y {poly_y.coef}")

        t_values = np.linspace(0, 1, num_points)

        return self.__decoreate_trajectory_xy(poly_x, poly_y, t_values)

    def __decoreate_trajectory_xy(self, poly_x: Polynomial, poly_y: Polynomial, t_values: np.ndarray) -> "Trajectory":
        x_values = poly_x(t_values)
        y_values = poly_y(t_values)
        path = list(zip(x_values, y_values))
        local_trajectory = Trajectory(path, name="Local Trajectory")
        s_value, d_values = self.convert_xy_path_to_sd_path(path)
        local_trajectory.path_s_from_parent = s_value
        local_trajectory.path_d_from_parent = d_values
        local_trajectory.poly_x = poly_x
        local_trajectory.poly_y = poly_y

        return local_trajectory

    def create_multiple_cubic_trajectories_xy(
        self,
        x_values: list[float],
        y_values: list[float],
        start_x_1st_derv: float,
        start_y_1st_derv: float,
        start_x_2nd_derv: float,
        start_y_2nd_derv: float,
        num_points=10,
    ) -> list["Trajectory"]:
        assert len(x_values) == len(y_values)

        local_trajectories = []
        k = len(x_values)

        for i, (x, y) in enumerate(zip(x_values, y_values)):

            local_trajectories.append(local_trajectory)
        return local_trajectories

    def convert_xy_to_sd(self, x: float, y: float) -> tuple[float, float]:
        s, d = self.convert_xy_path_to_sd_path([(x, y)])
        _s = s[0]
        _d = d[0]

        return _s, _d

    # s,d need to be current
    def convert_sd_to_xy(self, s: float, d: float) -> tuple[float, float]:
        closest_wp = self.get_closest_waypoint_frm_sd(s, d)

        if closest_wp == 0:
            next_wp = 1
            prev_wp = 0
        else:
            next_wp = closest_wp
            prev_wp = next_wp - 1

        # Calculate the heading of the track at the previous waypoint
        heading = math.atan2(
            self.path_y[next_wp] - self.path_y[prev_wp],
            self.path_x[next_wp] - self.path_x[prev_wp],
        )
        # Calculate the x and y coordinates on the reference path
        if 0 <= prev_wp < len(self.path_x) and 0 <= self.path_s[prev_wp] <= s:
            x = self.path_x[prev_wp] + (s - self.path_s[prev_wp]) * math.cos(heading)
            y = self.path_y[prev_wp] + (s - self.path_s[prev_wp]) * math.sin(heading)

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

            closest_wp = self.get_closest_waypoint_frm_xy(point[0], point[1])

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
            if (n_x * n_x + n_y * n_y) == 0:
                proj_x = 0
                proj_y = 0
            else:
                proj_norm = (x_x * n_x + x_y * n_y) / (n_x * n_x + n_y * n_y)  # normalized projection
                proj_x = proj_norm * n_x
                proj_y = proj_norm * n_y

            # Compute the Frenet s coordinate based on the longitudinal position along the reference path
            s = cumulative_distances[prev_wp] + np.sqrt(proj_x**2 + proj_y**2)
            # Compute the Frenet d coordinate based on the lateral distance from the reference path
            # The sign of the d coordinate is determined by the cross product of the vectors to the point and along the reference path
            d = -np.sign(x_x * n_y - x_y * n_x) * np.sqrt((x_x - proj_x) ** 2 + (x_y - proj_y) ** 2)

            frenet_coords.append((s, d))

        return zip(*frenet_coords)

    # A numpy version of the above function
    def convert_xy_path_to_sd_path_np(self, points):

        reference_path = self.__reference_path
        frenet_coords = np.empty((0, 2))
        cumulative_distances = self.__cumulative_distances
        for point in points:

            closest_wp = self.get_closest_waypoint_frm_xy(point[0], point[1])
            # To avoid returning the last point which is the same as the first

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
            if (n_x * n_x + n_y * n_y) == 0:
                proj_x = 0
                proj_y = 0
            else:
                proj_norm = (x_x * n_x + x_y * n_y) / (n_x * n_x + n_y * n_y)  # normalized projection
                proj_x = proj_norm * n_x
                proj_y = proj_norm * n_y

            # Compute the Frenet s coordinate based on the longitudinal position along the reference path
            s = cumulative_distances[prev_wp] + np.sqrt(proj_x**2 + proj_y**2)
            # Compute the Frenet d coordinate based on the lateral distance from the reference path
            # The sign of the d coordinate is determined by the cross product of the vectors to the point and along the reference path
            d = -np.sign(x_x * n_y - x_y * n_x) * np.sqrt((x_x - proj_x) ** 2 + (x_y - proj_y) ** 2)

            frenet_coords = np.vstack((frenet_coords, (s, d)))

        return frenet_coords

    def convert_sd_path_to_xy_path(self, s_values, d_values):
        # return [self.getXY(s, d) for s, d in zip(s_values, d_values)]
        x_values = []
        y_values = []

        for prev_wp in range(len(s_values) - 2):  # just reading s and d for next_wp of given trajectory
            next_wp = prev_wp + 1
            s = s_values[next_wp]
            d = d_values[next_wp]
            # Calculate the heading of the track at the previous waypoint
            heading = math.atan2(
                self.path_y[next_wp] - self.path_y[prev_wp],
                self.path_x[next_wp] - self.path_x[prev_wp],
            )

            # Calculate the x and y coordinates on the reference path
            if 0 <= prev_wp < len(self.path_x) and 0 <= self.path_s[prev_wp] <= s:
                x = self.path_x[prev_wp] + (s - self.path_s[prev_wp]) * math.cos(heading)
                y = self.path_y[prev_wp] + (s - self.path_s[prev_wp]) * math.sin(heading)

            # Calculate the perpendicular heading
            perp_heading = heading - math.pi / 2

            # Calculate the final x and y coordinates
            x_final = x - d * math.cos(perp_heading)  # negative seems make sense
            y_final = y - d * math.sin(perp_heading)

            x_values.append(x_final)
            y_values.append(y_final)

        return x_values, y_values

    def __precompute_cumulative_distances(self):
        reference_path = np.array(self.__reference_path)
        cumulative_distances = [0]
        for i in range(1, len(reference_path)):
            cumulative_distances.append(cumulative_distances[i - 1] + np.linalg.norm(reference_path[i] - reference_path[i - 1]))
        return cumulative_distances

    # TODO: this inefficient! need to look into a window only not the whole track
    def get_closest_waypoint_frm_xy(self, x, y):
        diffs = self.__reference_path - np.array((x, y))
        dists = np.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2)
        closest_wp = np.argmin(dists)
        return closest_wp

    # TODO: this inefficient! need to look into a window only not the whole track
    def get_closest_waypoint_frm_sd(self, s, d):
        diffs = self.__reference_sd_path - np.array((s, d))
        dists = np.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2)
        closest_wp = np.argmin(dists)
        return closest_wp

    def __str__(self):
        return f"Trajectory: {self.name}"


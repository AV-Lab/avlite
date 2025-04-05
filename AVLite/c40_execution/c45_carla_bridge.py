from c40_execution.c41_base_executer import WorldInterface
from c10_perception.c12_state import EgoState, AgentState
from c30_control.c31_base_controller import ControlComand
import carla
import math
import logging
import numpy as np

log = logging.getLogger(__name__)


class CarlaBridge(WorldInterface):
    def __init__(
        self, ego_state: EgoState, host="localhost", port=2000, scene_name="/Game/Carla/Maps/Town10HD_Opt", timeout=10.0
    ):
        log.info(f"Connecting to Carla at {host}:{port}")
        self.client = None
        self.world = None
        self.ego_state = ego_state

        # Carla stuff
        self.vehicle = None
        self.spectator = None
        self.camera_distance = 6.0  # Distance behind the car
        self.camera_height = 2.5  # Height above the car
        self.follow_camera = True  # Whether to follow the car with the camera
        self.spawn_points = []  # Available spawn points in the map
        self.scene_name = scene_name  # Store the scene name for resets
        self.current_camera_transform = None  # For smooth camera movement
        self.target_camera_transform = None  # For smooth camera movement
        self.vehicle_blueprint = None  # Store the vehicle blueprint

        try:
            self.client = carla.Client(host, port)
            self.client.set_timeout(timeout)
            log.info(f"Available maps: {self.client.get_available_maps()}")

            if scene_name not in self.client.get_available_maps():
                raise ValueError(f"Scene {scene_name} not found in available maps.")
            self.world = self.client.load_world(scene_name)
            log.info(f"Connected to Carla at {host}:{port} and loaded scene {scene_name}")

            # Get the spectator to control the camera
            self.spectator = self.world.get_spectator()

            self.spawn_points = self.world.get_map().get_spawn_points()
            log.info(f"Found {len(self.spawn_points)} spawn points in the map")

            weather = carla.WeatherParameters.ClearNoon
            self.world.set_weather(weather)

            # Initialize vehicle blueprint
            self.__initialize_vehicle_blueprint()
            self.start_bg_camera_and_state_update()
        except Exception as e:
            log.error(f"Failed to connect to Carla: {e}")
            log.error("Make sure the Carla simulator is running on the specified host and port.")

    def start_bg_camera_and_state_update(self, interval=0.1):
        """Start a periodic update of the camera position"""
        import threading
        import time

        def update():
            while True:
                self.__update_camera_position_and_state()
                time.sleep(interval)

        camera_thread = threading.Thread(target=update)
        camera_thread.daemon = True
        camera_thread.start()

    def __update_camera_position_and_state(self):
        """Update the camera position to follow behind the vehicle"""
        if not self.vehicle or not self.spectator or not self.follow_camera:
            return

        # Get the vehicle's transform
        vehicle_transform = self.vehicle.get_transform()

        # update state
        self.ego_state.x = vehicle_transform.location.x
        self.ego_state.y = vehicle_transform.location.y
        self.ego_state.theta = vehicle_transform.rotation.yaw * (3.14159 / 180.0)
        self.ego_state.velocity = self.vehicle.get_velocity().x

        # Calculate a position behind and above the vehicle
        # Convert yaw from degrees to radians
        yaw_rad = vehicle_transform.rotation.yaw * (3.14159 / 180.0)

        # Calculate position behind the car (negative of forward vector)
        dx = -self.camera_distance * math.cos(yaw_rad)
        dy = -self.camera_distance * math.sin(yaw_rad)

        # Create the camera transform
        camera_location = carla.Location(
            x=vehicle_transform.location.x + dx,
            y=vehicle_transform.location.y + dy,
            z=vehicle_transform.location.z + self.camera_height,
        )

        # Point the camera at the vehicle
        camera_rotation = carla.Rotation(pitch=-15, yaw=vehicle_transform.rotation.yaw, roll=0)  # Look down slightly

        camera_transform = carla.Transform(camera_location, camera_rotation)
        self.spectator.set_transform(camera_transform)

    def __initialize_vehicle_blueprint(self):
        """Initialize the vehicle blueprint to be used for spawning"""
        if not self.world:
            log.error("Cannot initialize vehicle blueprint: world not connected")
            return

        blueprint_library = self.world.get_blueprint_library()

        # Print available vehicle blueprints
        vehicle_blueprints = [bp.id for bp in blueprint_library.filter("vehicle.*")]
        log.info(f"Available vehicle blueprints: {vehicle_blueprints}")

        # Try to find the Tesla model, but use a fallback if not available
        try:
            self.vehicle_blueprint = blueprint_library.find("vehicle.tesla.model3")
            log.info("Using Tesla Model 3 as ego vehicle")
        except IndexError:
            # Try some common alternatives
            for vehicle_id in [
                "vehicle.audi.etron",
                "vehicle.lincoln.mkz2017",
                "vehicle.audi.a2",
                "vehicle.tesla.cybertruck",
            ]:
                try:
                    self.vehicle_blueprint = blueprint_library.find(vehicle_id)
                    log.info(f"Using alternative vehicle: {vehicle_id}")
                    break
                except IndexError:
                    continue
            else:
                # If none of the alternatives worked, use the first available vehicle
                if vehicle_blueprints:
                    self.vehicle_blueprint = blueprint_library.find(vehicle_blueprints[0])
                    log.info(f"Using first available vehicle: {vehicle_blueprints[0]}")
                else:
                    log.error("No vehicle blueprints available in Carla")
                    self.vehicle_blueprint = None

    def __spawn_ego_vehicle(self, state: EgoState):
        """Spawn the ego vehicle at the given state position"""
        if not self.world or not self.vehicle_blueprint:
            log.error("Cannot spawn vehicle: world not connected or blueprint not initialized")
            return

        # Check if state is None and create a default state if needed
        if state is None:
            log.warning("Ego state is None, creating default state at first spawn point")
            state = EgoState()
            if self.spawn_points:
                spawn_point = self.spawn_points[0]
                state.x = spawn_point.location.x
                state.y = spawn_point.location.y
                state.theta = spawn_point.rotation.yaw * (3.14159 / 180.0)
            else:
                # Default position if no spawn points
                state.x = 0.0
                state.y = 0.0
                state.theta = 0.0
            self.ego_state = state

        # Use a valid spawn point from Carla
        if self.spawn_points:
            # Find the closest spawn point to the requested state
            closest_point = None
            min_distance = float("inf")
            for point in self.spawn_points:
                distance = ((point.location.x - state.x) ** 2 + (point.location.y - state.y) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    closest_point = point

            # If we're too far from any spawn point, just use the first one
            if min_distance > 100.0:  # If more than 100 meters away
                log.warning(f"Requested position is too far from any valid spawn point. Using first spawn point.")
                spawn_point = self.spawn_points[0]
            else:
                spawn_point = closest_point

            log.info(
                f"Using spawn point at ({spawn_point.location.x}, {spawn_point.location.y}, {spawn_point.location.z})"
            )

            # Update the state to match the spawn point
            state.x = spawn_point.location.x
            state.y = spawn_point.location.y
            state.theta = spawn_point.rotation.yaw * (3.14159 / 180.0)
        else:
            log.warning("No spawn points found in Carla map! Using arbitrary spawn point.")
            spawn_point = carla.Transform(carla.Location(x=state.x, y=state.y, z=1.0))

        # Try to spawn the vehicle
        self.vehicle = self.world.spawn_actor(self.vehicle_blueprint, spawn_point)

        # If spawning fails, try other spawn points
        if not self.vehicle and self.spawn_points:
            log.warning("Failed to spawn at selected point. Trying other spawn points.")
            for i, spawn_point in enumerate(self.spawn_points):
                self.vehicle = self.world.try_spawn_actor(self.vehicle_blueprint, spawn_point)
                if self.vehicle:
                    log.info(f"Successfully spawned at alternative point {i}")
                    # Update the state to match the spawn point
                    state.x = spawn_point.location.x
                    state.y = spawn_point.location.y
                    state.theta = spawn_point.rotation.yaw * (3.14159 / 180.0)
                    break

            if not self.vehicle:
                log.error("Failed to spawn vehicle at any spawn point!")

    def update_ego_state(self, cmd: ControlComand, dt=0.01):
        """Update the ego state with the given command.
        This method applies control commands to the vehicle and updates the state.
        If the vehicle doesn't exist yet, it will be spawned.
        """
        # If vehicle doesn't exist, spawn it
        if not self.vehicle:
            self.__spawn_ego_vehicle(self.ego_state)

        log.debug(f"Applying control: {cmd}")
        assert self.ego_state is not None, "Ego state is None. Cannot update state without a reference."

        current_velocity = self.ego_state.velocity

        # Calculate throttle and brake values
        throttle = np.abs(cmd.acceleration) / self.ego_state.max_acceleration if cmd.acceleration > 0 else 0.0
        brake = np.abs(cmd.acceleration) / self.ego_state.min_acceleration if cmd.acceleration < 0 else 0.0

        # Convert to float to ensure correct type
        throttle = float(throttle)
        brake = float(brake)
        steer = float(-cmd.steer)

        # Determine reverse state
        is_nearly_stopped = current_velocity < 0.1  # threshold for "stopped"
        wants_reverse = cmd.acceleration < 0
        is_reverse = wants_reverse and is_nearly_stopped

        # In reverse mode, use throttle instead of brake for backward movement
        if is_reverse and wants_reverse:
            throttle = float(np.abs(cmd.acceleration) / self.ego_state.max_acceleration)
            brake = 0.0

        # When steering with zero throttle, maintain a small throttle to prevent stopping
        if throttle == 0.0 and brake == 0.0 and abs(cmd.steer) > 0.01:
            throttle = 0.05  # Small throttle value to maintain momentum during steering

        log.info(f"Velocity: {current_velocity}, Throttle: {throttle}, Brake: {brake}, Reverse: {is_reverse}")

        # Ensure all parameters are of the correct type for the Carla API
        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake, reverse=bool(is_reverse))
        self.vehicle.apply_control(control)

        # Update self.ego_state from vehicle
        self.get_ego_state()

    def get_ego_state(self):
        if not self.vehicle:
            self.__spawn_ego_vehicle(self.ego_state)
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        self.ego_state.x = transform.location.x
        self.ego_state.y = transform.location.y
        self.ego_state.theta = transform.rotation.yaw * (3.14159 / 180.0)
        self.ego_state.velocity = (velocity.x**2 + velocity.y**2) ** 0.5
        return self.ego_state

    def spawn_agent(self, agent_state: AgentState):
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find("vehicle.audi.a2")

        # Use a valid spawn point from Carla
        if self.spawn_points:
            # Find the closest spawn point to the requested state
            closest_point = None
            min_distance = float("inf")
            for point in self.spawn_points:
                distance = ((point.location.x - agent_state.x) ** 2 + (point.location.y - agent_state.y) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    closest_point = point

            # If we're too far from any spawn point, just use the first one
            if min_distance > 100.0:  # If more than 100 meters away
                log.warning(f"Requested position is too far from any valid spawn point. Using first spawn point.")
                spawn_point = self.spawn_points[0]
            else:
                spawn_point = closest_point

            log.info(
                f"Spawning agent at ({spawn_point.location.x}, {spawn_point.location.y}, {spawn_point.location.z})"
            )

            # Update the agent state to match the spawn point
            agent_state.x = spawn_point.location.x
            agent_state.y = spawn_point.location.y
            agent_state.theta = spawn_point.rotation.yaw * (3.14159 / 180.0)
        else:
            log.warning("No spawn points found in Carla map! Using arbitrary spawn point.")
            spawn_point = carla.Transform(carla.Location(x=agent_state.x, y=agent_state.y, z=1.0))

        # Try to spawn the agent
        agent = self.world.try_spawn_actor(vehicle_bp, spawn_point)

        # If spawning fails, try other spawn points
        if not agent and self.spawn_points:
            log.warning("Failed to spawn agent at selected point. Trying other spawn points.")
            for i, spawn_point in enumerate(self.spawn_points):
                agent = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                if agent:
                    log.info(f"Successfully spawned agent at alternative point {i}")
                    # Update the agent state to match the spawn point
                    agent_state.x = spawn_point.location.x
                    agent_state.y = spawn_point.location.y
                    agent_state.theta = spawn_point.rotation.yaw * (3.14159 / 180.0)
                    break

            if not agent:
                log.error("Failed to spawn agent at any spawn point!")

    def reset(self):
        """Reset the simulator and state.
        This method destroys the current vehicle, resets the simulation,
        and prepares the environment for a new run.
        """
        log.info("Resetting Carla simulation...")

        # Destroy the current vehicle if it exists
        if self.vehicle:
            try:
                self.vehicle.destroy()
                log.info("Destroyed existing vehicle")
            except Exception as e:
                log.error(f"Error destroying vehicle: {e}")
            finally:
                self.vehicle = None

        # Destroy all other actors that might have been created
        # (like other vehicles, sensors, etc.)
        if self.world:
            try:
                for actor in self.world.get_actors():
                    # Only destroy actors that are vehicles (not the spectator, etc.)
                    if "vehicle" in actor.type_id:
                        actor.destroy()
                log.info("Destroyed all vehicle actors")
            except Exception as e:
                log.error(f"Error destroying actors: {e}")

        # Reset camera transforms
        self.current_camera_transform = None
        self.target_camera_transform = None

        # Reset the world's state if possible
        if self.client:
            try:
                # Apply a tick to synchronize
                self.world.tick()

                # Reset the simulation to its initial state
                # This is a more thorough reset than just destroying actors
                self.world = self.client.reload_world()

                # Get the spectator again after world reload
                self.spectator = self.world.get_spectator()

                # Refresh spawn points
                self.spawn_points = self.world.get_map().get_spawn_points()

                # Set weather to clear day again
                weather = carla.WeatherParameters.ClearNoon
                self.world.set_weather(weather)

                # Re-initialize the vehicle blueprint
                self.__initialize_vehicle_blueprint()

                log.info("Carla world reset complete")
            except Exception as e:
                log.error(f"Error resetting world: {e}")

        log.info("Reset complete")

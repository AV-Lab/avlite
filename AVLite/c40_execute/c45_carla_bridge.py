from c40_execute.c41_executer import WorldInterface
from c10_perceive.c12_state import EgoState, AgentState
from c30_control.c31_base_controller import ControlComand
import carla

class CarlaBridge(WorldInterface):
    def __init__(self, host='localhost', port=2000):
        self.client = carla.Client(host, port)
        self.client.set_timeout(2.0)
        self.world = None
        self.vehicle = None

    def load_scene(self, scene_name):
        self.world = self.client.load_world(scene_name)

    def update_ego_state(self, state: EgoState, cmd: ControlComand, dt=0.01):
        if not self.vehicle:
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.find("vehicle.tesla.model3")
            spawn_point = carla.Transform(carla.Location(x=state.x, y=state.y, z=1.0))
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

        control = carla.VehicleControl(
            throttle=max(0.0, cmd.acc / 10.0),
            brake=-min(0.0, cmd.acc / 10.0),
            steer=cmd.steer
        )
        self.vehicle.apply_control(control)

        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        state.x = transform.location.x
        state.y = transform.location.y
        state.theta = transform.rotation.yaw * (3.14159 / 180.0)
        state.velocity = (velocity.x**2 + velocity.y**2)**0.5
        self.ego_state = state

    def spawn_agent(self, agent_state: AgentState):
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find("vehicle.audi.a2")
        spawn_point = carla.Transform(carla.Location(x=agent_state.x, y=agent_state.y, z=1.0))
        self.world.try_spawn_actor(vehicle_bp, spawn_point)

    def get_copy(self):
        copied_bridge = CarlaBridge()
        copied_bridge.world = self.world
        copied_bridge.vehicle = self.vehicle
        return copied_bridge

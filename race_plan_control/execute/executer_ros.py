from plan.planner import Planner
import rclpy
from rclpy.node import Node
from a2rl_bs_msgs.msg import Localization, EgoState
import yaml


class local_planner(Node, Planner):
    def __init__(self, path_to_track, frenet_zoom=15, xy_zoom=15):
        super().__init__(path_to_track, frenet_zoom, xy_zoom)

        self.create_subscription(
            Localization, "/a2rl/observer/ego_loc", self.loc_callback, 10
        )
        self.create_subscription(
            EgoState, "/a2rl/observer/ego_state", self.state_callback, 10
        )

    def state_callback(self, msg):
        # self.get_logger().info('Velocity x: %s' % str(msg.velocity.x))
        self.x_vel = msg.velocity.x
        self.y_vel = msg.velocity.y

    def loc_callback(self, msg):
        x_current = msg.position.x
        y_current = msg.position.y
        self.traversed_x.append(msg.position.x)
        self.traversed_y.append(msg.position.y)

        self.tj.update_waypoint(x_current, y_current)

        #### Frenet Coordinates
        s_, d_ = self.tj.convert_to_frenet([(msg.position.x, msg.position.y)])
        self.d.append(d_[0])
        self.s.append(s_[0])

        if len(self.d) > 0:
            d_mean = sum(self.d) / len(self.d)
            self.mse = sum((di - d_mean) ** 2 for di in self.d) / len(self.d)


def main(args=None):
    rclpy.init(args=args)

    track_boundaries_path = "/home/mkhonji/workspaces/A2RL_Integration/src/1_dashboards/resource/yasmarina.track.json"

    config_file = "/home/mkhonji/workspaces/A2RL_Integration/config/config.yaml"
    with open(config_file, "r") as f:
        config_data = yaml.safe_load(f)
        path_to_track = config_data["path_to_track"]

    plot_subscriber = local_planner(path_to_track, track_boundaries_path)

    rclpy.spin(plot_subscriber)

    plot_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

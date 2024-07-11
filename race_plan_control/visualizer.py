import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from plan.planner import Planner
from control.pid import Controller
import yaml
from util import plot 
import time
import logging
from tkinter.scrolledtext import ScrolledText
# import control.pid as pid

class PlotApp(tk.Tk):
    def __init__(self, pl):
        super().__init__()
        self.pl = pl
        self.controller = Controller()

        self.title("Path Planning Visualization")
        self.geometry("1200x600")

        # Variables for plot visibility and animation
        self.show_past_locations = tk.BooleanVar(value=True)
        self.show_last_100_locations = tk.BooleanVar(value=True)
        self.show_boundaries = tk.BooleanVar(value=True)
        self.animation_running = False

        # Plot frame
        self.plot_frame = ttk.Frame(self)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

        # Visualize frame setup
        self.visualize_frame = ttk.LabelFrame(self, text="Visualize")
        self.visualize_frame.pack(fill=tk.X, side=tk.TOP, padx=10, pady=5)


        ## UI Elements for Visualize - Checkboxes
        checkboxes_frame = ttk.Frame(self.visualize_frame)
        checkboxes_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Checkbutton(checkboxes_frame, text="Show Past Locations", variable=self.show_past_locations, command=self.update_plot).pack(anchor=tk.W, side=tk.LEFT)
        ttk.Checkbutton(checkboxes_frame, text="Show Last 100 Locations", variable=self.show_last_100_locations, command=self.update_plot).pack(anchor=tk.W, side=tk.LEFT)
        ttk.Checkbutton(checkboxes_frame, text="Show Boundaries", variable=self.show_boundaries, command=self.update_plot).pack(anchor=tk.W, side=tk.LEFT)

        self.coordinates_label = ttk.Label(checkboxes_frame, text="")
        self.coordinates_label.pack(side=tk.RIGHT)

        ## UI Elements for Visualize - Buttons
        global_frame = ttk.Frame(self.visualize_frame)
        global_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(global_frame, text="Global Coordinate:").pack(anchor=tk.W, side=tk.LEFT)
        ttk.Button(global_frame, text="Zoom In", command=self.zoom_in).pack(side=tk.LEFT)
        ttk.Button(global_frame, text="Zoom Out", command=self.zoom_out).pack(side=tk.LEFT)
        self.vehicle_state_label = ttk.Label(global_frame, text="")
        self.vehicle_state_label.pack(side=tk.RIGHT)
        frenet_frame = ttk.Frame(self.visualize_frame)
        frenet_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(frenet_frame, text="Frenet Coordinate:").pack(anchor=tk.W, side=tk.LEFT)
        ttk.Button(frenet_frame, text="Zoom In", command=self.zoom_in_frenet).pack(side=tk.LEFT)
        ttk.Button(frenet_frame, text="Zoom Out", command=self.zoom_out_frenet).pack(side=tk.LEFT)
        

        # Plan and control frame
        self.plan_control_frame = ttk.Frame(self)
        self.plan_control_frame.pack(fill=tk.X,padx=10, pady=5)

        # Plan frame
        self.plan_frame = ttk.LabelFrame(self.plan_control_frame, text="Plan")
        self.plan_frame.pack(fill=tk.X,side=tk.LEFT, padx=10, pady=5)

        ## UI Elements for Plan Control
        wp_frame = ttk.Frame(self.plan_frame)
        wp_frame.pack(fill=tk.X)
        ttk.Button(wp_frame, text="Set Waypoint", command=self.set_wp).pack(side=tk.LEFT)
        self.timestep_entry = ttk.Entry(wp_frame)
        self.timestep_entry.insert(0, "0")
        self.timestep_entry.pack(side=tk.LEFT)
        self.plan_time_labl = ttk.Label(wp_frame, text="")
        self.plan_time_labl.pack(side=tk.LEFT, padx=10)

        ttk.Button(self.plan_frame, text="Animate", command=self.start_animation).pack(side=tk.LEFT)
        ttk.Button(self.plan_frame, text="Stop", command=self.stop_animation).pack(side=tk.LEFT)
        ttk.Button(self.plan_frame, text="Replan", command=self.replan).pack(side=tk.LEFT, padx=30)
        ttk.Button(self.plan_frame, text="Step", command=self.step_plan).pack(side=tk.RIGHT)
        # ttk.Button(self.plan_frame, text="Prev", command=self.step_to_prev_wp).pack(side=tk.LEFT)

        # Control Frame
        self.control_frame = ttk.LabelFrame(self.plan_control_frame, text="Control (simulate)")
        self.control_frame.pack(fill=tk.X, side=tk.RIGHT, padx=10, pady=5)
        dt_frame = ttk.Frame(self.control_frame)
        dt_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(dt_frame, text="Δt ").pack(side=tk.LEFT)
        self.dt_entry = ttk.Entry(dt_frame)
        self.dt_entry.insert(2, "0.1")
        self.dt_entry.pack(side=tk.LEFT)

        ttk.Button(self.control_frame, text="Control Step", command=self.step_control).pack(side=tk.LEFT)
        
        ttk.Button(self.control_frame, text="Steer Left", command=self.step_steer_left).pack(side=tk.LEFT)
        ttk.Button(self.control_frame, text="Steer Right", command=self.step_steer_right).pack(side=tk.LEFT)
        ttk.Button(self.control_frame, text="Accelerate", command=self.step_acc).pack(side=tk.LEFT)
        ttk.Button(self.control_frame, text="Deccelerate", command=self.step_dec).pack(side=tk.LEFT)
        
        
        # Log Frame
        self.log_frame = ttk.LabelFrame(self, text="Log")
        self.log_frame.pack(fill=tk.X, padx=10, pady=5)
        # self.log_frame.pack(fill=tk.X, side=tk.TOP, expand=True, padx=10, pady=5)
        log_area = ScrolledText(self.log_frame, state='disabled', height=5)
        log_area.pack(side=tk.LEFT, fill=tk.BOTH,expand=True)

        # Configure logging
        logger = logging.getLogger()
        text_handler = TextHandler(log_area)
        formatter = logging.Formatter('%(module)s (%(filename)s) [%(levelname)s]: %(message)s')
        text_handler.setFormatter(formatter)
        logger.addHandler(text_handler)
        logger.setLevel(logging.INFO)
        
        logging.info("Log initialized.")            



        self.xy_zoom = 30
        self.frenet_zoom = 15
        self.fig = plot.fig
        self.ax1 = plot.ax1
        self.ax2 = plot.ax2
        
        self.sim = plot.Car(speed=30)
        plot.plot(self.pl, car=self.sim)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)  # A tk.DrawingArea.
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def _replot(self):
        plot.plot(self.pl,self.sim, xy_zoom=self.xy_zoom, frenet_zoom=self.frenet_zoom)
        self.canvas.draw()
        self.vehicle_state_label.config(text=f"Vehicle State: X: {self.sim.x:.2f}, Y: {self.sim.y:.2f}, Speed: {self.sim.speed:.2f}, Theta: {self.sim.theta:.2f}")

    def on_mouse_move(self, event):
        if event.inaxes: 
            x, y = event.xdata, event.ydata  
            if event.inaxes == self.ax1:
                self.coordinates_label.config(text=f"Global: X: {x:.2f}, Y: {y:.2f}")
            elif event.inaxes == self.ax2:
                self.coordinates_label.config(text=f"Frenet: X: {x:.2f}, Y: {y:.2f}")
        else:
            # Optionally, clear the coordinates display when the mouse is not over the axes
            self.coordinates_label.config(text="")


    def update_plot(self):
        # Placeholder for the method to update the plot based on the checkboxes
        pass


    def zoom_in(self):
        self.xy_zoom -= 5 if self.xy_zoom > 5 else 0
        self._replot()

    def zoom_out(self):
        self.xy_zoom += 5
        self._replot()

    def zoom_in_frenet(self):
        self.frenet_zoom -= 5 if self.frenet_zoom > 5 else 0 
        self._replot()

    def zoom_out_frenet(self):
        self.frenet_zoom += 5 
        self._replot()


    def set_wp(self):
        print("value is ", self.timestep_entry.get())
        timestep_value = int(self.timestep_entry.get())
        self.pl.reset(wp=timestep_value)
        self._replot()

    def replan(self):
        t1 = time.time()
        self.pl.replan()
        t2 = time.time()
        self.plan_time_labl.config(text=f"Plan Time: {(t2-t1)*1000:.2f} ms")
        self._replot()

    def start_animation(self):
        # Placeholder for the method to start animation
        pass
    def stop_animation(self):
        # Placeholder for the method to start animation
        pass

    def step_plan(self):
        # Placeholder for the method to step to the next waypoint
        t1 = time.time()
        self.pl.step()
        logging.info(f"Step Time: {(time.time()-t1)*1000:.2f} ms")
        self.sim.set_state(self.pl.xdata[-1], self.pl.ydata[-1])
        self.vehicle_state_label.config(text=f"Vehicle State: X: {self.sim.x:.2f}, Y: {self.sim.y:.2f}, Speed: {self.sim.speed:.2f}, Theta: {self.sim.theta:.2f}")
        self.timestep_entry.delete(0, tk.END)
        self.timestep_entry.insert(0, str(self.pl.race_trajectory.next_wp-1)) 
        self._replot()


    def step_control(self):
        d = self.pl.past_d[-1]
        logging.info(d)
        steer = self.controller.control(d)
        logging.info(f"Steering Angle: {steer}")
        self.sim.step(steering_angle=steer)
        self._replot()
        

    def step_steer_left(self):
        dt = float(self.dt_entry.get())
        self.sim.step(dt = dt, steering_angle = 0.1)
        self._replot()

    def step_steer_right(self):
        dt = float(self.dt_entry.get())
        self.sim.step(dt = dt, steering_angle = -0.1)
        self._replot()
        
    def step_acc(self):
        dt = float(self.dt_entry.get())
        self.sim.step(dt = dt, acceleration = 0.1)
        self._replot()

    def step_dec(self):
        dt = float(self.dt_entry.get())
        self.sim.step(dt = dt, acceleration = -0.1)
        self._replot()
    


class TextHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, msg + '\n')
        self.text_widget.configure(state='disabled')
        self.text_widget.yview(tk.END)


def load():
    config_file = "/home/mkhonji/workspaces/A2RL_Integration/config/config.yaml"
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
        path_to_track = config_data["path_to_track"]

    pl = Planner(path_to_track)
    return pl

def main():
    pl = load()
    # pid.test()
    
    app = PlotApp(pl)
    app.mainloop()  
if __name__ == "__main__":
    main()

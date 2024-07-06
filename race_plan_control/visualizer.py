import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from plan.planner import Planner
import yaml
import util
import control.pid as pid

class PlotApp(tk.Tk):
    def __init__(self, pl):
        super().__init__()
        self.pl = pl
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
        frenet_frame = ttk.Frame(self.visualize_frame)
        frenet_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(frenet_frame, text="Frenet Coordinate:").pack(anchor=tk.W, side=tk.LEFT)
        ttk.Button(frenet_frame, text="Zoom In", command=self.zoom_in_frenet).pack(side=tk.LEFT)
        ttk.Button(frenet_frame, text="Zoom Out", command=self.zoom_out_frenet).pack(side=tk.LEFT)
        coordinate_frame = ttk.Frame(self.visualize_frame)
        coordinate_frame.pack(fill=tk.X, padx=5, pady=5)

        # Control frame
        self.control_frame = ttk.LabelFrame(self, text="Control (Dummy Planner)")
        self.control_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)

        ## UI Elements for Control
        wp_frame = ttk.Frame(self.control_frame)
        wp_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(wp_frame, text="Set Waypoint", command=self.set_wp).pack(side=tk.LEFT)
        self.timestep_entry = ttk.Entry(wp_frame)
        self.timestep_entry.pack(side=tk.LEFT)

        ttk.Button(self.control_frame, text="Animate", command=self.start_animation).pack(side=tk.LEFT)
        ttk.Button(self.control_frame, text="Stop", command=self.stop_animation).pack(side=tk.LEFT)
        ttk.Button(self.control_frame, text="Replan", command=self.replan).pack(side=tk.LEFT, padx=30)
        ttk.Button(self.control_frame, text="Step", command=self.step_to_next_wp).pack(side=tk.LEFT)
        ttk.Button(self.control_frame, text="Prev", command=self.step_to_prev_wp).pack(side=tk.LEFT)


        # Plotting setup
        # self.fig = Figure(figsize=(5, 4), dpi=100)
        # gs = GridSpec(2, 1, height_ratios=[3, 1], figure=self.fig)  # Adjust the height ratios as needed
        # self.ax1 = self.fig.add_subplot(gs[0])
        # self.ax2 = self.fig.add_subplot(gs[1])  


        self.xy_zoom = 30
        self.frenet_zoom = 15
        self.fig = util.fig
        self.ax1 = util.ax1
        self.ax2 = util.ax2
        # self.pl.plot()
        util.plot(self.pl)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)  # A tk.DrawingArea.
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

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
        self.xy_zoom -= 10 if self.pl.xy_zoom > 10 else 0
        util.plot(self.pl, xy_zoom=self.xy_zoom, frenet_zoom=self.frenet_zoom)
        self.canvas.draw()

    def zoom_out(self):
        self.xy_zoom += 10
        util.plot(self.pl, xy_zoom=self.xy_zoom, frenet_zoom=self.frenet_zoom)
        self.canvas.draw()

    def zoom_in_frenet(self):
        self.frenet_zoom -= 10 if self.frenet_zoom > 10 else 0 
        util.plot(self.pl, xy_zoom=self.xy_zoom, frenet_zoom=self.frenet_zoom)
        self.canvas.draw()

    def zoom_out_frenet(self):
        self.frenet_zoom += 10
        util.plot(self.pl, xy_zoom=self.xy_zoom, frenet_zoom=self.frenet_zoom)
        self.canvas.draw()
    


    def set_wp(self):
        print("value is ", self.timestep_entry.get())
        timestep_value = int(self.timestep_entry.get())
        self.pl.step_idx = timestep_value

    def replan(self):
        self.pl.replan()
        util.plot(self.pl, xy_zoom=self.xy_zoom, frenet_zoom=self.frenet_zoom)
        self.canvas.draw()
    
    def start_animation(self):
        # Placeholder for the method to start animation
        pass
    def stop_animation(self):
        # Placeholder for the method to start animation
        pass

    def step_to_next_wp(self):
        # Placeholder for the method to step to the next waypoint
        self.pl.step()
        util.plot(self.pl, xy_zoom=self.xy_zoom, frenet_zoom=self.frenet_zoom)
        self.canvas.draw()
    def step_to_prev_wp(self):
        # Placeholder for the method to step to the previous waypoint
        pass


def load():
    config_file = "/home/mkhonji/workspaces/A2RL_Integration/config/config.yaml"
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
        path_to_track = config_data["path_to_track"]

    pl = Planner(path_to_track)
    return pl

def main():
    pl = load()
    pid.test()
    
    app = PlotApp(pl)
    app.mainloop()  
if __name__ == "__main__":
    main()
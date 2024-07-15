import sys
sys.path.append("/home/mkhonji/workspaces/race_plan_control/race_plan_control")

from execute import plot
from execute.executer import Executer

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from plan.planner import Planner
from control.controller import Controller

import time
import logging
from tkinter.scrolledtext import ScrolledText

class PlotApp(tk.Tk):
    def __init__(self, pl:Planner, controller:Controller, sim:Executer,  only_visualize=False):
        super().__init__()
        self.pl = pl
        self.cn = controller
        self.sim = sim;
        
        self.title("Path Planning Visualization")
        self.geometry("1200x900")


        # Variables for checkboxes
        self.show_past_locations = tk.BooleanVar(value=True)
        self.show_last_100_locations = tk.BooleanVar(value=True)
        self.show_boundaries = tk.BooleanVar(value=True)
        self.animation_running = False

        self.sim_option = tk.StringVar(value="Simple")  # Default selection


        #--------------------------------------------------------------------------------------------
        #-Plot Frame --------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------------

        self.plot_frame = ttk.Frame(self)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

        # Visualize frame setup
        self.visualize_frame = ttk.LabelFrame(self, text="Visualize")
        self.visualize_frame.pack(fill=tk.X, side=tk.TOP, padx=10, pady=5)


        ## UI Elements for Visualize - Checkboxes
        checkboxes_frame = ttk.Frame(self.visualize_frame)
        checkboxes_frame.pack(fill=tk.X, padx=5)
        ttk.Checkbutton(checkboxes_frame, text="Show Past Locations", variable=self.show_past_locations, command=self.update_plot).pack(anchor=tk.W, side=tk.LEFT)

        self.coordinates_label = ttk.Label(checkboxes_frame, text="")
        self.coordinates_label.pack(side=tk.RIGHT)

        ## UI Elements for Visualize - Buttons
        global_frame = ttk.Frame(self.visualize_frame)
        global_frame.pack(fill=tk.X, padx=5)

        ttk.Label(global_frame, text="Global Coordinate:").pack(anchor=tk.W, side=tk.LEFT)
        ttk.Button(global_frame, text="Zoom In", command=self.zoom_in).pack(side=tk.LEFT)
        ttk.Button(global_frame, text="Zoom Out", command=self.zoom_out).pack(side=tk.LEFT)
        self.vehicle_state_label = ttk.Label(global_frame, text="")
        self.vehicle_state_label.pack(side=tk.RIGHT)
        frenet_frame = ttk.Frame(self.visualize_frame)
        frenet_frame.pack(fill=tk.X, padx=5)
        ttk.Label(frenet_frame, text="Frenet Coordinate:").pack(anchor=tk.W, side=tk.LEFT)
        ttk.Button(frenet_frame, text="Zoom In", command=self.zoom_in_frenet).pack(side=tk.LEFT)
        ttk.Button(frenet_frame, text="Zoom Out", command=self.zoom_out_frenet).pack(side=tk.LEFT)
        
        #------------------------------------------------------
        #-Plan Control Exec Frame -----------------------------
        #------------------------------------------------------

        self.plan_sim_control_frame = ttk.Frame(self)
        self.plan_sim_control_frame.pack(fill=tk.X)

        ## Plan frame
        self.plan_frame = ttk.LabelFrame(self.plan_sim_control_frame, text="Plan (Manual)")
        self.plan_frame.pack(fill=tk.X,side=tk.LEFT, padx=10, pady=5)
        
        wp_frame = ttk.Frame(self.plan_frame)
        wp_frame.pack(fill=tk.X)

        ttk.Button(wp_frame, text="Set Waypoint", command=self.set_waypoint).pack(side=tk.LEFT)
        self.timestep_entry = ttk.Entry(wp_frame, width=6)
        self.timestep_entry.insert(0, "0")
        self.timestep_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(wp_frame, text=f"{len(self.pl.reference_path)-1}").pack(side=tk.LEFT, padx=5)

        ttk.Button(self.plan_frame, text="Replan", command=self.replan).pack(side=tk.LEFT)
        ttk.Button(self.plan_frame, text="Step", command=self.step_plan).pack(side=tk.LEFT,fill=tk.X, expand=True)


        ## Control Frame
        self.control_frame = ttk.LabelFrame(self.plan_sim_control_frame, text="Control (Manual)")
        self.control_frame.pack(fill=tk.X, side=tk.LEFT)
        dt_frame = ttk.Frame(self.control_frame)
        dt_frame.pack(fill=tk.X)
        ttk.Label(dt_frame, text="Δt ").pack(side=tk.LEFT, padx=5, pady=5)
        self.dt_entry = ttk.Entry(dt_frame, width=5)
        self.dt_entry.insert(2, "0.1")
        self.dt_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(dt_frame, text="Control Step", command=self.step_control).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Button(self.control_frame, text="Steer Left", command=self.step_steer_left).pack(side=tk.LEFT)
        ttk.Button(self.control_frame, text="Steer Right", command=self.step_steer_right).pack(side=tk.LEFT)
        ttk.Button(self.control_frame, text="Accelerate", command=self.step_acc).pack(side=tk.LEFT)
        ttk.Button(self.control_frame, text="Deccelerate", command=self.step_dec).pack(side=tk.LEFT)


        #-----------
        ## Execute Frame
        self.simulate_frame = ttk.LabelFrame(self.plan_sim_control_frame, text="Execute (Auto)")
        self.simulate_frame.pack(fill=tk.BOTH,expand=True, side=tk.LEFT, padx=10, pady=5)

        sim_first_frame = ttk.Frame(self.simulate_frame)
        sim_first_frame.pack(fill=tk.X)
        sim_second_frame = ttk.Frame(self.simulate_frame)
        sim_second_frame.pack(fill=tk.X)

        ttk.Label(sim_first_frame, text="Δt ").pack(side=tk.LEFT, padx=5, pady=5)
        self.dt_sim_entry = ttk.Entry(sim_first_frame, width=5)
        self.dt_sim_entry.insert(0, "0.02")
        self.dt_sim_entry.pack(side=tk.LEFT)
        
        ttk.Radiobutton(sim_first_frame, text="Simple", variable=self.sim_option, value="Simple", command=self.update_plot).pack(side=tk.RIGHT)
        ttk.Radiobutton(sim_first_frame, text="ROS", variable=self.sim_option, value="ROS", command=self.update_plot).pack(side=tk.RIGHT)
        ttk.Radiobutton(sim_first_frame, text="Carla", variable=self.sim_option, value="Carla", command=self.update_plot).pack(side=tk.RIGHT)
        
        self.start_sim_button = ttk.Button(sim_second_frame, text="Start", command=self.start_sim)
        self.start_sim_button.pack(fill=tk.X, side=tk.LEFT, expand=True)
        ttk.Button(sim_second_frame, text="Stop", command=self.stop_sim).pack(side=tk.LEFT)
        ttk.Button(sim_second_frame, text="Reset", command=self.reset_sim).pack(side=tk.LEFT)
        
        #-----------
        
        #-----------------------------------------------------------------------------------------------
        #-End of Plan Contorl Sim Frame ----------------------------------------------------------------
        #-----------------------------------------------------------------------------------------------
        
        # Log Frame
        self.log_frame = ttk.LabelFrame(self, text="Log")
        self.log_frame.pack(fill=tk.X, padx=10, pady=5)
        # self.log_frame.pack(fill=tk.X, side=tk.TOP, expand=True, padx=10, pady=5)
        log_area = ScrolledText(self.log_frame, state='disabled', height=8)
        log_area.pack(side=tk.LEFT, fill=tk.BOTH,expand=True)


        #-----------------------------------------------------------------------------------------------
        #-End of UI Elements----------------------------------------------------------------------------
        #-----------------------------------------------------------------------------------------------

        self.xy_zoom = 30
        self.frenet_zoom = 30
        self.fig = plot.fig
        self.ax1 = plot.ax1
        self.ax2 = plot.ax2
        

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)  # A tk.DrawingArea.
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.after(1000, self._replot)


        # Disable all the buttons if only_visualize is set to True        
        if only_visualize:
            for child in self.plan_sim_control_frame.winfo_children():
                try:
                    print(child)
                    child.configure(state='disabled')
                except tk.TclError as e:
                    pass
        
        # Configure logging
        logger = logging.getLogger()
        text_handler = TextHandler(log_area)
        formatter = logging.Formatter('%(module)s (%(filename)s L: %(lineno)d) [%(levelname)s]: %(message)s')
        text_handler.setFormatter(formatter)
        logger.addHandler(text_handler)
        logger.setLevel(logging.INFO)
        
        logging.info("Log initialized.")            

    def _replot(self):
        canvas_widget = self.canvas.get_tk_widget()
        width = canvas_widget.winfo_width()
        height = canvas_widget.winfo_height()
        aspect_ratio = width/height
        # logging.info(f"Canvas Size: {width}x{height} px [aspect ratio: {aspect_ratio:.2f}]")

        plot.plot(self.pl,self.sim, aspect_ratio, xy_zoom=self.xy_zoom, frenet_zoom=self.frenet_zoom)
        self.canvas.draw()
        self.vehicle_state_label.config(text=f"Vehicle State: X: {self.sim.state.x:.2f}, Y: {self.sim.state.y:.2f}, Speed: {self.sim.state.speed:.2f}, Theta: {self.sim.state.theta:.2f}")




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

    #--------------------------------------------------------------------------------------------
    #-Plan---------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------

    def set_waypoint(self):
        timestep_value = int(self.timestep_entry.get())
        self.pl.reset(wp=timestep_value)
        self._replot()

    def replan(self):
        t1 = time.time()
        self.pl.replan()
        t2 = time.time()
        logging.info(f"Plan Time: {(t2-t1)*1000:.2f} ms")
        self._replot()


    def step_plan(self):
        # Placeholder for the method to step to the next waypoint
        t1 = time.time()
        self.pl.step()
        logging.info(f"Step Time: {(time.time()-t1)*1000:.2f} ms")
        self.timestep_entry.delete(0, tk.END)
        self.timestep_entry.insert(0, str(self.pl.race_trajectory.next_wp-1)) 
        self._replot()

    #--------------------------------------------------------------------------------------------
    #-Control------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------

    def step_control(self):
        d = self.pl.past_d[-1]
        logging.info(d)
        steer = self.cn.control(d)
        logging.info(f"Steering Angle: {steer}")
        
        dt = float(self.dt_entry.get())
        self.sim.update(steering_angle=steer, dt=dt)
        self._replot()
        

    def step_steer_left(self):
        dt = float(self.dt_entry.get())
        self.sim.update(dt = dt, steering_angle = 0.1)
        self._replot()

    def step_steer_right(self):
        dt = float(self.dt_entry.get())
        self.sim.update(dt = dt, steering_angle = -0.1)
        self._replot()
        
    def step_acc(self):
        dt = float(self.dt_entry.get())
        self.sim.update(dt = dt, acceleration = 1)
        self._replot()

    def step_dec(self):
        dt = float(self.dt_entry.get())
        self.sim.update(dt = dt, acceleration = -1)
        self._replot()
    
    
    #--------------------------------------------------------------------------------------------
    #-SIM----------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------

    def start_sim(self):
        self.animation_running = True
        self.start_sim_button.config(state=tk.DISABLED)
        self._sim_loop()

    def _sim_loop(self):
        if self.animation_running:
            sec = float(self.dt_sim_entry.get())
            self.sim.run(dt=sec)
            self._replot()
            self.after(int(sec*1000), self._sim_loop)

    def stop_sim(self):
        self.animation_running = False  
        self.start_sim_button.config(state=tk.NORMAL)
    
    def reset_sim(self):
        self.sim.reset()
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



import main
if __name__ == "__main__":
    main.main()

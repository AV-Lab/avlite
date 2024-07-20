from race_plan_control.plan.planner import Planner
from race_plan_control.control.controller import Controller
import race_plan_control.execute.plot as plot
from race_plan_control.execute.executer import Executer


import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import logging
from tkinter.scrolledtext import ScrolledText

log = logging.getLogger(__name__)
log_blacklist = set() # used to filter 'excute', 'plan', 'control' subpackage logs

class PlotApp(tk.Tk):
    def __init__(self, pl:Planner, controller:Controller, exec:Executer,  only_visualize=False):
        super().__init__()
        self.pl = pl
        self.cn = controller
        self.exec = exec;
        
        self.title("Path Planning Visualization")
        self.geometry("1200x1100")


        #--------------------------------------------------------------------------------------------
        # Variables for checkboxes ------------------------------------------------------------------
        #--------------------------------------------------------------------------------------------
        self.show_past_locations = tk.BooleanVar(value=True)
        self.show_last_100_locations = tk.BooleanVar(value=True)
        self.show_boundaries = tk.BooleanVar(value=True)
        self.animation_running = False

        self.exec_option = tk.StringVar(value="Simple") 
        self.debug_option = tk.StringVar(value="INFO")  
        
        self.show_plan_logs = tk.BooleanVar(value=True)
        self.show_control_logs = tk.BooleanVar(value=True)
        self.show_execute_logs = tk.BooleanVar(value=True)


        #--------------------------------------------------------------------------------------------
        #-Plot Frame --------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------------

        self.plot_frame = ttk.Frame(self)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

        # Visualize frame setup
        #------------------------------------------------------
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

        ttk.Label(global_frame, text="Global Coordinate").pack(anchor=tk.W, side=tk.LEFT)
        ttk.Button(global_frame, text="Zoom In", command=self.zoom_in).pack(side=tk.LEFT)
        ttk.Button(global_frame, text="Zoom Out", command=self.zoom_out).pack(side=tk.LEFT)
        self.vehicle_state_label = ttk.Label(global_frame, text="")
        self.vehicle_state_label.pack(side=tk.RIGHT)
        frenet_frame = ttk.Frame(self.visualize_frame)
        frenet_frame.pack(fill=tk.X, padx=5)
        ttk.Label(frenet_frame, text="Frenet Coordinate").pack(anchor=tk.W, side=tk.LEFT)
        ttk.Button(frenet_frame, text="Zoom In", command=self.zoom_in_frenet).pack(side=tk.LEFT)
        ttk.Button(frenet_frame, text="Zoom Out", command=self.zoom_out_frenet).pack(side=tk.LEFT)
        
        #-----------------------------------------------------------
        #-Plan Control Exec Frame ----------------------------------
        #-----------------------------------------------------------

        self.plan_sim_control_frame = ttk.Frame(self)
        self.plan_sim_control_frame.pack(fill=tk.X)

        ## Plan frame
        #-----------------------------------------------------------
        self.plan_frame = ttk.LabelFrame(self.plan_sim_control_frame, text="Plan (Manual)")
        self.plan_frame.pack(fill=tk.X,side=tk.LEFT, padx=10, pady=5)
        
        wp_frame = ttk.Frame(self.plan_frame)
        wp_frame.pack(fill=tk.X)

        ttk.Button(wp_frame, text="Set Waypoint", command=self.set_waypoint).pack(side=tk.LEFT)
        self.global_tj_wp_entry = ttk.Entry(wp_frame, width=6)
        self.global_tj_wp_entry.insert(0, "0")
        self.global_tj_wp_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(wp_frame, text=f"{len(self.pl.reference_path)-1}").pack(side=tk.LEFT, padx=5)

        ttk.Button(self.plan_frame, text="Replan", command=self.replan).pack(side=tk.LEFT)
        ttk.Button(self.plan_frame, text="Step", command=self.step_plan).pack(side=tk.LEFT,fill=tk.X, expand=True)


        ## Control Frame
        #-------------------------------------------------------
        self.control_frame = ttk.LabelFrame(self.plan_sim_control_frame, text="Control (Manual)")
        self.control_frame.pack(fill=tk.X, side=tk.LEFT)
        dt_frame = ttk.Frame(self.control_frame)
        dt_frame.pack(fill=tk.X)
        ttk.Label(dt_frame, text="Δt ").pack(side=tk.LEFT, padx=5, pady=5)
        self.dt_entry = ttk.Entry(dt_frame, width=5)
        self.dt_entry.insert(2, "0.1")
        self.dt_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(dt_frame, text="Control Step", command=self.step_control).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(dt_frame, text="Re-align", command=self.align_control).pack(side=tk.LEFT) # Re-alignes with plan
        
        ttk.Button(self.control_frame, text="Steer Left", command=self.step_steer_left).pack(side=tk.LEFT)
        ttk.Button(self.control_frame, text="Steer Right", command=self.step_steer_right).pack(side=tk.LEFT)
        ttk.Button(self.control_frame, text="Accelerate", command=self.step_acc).pack(side=tk.LEFT)
        ttk.Button(self.control_frame, text="Deccelerate", command=self.step_dec).pack(side=tk.LEFT)


        ## Execute Frame
        #-------------------------------------------------------
        self.execution_frame = ttk.LabelFrame(self.plan_sim_control_frame, text="Execute (Auto)")
        self.execution_frame.pack(fill=tk.BOTH,expand=True, side=tk.LEFT, padx=10, pady=5)

        exec_first_frame = ttk.Frame(self.execution_frame)
        exec_first_frame.pack(fill=tk.X)
        exec_second_frame = ttk.Frame(self.execution_frame)
        exec_second_frame.pack(fill=tk.X)

        ttk.Label(exec_first_frame, text="Control Δt ").pack(side=tk.LEFT, padx=5, pady=5)
        self.dt_exec_cn_entry = ttk.Entry(exec_first_frame, width=5)
        self.dt_exec_cn_entry.insert(0, "0.02")
        self.dt_exec_cn_entry.pack(side=tk.LEFT)
        
        ttk.Label(exec_first_frame, text="Replan Δt ").pack(side=tk.LEFT, padx=5, pady=5)
        self.dt_exec_pl_entry = ttk.Entry(exec_first_frame, width=5)
        self.dt_exec_pl_entry.insert(0, ".7")
        self.dt_exec_pl_entry.pack(side=tk.LEFT)
        
        ttk.Radiobutton(exec_first_frame, text="Simple", variable=self.exec_option, value="Simple", command=self.update_plot).pack(side=tk.RIGHT)
        ttk.Radiobutton(exec_first_frame, text="ROS", variable=self.exec_option, value="ROS", command=self.update_plot).pack(side=tk.RIGHT)
        ttk.Radiobutton(exec_first_frame, text="Carla", variable=self.exec_option, value="Carla", command=self.update_plot).pack(side=tk.RIGHT)
        ttk.Label(exec_first_frame, text="Interface:").pack(side=tk.RIGHT)
        
        self.start_sim_button = ttk.Button(exec_second_frame, text="Start", command=self.start_exec)
        self.start_sim_button.pack(fill=tk.X, side=tk.LEFT, expand=True)
        ttk.Button(exec_second_frame, text="Stop", command=self.stop_sim).pack(side=tk.LEFT)
        ttk.Button(exec_second_frame, text="Step", command=self.step_sim).pack(side=tk.LEFT)
        ttk.Button(exec_second_frame, text="Reset", command=self.reset_sim).pack(side=tk.LEFT)
        
        #-----------------------------------------------------------------------------------------------
        #-End of Plan Contorl Exec Frame ---------------------------------------------------------------
        #-----------------------------------------------------------------------------------------------
        
        # Log Frame
        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.pack(fill=tk.X, padx=10, pady=5)

        log_cb_frame = ttk.Frame(log_frame)
        log_cb_frame.pack(fill=tk.X)
        pl_logs = ttk.Checkbutton(log_cb_frame, text="Plan Logs", variable=self.show_plan_logs, command=self.update_log)
        pl_logs.pack(side=tk.LEFT)
        pl_logs.var = self.show_plan_logs
        self.show_plan_logs.set(False)

        ttk.Checkbutton(log_cb_frame, text="Control Logs", variable=self.show_control_logs, command=self.update_log).pack(side=tk.LEFT)
        ttk.Checkbutton(log_cb_frame, text="Execute Logs", variable=self.show_execute_logs, command=self.update_log).pack(side=tk.LEFT)
        
        ttk.Radiobutton(log_cb_frame, text="None", variable=self.debug_option, value="None", command=self.update_plot).pack(side=tk.RIGHT)
        ttk.Radiobutton(log_cb_frame, text="INFO", variable=self.debug_option, value="INFO", command=self.update_plot).pack(side=tk.RIGHT)
        ttk.Radiobutton(log_cb_frame, text="DEBUG", variable=self.debug_option, value="DEBUG", command=self.update_plot).pack(side=tk.RIGHT)
        ttk.Label(log_cb_frame, text="Log Level:").pack(side=tk.RIGHT)
                                                                                                                        
        log_area = ScrolledText(log_frame, state='disabled', height=8)
        log_area.pack(fill=tk.BOTH,expand=True)


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
        self.after(500, self._replot)


        # Disable all the buttons if only_visualize is set to True        
        if only_visualize:
            for child in self.plan_sim_control_frame.winfo_children():
                try:
                    print(child)
                    child.configure(state='disabled')
                except tk.TclError as e:
                    pass
        #------------------------------------------- 
        #-Configure logging-------------------------
        #------------------------------------------- 
        logger = logging.getLogger()
        text_handler = PlotApp.LogTextHandler(log_area)
        formatter = logging.Formatter('[%(levelname).4s] %(name)-40s (L: %(lineno)3d): %(message)s')
        text_handler.setFormatter(formatter)
        logger.addHandler(text_handler)
        logger.setLevel(logging.INFO) 
        log.info("Log initialized.")            

    def _replot(self):
        canvas_widget = self.canvas.get_tk_widget()
        width = canvas_widget.winfo_width()
        height = canvas_widget.winfo_height()
        aspect_ratio = width/height

        plot.plot(self.pl,self.exec, aspect_ratio, xy_zoom=self.xy_zoom, frenet_zoom=self.frenet_zoom)
        self.canvas.draw()
        self.vehicle_state_label.config(text=f"Vehicle State: X: {self.exec.state.x:.2f}, Y: {self.exec.state.y:.2f}, Speed: {self.exec.state.speed:.2f}, Theta: {self.exec.state.theta:.2f}")




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
        timestep_value = int(self.global_tj_wp_entry.get())
        self.pl.reset(wp=timestep_value)
        self._replot()

    def replan(self):
        t1 = time.time()
        self.pl.replan()
        t2 = time.time()
        log.info(f"Plan Time: {(t2-t1)*1000:.2f} ms")
        self._replot()


    def step_plan(self):
        # Placeholder for the method to step to the next waypoint
        t1 = time.time()
        self.pl.step_wp()
        log.info(f"Step Time: {(time.time()-t1)*1000:.2f} ms")
        self.global_tj_wp_entry.delete(0, tk.END)
        self.global_tj_wp_entry.insert(0, str(self.pl.global_trajectory.next_wp-1)) 
        self._replot()

    #--------------------------------------------------------------------------------------------
    #-Control------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------

    def step_control(self):
        d = self.pl.past_d[-1]
        steer = self.cn.control(d)
        
        dt = float(self.dt_entry.get())
        self.exec.update_state(steering_angle=steer, dt=dt)
        self._replot()
    def align_control(self):
        self.exec.state.x = self.pl.global_trajectory.reference_x[self.pl.global_trajectory.next_wp - 1]
        self.exec.state.y = self.pl.global_trajectory.reference_y[self.pl.global_trajectory.next_wp - 1]

        self._replot()  

    def step_steer_left(self):
        dt = float(self.dt_entry.get())
        self.exec.update_state(dt = dt, steering_angle = 0.1)
        self._replot()

    def step_steer_right(self):
        dt = float(self.dt_entry.get())
        self.exec.update_state(dt = dt, steering_angle = -0.1)
        self._replot()
        
    def step_acc(self):
        dt = float(self.dt_entry.get())
        self.exec.update_state(dt = dt, acceleration = 1)
        self._replot()

    def step_dec(self):
        dt = float(self.dt_entry.get())
        self.exec.update_state(dt = dt, acceleration = -1)
        self._replot()
    
    
    #--------------------------------------------------------------------------------------------
    #-SIM----------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------

    def start_exec(self):
        self.animation_running = True
        self.start_sim_button.config(state=tk.DISABLED)
        self._exec_loop()

    def _exec_loop(self):
        if self.animation_running:
            cn_dt = float(self.dt_exec_cn_entry.get())
            pl_dt = float(self.dt_exec_pl_entry.get())

            self.exec.run(control_dt=cn_dt, replan_dt=pl_dt)
            self._replot()
            self.global_tj_wp_entry.delete(0, tk.END)
            self.global_tj_wp_entry.insert(0, str(self.pl.global_trajectory.next_wp-1)) 
            self.after(int(cn_dt*1000), self._exec_loop)

    def stop_sim(self):
        self.animation_running = False  
        self.start_sim_button.config(state=tk.NORMAL)
    
    def step_sim(self):
        cn_dt = float(self.dt_exec_cn_entry.get())
        pl_dt = float(self.dt_exec_pl_entry.get())
        self.exec.run(control_dt=cn_dt, replan_dt=pl_dt)
        self._replot()

    def reset_sim(self):
        self.exec.reset()
        self._replot()

    def update_log(self):
        if self.show_plan_logs.get():
            log_blacklist.discard('plan')
        else:
            log_blacklist.add('plan')

        if self.show_control_logs.get():
            log_blacklist.discard('control')
        else:   
            log_blacklist.add('control')

        if self.show_execute_logs.get():
            log_blacklist.discard('execute')
        else:
            log_blacklist.add('execute')


    class LogTextHandler(logging.Handler):
        def __init__(self, text_widget):
            super().__init__()
            self.text_widget = text_widget

        def emit(self, record):
            for bl in log_blacklist:
                if bl+"." in record.name:
                    return
            msg = self.format(record)
            self.text_widget.configure(state='normal')
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.configure(state='disabled')
            self.text_widget.yview(tk.END)

if __name__ == "__main__":
    import race_plan_control.main as main
    main.run()

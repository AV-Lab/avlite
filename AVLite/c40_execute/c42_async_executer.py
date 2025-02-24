from __future__ import annotations
from c10_perceive.c11_perception_model import PerceptionModel
from c20_plan.c21_base_planner import BasePlanner
from c20_plan.c24_trajectory import Trajectory
from c30_control.c31_base_controller import BaseController, ControlComand
from c10_perceive.c12_state import EgoState
from c40_execute.c41_executer import Executer, WorldInterface

import multiprocessing as mp
from multiprocessing.managers import BaseManager
from multiprocessing import Value, Lock, Queue
import time

import logging
from logging.handlers import QueueHandler, QueueListener


log = logging.getLogger(__name__)



class AsyncExecuter(Executer):
    def __init__(
        self,
        pm: PerceptionModel,
        pl: BasePlanner,
        cn: BaseController,
        world: WorldInterface,
        call_replan=True,
        call_control=True,
        call_perceive=False,
        replan_dt=0.5,
        control_dt=0.05,
    ):
        super().__init__(pm, pl, cn, world, replan_dt=replan_dt, control_dt=control_dt)
        BaseManager.register("BasePlanner", 
                    callable=lambda: self.planner.get_copy(),
                    exposed=("replan", "step", "get_serializable_local_plan", "get_location_xy", "get_location_sd", "get_replan_dt", "set_replan_dt"))
        BaseManager.register('BaseController', 
                    callable=lambda: cn.get_copy(),
                    exposed=('update_serializable_trajectory', 'control', 'get_control_dt', 'set_control_dt'))
        BaseManager.register('WorldInterface', 
                    callable=lambda: world.get_copy(),
                    exposed=('get_ego_state', 'update_ego_state'))
        


        self.manager = BaseManager()
        self.manager.start()

        # self.shared_ego_state = self.manager.EgoState()
        self.shared_planner = self.manager.BasePlanner()
        self.shared_controller = self.manager.BaseController()
        self.shared_world = self.manager.WorldInterface()
        # self.shared_trajectory = self.manager.Trajectory()

        self.__planner_last_replan_time = Value("d", time.time())  # Shared double variable
        self.__planner_elapsed_time = Value("d", 0.0)  # Shared double variable
        self.__planner_start_time = Value("d", time.time())  # Shared double variable
        self.__controller_last_step_time = Value("d", 0.0)  # Shared double variable

        self.lock_planner = Lock()
        self.lock_controller = Lock()
        self.lock_world = Lock()


        self.__log_queue = Queue()
        self.__queue_listener = QueueListener(
            self.__log_queue, logging.getLogger().handlers[0]  # Forward to default handler
        )
        self.__queue_listener.start()
        self.setup_process_logging()

        self.call_replan = call_replan
        self.call_control = call_control
        self.call_perceive = call_perceive

        self.processes = []
        self.processes_started = False

        self.create_processes()

    def step(self, control_dt=0.01, replan_dt=0.01, call_replan=True, call_control=True, call_perceive=False):
        self.control_dt = control_dt
        self.replan_dt = replan_dt

        if self.processes_started == False:
            self.create_processes()
            self.start_processes()
            return
        elif all(not p.is_alive() for p in self.processes) and self.processes_started == True:
            log.warning(f"All processes are dead. Recreating and starting processes.")
            self.stop()
            self.create_processes()
            self.start_processes()
            return
        elif any(p.is_alive() for p in self.processes) and not all(p.is_alive() for p in self.processes):
            # UI coordination tasks
            log.error(
                    f"Some Async Executer Processes are dead! Planner status: {self.planner_process.is_alive()}, Controller status: {self.controller_process.is_alive()} . Call stop() to terminate all processes."
            )
            # self.stop()
            return


        with self.lock_world:
            self.ego_state = self.shared_world.get_ego_state()

        with self.lock_planner:
            self.planner.location_xy = self.shared_planner.get_location_xy()
            self.planner.location_sd = self.shared_planner.get_location_sd()
        
        if self.shared_planner.get_replan_dt() > 0 and self.shared_controller.get_control_dt() > 0:
            log.info(f"planner dt: {1/self.shared_planner.get_replan_dt():.2f} fps, control dt: {1/self.shared_controller.get_control_dt():.2f} fps")
            self.planner_fps = int(1/self.shared_planner.get_replan_dt())
            self.control_fps = int(1/self.shared_controller.get_control_dt())


    def worker_planner(self,*args):
        time.sleep(self.replan_dt)
        log.info(f"Plan Worker Started")
        log.info(f"replan dt:  {self.replan_dt}")

        while True:
            t1 = time.time()
            with self.lock_planner:
                dt = t1 - self.__planner_last_replan_time.value
                self.__planner_elapsed_time.value += time.time() - self.__planner_start_time.value
                if dt > self.replan_dt:
                    self.__planner_last_replan_time.value = time.time()
                    self.shared_planner.replan()
                    path,vel = self.shared_planner.get_serializable_local_plan()
                    with self.lock_controller:
                        self.shared_controller.update_serializable_trajectory(path, vel)
                        # self.__controller_ready.value = True

                with self.lock_world:
                    state = self.shared_world.get_ego_state()
                    self.shared_planner.step(state)

                self.shared_planner.set_replan_dt(time.time()-t1)

            t2 = time.time()
            sleep_time = max(0, self.replan_dt - (t2 - t1))
            time.sleep(sleep_time)
            log.info(f"Planner iteration actual step time {t2-t1:.3f}  - > sleep time: {sleep_time:.2f} s")


    def worker_controller(self, *args):
        time.sleep(self.replan_dt)
        log.info(f"Controller Worker Started")
        while True:
            t1 = time.time()
            # if self.__controller_ready.value:
            with self.lock_world:
                dt = t1 - self.__controller_last_step_time.value
                self.__controller_last_step_time.value = time.time()
                if dt > self.control_dt:
                    state = self.shared_world.get_ego_state()
                    cmd = self.shared_controller.control(state)
                    self.shared_world.update_ego_state(state, cmd, dt=0.01)
                    self.shared_controller.set_control_dt(time.time()-t1)
            t2 = time.time()
            sleep_time = max(0, self.control_dt - (t2 - t1))
            time.sleep(sleep_time)
            log.info(f"Controller iteration actual step time {t2-t1:.3f}  - > sleep time: {sleep_time:.2f} s")



    def worker_perceiver(self, *args):
        raise NotImplementedError


    def stop(self):
        count = 0
        for p in self.processes:
            if p.is_alive():
                log.info(f"Terminating process {p.name}")
                count += 1
                p.terminate()  # Force terminate the process
                p.join()  # Wait for process to finish
        log.info(f"Async Executer Processes Stopped. {count}/{len(self.processes)}  processes terminated.")
        self.processes = []
        self.planner_process = None
        self.controller_process = None
        self.processes_started = False

    def create_processes(self):
        self.processes = []
        self.planner_process = None
        self.controller_process = None
        # for i in range(2):
        # p = mp.Process(target=self._dummy_worker, args=(f"Process {i}",))
        # self.processes.append(p)

        if self.call_replan:
            self.planner_process = mp.Process(
                target=self.worker_planner,
                args=(
                    "Planner",
                    self.shared_world,
                    self.shared_planner,
                    self.shared_controller,
                    self.__planner_elapsed_time,
                    self.__planner_start_time,
                    self.__planner_last_replan_time
                ),
            )
            # p = mp.Process(target=self._plan_worker)
            self.processes.append(self.planner_process)

        if self.call_control:
            self.controller_process = mp.Process(target=self.worker_controller,
                args=(
                    "Controller",
                    self.__controller_last_step_time,
                    self.shared_world,
                ),
            )
            self.processes.append(self.controller_process)

    # TODO: fix this 
    def start_processes(self):
        if self.processes_started:
            log.warning("Processes already started. Call stop() to restart.")
            return
        if len(self.processes) < 2:
            log.warning("Not processes created to start. Call create_processes() first.")  
            return

        t1 = time.time()
        self.setup_process_logging()
        log.info(f"Starting Planner...")
        self.__planner_start_time = Value("d", time.time())  # Shared double variable
        self.planner_process.start()
        
        log.info(f"Starting Controller...")
        self.controller_process.start()
        
        # self.__planner_time_since_last_replan = Value("d", 0.0)  # Shared double variable
        self.processes_started = True
        log.info(f"Processes started in {time.time()-t1:.3f} s")

    def setup_process_logging(self):
        """Configure worker process to send logs to queue"""
        # Remove all handlers
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)

        # Add queue handler
        queue_handler = QueueHandler(self.__log_queue)
        root.addHandler(queue_handler)
        # root.setLevel(logging.INFO)

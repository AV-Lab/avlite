from __future__ import annotations
from c10_perception.c12_base_perception import PerceptionModel
from c20_planning.c23_base_local_planner import BaseLocalPlanner
from c30_control.c32_base_controller import BaseController
from c40_execution.c41_base_executer import BaseExecuter, WorldInterface

import multiprocessing as mp
from multiprocessing.managers import BaseManager
from multiprocessing import Value, Lock, Queue
import time

import logging
from logging.handlers import QueueHandler, QueueListener


log = logging.getLogger(__name__)


class AsyncExecuter(BaseExecuter):
    def __init__(
        self,
        pm: PerceptionModel,
        pl: BaseLocalPlanner,
        cn: BaseController,
        world: WorldInterface,
        call_replan=True,
        call_control=True,
        call_perceive=False,
        replan_dt=0.5,
        control_dt=0.05,
    ):
        super().__init__(pm, pl, cn, world, replan_dt=replan_dt, control_dt=control_dt)
        BaseManager.register(
            "BasePlanner",
            callable=lambda: self.local_planner.get_copy(),
            exposed=(
                "replan",
                "step",
                "get_serializable_local_plan",
                "get_location_xy",
                "get_location_sd",
                "get_replan_dt",
                "set_replan_dt",
                "get_copy"
            ),
        )
        BaseManager.register(
            "BaseController",
            callable=lambda: self.controller.get_copy(),
            exposed=(
                "update_serializable_trajectory",
                "control",
                "get_control_dt",
                "set_control_dt",
                "get_cte_steer",
                "get_cte_velocity",
                "get_cmd",
            ),
        )
        BaseManager.register(
            "WorldInterface", callable=lambda: world.get_copy(), exposed=("get_ego_state", "update_ego_state")
        )

        self.manager = BaseManager()
        self.manager.start()

        ####### Shared objects and Vars #######
        self.shared_planner = self.manager.BasePlanner()
        self.shared_controller = self.manager.BaseController()
        self.shared_world = self.manager.WorldInterface()

        self.__planner_last_step_time = Value("d", time.time())  # Shared double variable
        self.__planner_elapsed_time = Value("d", 0.0)  # Shared double variable
        self.__planner_start_time = Value("d", time.time())  # Shared double variable
        self.__controller_last_step_time = Value("d", 0.0)  # Shared double variable
        self.__kill_flag = Value("b", False)  # Shared boolean variable

        self.lock_planner = Lock()
        self.lock_controller = Lock()
        self.lock_world = Lock()
        ##############################

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

    def step(self, control_dt=0.01, replan_dt=0.01, sim_dt=0.01, call_replan=True, call_control=True, call_perceive=False):
        self.control_dt = control_dt
        self.replan_dt = replan_dt
        self.sim_dt = sim_dt
        self.call_replan = call_replan
        self.call_control = call_control
        self.call_perceive = call_perceive

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
            log.error(
                f"Some Async Executer Processes are dead! Planner status: {self.planner_process.is_alive()}, Controller status: {self.controller_process.is_alive()} . Call stop() to terminate all processes."
            )
            return

        with self.lock_world:
            self.ego_state = self.shared_world.get_ego_state()

        # with self.lock_planner:
        self.local_planner.location_xy = self.shared_planner.get_location_xy()
        self.local_planner.location_sd = self.shared_planner.get_location_sd()

        if self.shared_planner.get_replan_dt() > 0:
            log.info(
                f"planner dt: {1/self.shared_planner.get_replan_dt():.2f} fps, control dt: {self.shared_controller.get_control_dt():.2f} fps"
            )
            self.planner_fps = 1 / self.shared_planner.get_replan_dt()

        # with self.lock_controller:
        if self.shared_controller.get_control_dt() > 0:
            self.controller.cmd = self.shared_controller.get_cmd()
            self.controller.cte_steer = self.shared_controller.get_cte_steer()
            self.controller.cte_velocity = self.shared_controller.get_cte_velocity()
            self.control_fps = 1 / self.shared_controller.get_control_dt()

    # def worker_planner(self, *args):
    #     time.sleep(self.replan_dt)
    #     log.info(f"Plan Worker Started")
    #     log.info(f"replan dt:  {self.replan_dt}")
    #
    #     while not self.__kill_flag.value and self.call_replan:
    #         try:
    #             t1 = time.time()
    #             with self.lock_planner:
    #                 dt = t1 - self.__planner_last_step_time.value
    #                 self.__planner_elapsed_time.value += time.time() - self.__planner_start_time.value
    #                 if dt > self.replan_dt:
    #                     self.shared_planner.set_replan_dt(dt)
    #                     self.__planner_last_step_time.value = time.time()
    #                     self.shared_planner.replan()
    #                     path, vel = self.shared_planner.get_serializable_local_plan()
    #                 if dt > self.sim_dt:
    #                     with self.lock_controller:
    #                         self.shared_controller.update_serializable_trajectory(path, vel)
    #
    #                 with self.lock_world:
    #                     state = self.shared_world.get_ego_state()
    #                     self.shared_planner.step(state)
    #
    #
    #             t2 = time.time()
    #             sleep_time = max(0, self.replan_dt - (t2 - t1))
    #             time.sleep(sleep_time)
    #             log.info(f"Planner iteration actual step time {t2-t1:.3f}  - > sleep time: {sleep_time:.2f} s")
    #         except Exception as e:
    #             log.error(f"Error in planner worker: {e}", exc_info=True)
    #             time.sleep(0.1)  # Avoid tight loop if persistent errors
    def worker_planner(self, *args):
        log.info(f"Plan Worker Started")
        log.info(f"replan dt:  {self.replan_dt}")
        
        # Initialize path and velocity
        path, vel = None, None
        next_plan_time = time.time()
        
        while not self.__kill_flag.value and self.call_replan:
            try:
                current_time = time.time()
                # Wait until it's time for the next planning cycle
                if current_time < next_plan_time:
                    time.sleep(next_plan_time - current_time)
                    current_time = time.time()
                next_plan_time = current_time + self.replan_dt
                
                t1 = time.time()
                # with self.lock_planner:
                dt = t1 - self.__planner_last_step_time.value
                self.__planner_elapsed_time.value += time.time() - self.__planner_start_time.value
                
                # Execute planning cycle
                self.shared_planner.set_replan_dt(dt)
                self.__planner_last_step_time.value = time.time()
                self.shared_planner.replan()
                path, vel = self.shared_planner.get_serializable_local_plan()
                
                # Update controller if enough time has passed
                if dt > self.sim_dt:
                    # with self.lock_contoller:
                    self.shared_controller.update_serializable_trajectory(path, vel)

                    # with self.lock_world:
                    state = self.shared_world.get_ego_state()
                    self.shared_planner.step(state)

                t2 = time.time()
                log.info(f"Planner iteration: dt={dt:.3f}s, execution time={t2-t1:.3f}s")
            except Exception as e:
                log.error(f"Error in planner worker: {e}", exc_info=True)
                time.sleep(0.1)  # Avoid tight loop if persistent errors
                next_plan_time = time.time() + self.replan_dt  # Reset timer after error


    def worker_controller(self, *args):
        log.info(f"Controller Worker Started")
        while not self.__kill_flag.value and self.call_control:
            try:
                t1 = time.time()
                # with self.lock_worl:
                dt = t1 - self.__controller_last_step_time.value
                if dt > 10 * self.control_dt: # If controller is sleeping for too long, reset the time
                    self.__controller_last_step_time.value = t1
                elif dt > self.control_dt:
                    self.shared_controller.set_control_dt(dt)
                    self.__controller_last_step_time.value = t1
                    state = self.shared_world.get_ego_state()
                    cmd = self.shared_controller.control(state)
                    self.shared_world.update_ego_state(cmd, dt=self.sim_dt)
                t2 = time.time()
                sleep_time = max(0, self.control_dt - (t2 - t1))
                time.sleep(sleep_time)
                log.info(f"Controller iteration actual step time {t2-t1:.3f}  - > sleep time: {sleep_time:.2f} s")
            except Exception as e:
                log.error(f"Error in controller worker: {e}", exc_info=True)
                time.sleep(0.1)
                


    def worker_perceiver(self, *args):
        raise NotImplementedError

    def stop(self):

        # stop the queue listener to prevent logging issues
        if hasattr(self, "__queue_listener") and self.__queue_listener:
            self.__queue_listener.stop()

        count = 0
        for p in self.processes:
            if p and p.is_alive():
                log.info(f"Terminating process {p.name}")
                count += 1
                self.__kill_flag.value = True
                # p.terminate()
                p.join(timeout=1.0)  # Add timeout to prevent indefinite blocking
                if p.is_alive():
                    log.warning(f"Process {p.name} did not terminate cleanly, forcing exit")
                    p.kill()  # Force kill if terminate doesn't work
        
        # time.sleep(0.2) # Wait for processes to term
        log.info(f"Async Executer Processes Stopped. {count}/{len(self.processes)} processes terminated.")
        self.processes = []
        self.planner_process = None
        self.controller_process = None
        self.processes_started = False

    def create_processes(self):
        self.processes = []
        self.planner_process = None
        self.controller_process = None

        self.planner_process = mp.Process(
            target=self.worker_planner,
            args=(
                "Planner",
            ),
        )
        # p = mp.Process(target=self._plan_worker)
        self.processes.append(self.planner_process)

        self.controller_process = mp.Process(
            target=self.worker_controller,
            args=(
                "Controller",
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
        self.__kill_flag.value = False

        t1 = time.time()
        self.setup_process_logging()
        log.info(f"Starting Planner...")
        self.__planner_start_time = Value("d", time.time())  # Shared double variable
        if self.planner_process:
            self.planner_process.start()

        log.info(f"Starting Controller...")
        if self.controller_process:
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

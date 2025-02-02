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

class TrajectoryManager(BaseManager):
    pass


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
                    exposed=("replan", "step", "get_serializable_local_plan", "get_location_xy", "get_location_sd"))
        BaseManager.register('BaseController', 
                    callable=lambda: cn.get_copy(),
                    exposed=('update_serializable_trajectory', 'control', 'dummy', 'reset'))
        BaseManager.register('WorldInterface', 
                    callable=lambda: world.get_copy(),
                    exposed=('get_ego_state', 'update_ego_state'))
        # BaseManager.register('Trajectory', 
        #             callable=lambda: self.planner.get_local_plan(),
        #             exposed=('update_trajectory', 'control', 'dummy', 'reset'))
        


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
        self.setup_worker_logging()

        self.call_replan = call_replan
        self.call_control = call_control
        self.call_perceive = call_perceive

        self.processes = []
        self.processes_started = False

        self.create_processes()

    def step(self, control_dt=0.01, replan_dt=0.01, call_replan=True, call_control=True, call_perceive=False):
        if self.processes_started == False:
            self.start_processes()
            return
        elif all(not p.is_alive() for p in self.processes) and self.processes_started == True:
            log.warning(f"All processes finished. Recreating and starting new processes in 1 sec.")
            time.sleep(1)
            self.create_processes()
            self.start_processes()
            return
        elif any(p.is_alive() for p in self.processes) and not all(p.is_alive() for p in self.processes):
            # UI coordination tasks
            alive_processes = [p for p in self.processes if p.is_alive()]
            log.error(
                f"Some Async Executer Processes are dead! {len(alive_processes)}/{len(self.processes)} processes are still alive."
            )
            return


        with self.lock_world:
            self.ego_state = self.shared_world.get_ego_state()


        with self.lock_planner:
            # self.planner. = self.shared_planner.get_location()[0]
            self.planner.location_xy = self.shared_planner.get_location_xy()
            self.planner.location_sd = self.shared_planner.get_location_sd()

        # self.planner.location_xy = (self.ego_state.x, self.ego_state.y)
        # self.shared_planner.replan()
        # path,vel = self.shared_planner.get_serializable_local_plan()
        # self.shared_controller.update_serializable_trajectory(path, vel)



    def worker_planner(self, replan_dt,*args):
        log.info(f"Plan Worker Started")

        while True:
            time.sleep(0.05)
            with self.lock_planner:
                self.__planner_elapsed_time.value += time.time() - self.__planner_start_time.value
                dt = time.time() - self.__planner_last_replan_time.value
                # log.info(f"Planner iteration: {dt:.2f} s")
                if dt > self.replan_dt:
                    # log.info(f"Replanning... at {dt:.2f} s")
                    self.__planner_last_replan_time.value = time.time()
                    self.shared_planner.replan()
                    path,vel = self.shared_planner.get_serializable_local_plan()
                    with self.lock_controller:
                        self.shared_controller.update_serializable_trajectory(path, vel)

                with self.lock_world:
                    state = self.shared_world.get_ego_state()
                    self.shared_planner.step(state)


    def worker_controller(self, control_dt, *args):
        log.info(f"Controller Worker Started")
        while True:
            time.sleep(0.01)
            # with self.lock_world:
            state = self.shared_world.get_ego_state()
            cmd = self.shared_controller.control(state)
            self.shared_world.update_ego_state(state, cmd, dt=0.01)



    def worker_perceiver(self, *args):
        raise NotImplementedError

    def worker_dummy(self, name):
        # Simulate some work done in a separate process
        # self.setup_worker_logging()
        for i in range(10000):
            msg = f"{name} - iteration {i}"
            print(msg)
            log.info(msg)

            # self.queue.put(msg)
            time.sleep(0.2)
        msg2 = f"{name} - finished"
        print(msg2)
        log.info(msg2)
        # self.queue.put(msg2)

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

    def create_processes(self):
        self.processes = []
        # for i in range(2):
        # p = mp.Process(target=self._dummy_worker, args=(f"Process {i}",))
        # self.processes.append(p)

        if self.call_replan:
            p = mp.Process(
                target=self.worker_planner,
                args=(
                    "Planner",
                    self.replan_dt,
                    self.shared_world,
                    self.shared_planner,
                    self.shared_controller,
                    self.__planner_elapsed_time,
                    self.__planner_start_time,
                    self.__planner_last_replan_time
                ),
            )
            # p = mp.Process(target=self._plan_worker)
            self.processes.append(p)

        if self.call_control:
            p = mp.Process(target=self.worker_controller,
                args=(
                    "Controller",
                    self.control_dt,
                    self.__controller_last_step_time,
                    self.shared_world,
                ),
            )
            self.processes.append(p)
        #
        # if self.call_perceive:
        #     p = mp.Process(target=self.worker_perceiver)
        #     self.processes.append(p)

    # TODO: fix this 
    def start_processes(self):
        self.setup_worker_logging()
        log.info(f"Starting processes...")
        for p in self.processes:
            self.__planner_start_time = Value("d", time.time())  # Shared double variable
            p.start()
        
        # self.__planner_time_since_last_replan = Value("d", 0.0)  # Shared double variable
        self.processes_started = True

    def setup_worker_logging(self):
        """Configure worker process to send logs to queue"""
        # Remove all handlers
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)

        # Add queue handler
        queue_handler = QueueHandler(self.__log_queue)
        root.addHandler(queue_handler)
        # root.setLevel(logging.INFO)

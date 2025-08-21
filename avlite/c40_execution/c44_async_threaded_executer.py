from __future__ import annotations

from avlite.c10_perception.c12_perception_strategy import PerceptionModel
from avlite.c10_perception.c12_perception_strategy import PerceptionStrategy
from avlite.c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from avlite.c20_planning.c23_local_planning_strategy import LocalPlannerStrategy
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c40_execution.c41_execution_model import Executer
from avlite.c40_execution.c43_sync_executer import SyncExecuter, WorldBridge

from logging.handlers import QueueHandler, QueueListener
from queue import Queue

import threading
import time
import logging

log = logging.getLogger(__name__)

# TODO: Perception to be moved to a separate thread
class AsyncThreadedExecuter(Executer):
    def __init__(
        self,
        perception_model: PerceptionModel,
        perception: PerceptionStrategy,
        global_planner: GlobalPlannerStrategy,
        local_planner: LocalPlannerStrategy,
        controller: ControlStrategy,
        world: WorldBridge,
        replan_dt=0.5,
        control_dt=0.05,
    ):
        super().__init__(perception_model, perception, global_planner, local_planner, controller, world, replan_dt=replan_dt, control_dt=control_dt)

        # Thread-specific attributes - no need for shared Values
        self.__planner_last_step_time = time.time()
        self.__planner_elapsed_time = 0.0
        self.__planner_start_time = time.time()
        self.__controller_last_step_time = 0.0
        self.__kill_flag = False

        # Locks for thread safety
        self.lock_planner = threading.Lock()
        self.lock_controller = threading.Lock()
        self.lock_world = threading.Lock()

        self.call_replan = True
        self.call_control = True
        self.call_perceive = True

        self.__log_queue = Queue()
        self.__queue_listener = QueueListener(self.__log_queue, logging.getLogger().handlers[0])
        self.__queue_listener.start()
        self.setup_process_logging()

        self.threads = []
        self.threads_started = False

        self.planner_thread = None
        self.controller_thread = None
        self.perception_thread = None

        self.create_threads()

    def step( self, control_dt=0.01, replan_dt=0.01, sim_dt=0.01, call_replan=True, call_control=True, call_perceive=False):
        self.control_dt = control_dt
        self.replan_dt = replan_dt
        self.sim_dt = sim_dt
        self.call_replan = call_replan
        self.call_control = call_control
        self.call_perceive = call_perceive

        if not self.threads_started:
            log.info(f"Threads not started yet. Creating and starting threads.")
            self.create_threads()
            self.start_threads()
            return
        elif self.threads_started and all(not t.is_alive() for t in self.threads):
            log.warning(f"All threads are dead. Recreating and starting threads.")
            self.stop()
            self.create_threads()
            self.start_threads()
            return
        elif (
            self.threads_started
            and (
                (self.planner_thread and call_replan != self.planner_thread.is_alive())
                or (self.controller_thread and call_control != self.controller_thread.is_alive())
            )
        ):  # or call_perceive != (self.perception_thread.is_alive() if self.perception_thread else False):

            log.error( f"Some threads are dead: {self.planner_thread.is_alive() if self.planner_thread else 'None'}, Controller status: {self.controller_thread.is_alive() if self.controller_thread else 'None'} . Call stop() to terminate all threads.")
            self.create_threads()
            self.start_threads()
            return

        # delta_t_exec = time.time() - self.__prev_exec_time if self.__prev_exec_time is not None else 0
        # self.__prev_exec_time = time.time()
        # self.elapsed_real_time += delta_t_exec

    def worker_planning(self):
        log.info(f"Plan Worker Started")
        log.info(f"replan dt: {self.replan_dt}")

        while not self.__kill_flag and self.call_replan:
            try:
                t1 = time.time()
                # with self.lock_planner:
                dt = t1 - self.__planner_last_step_time
                self.__planner_elapsed_time += time.time() - self.__planner_start_time

                if dt > 10 * self.replan_dt:
                    self.__planner_last_step_time = t1

                elif dt > self.replan_dt:
                    self.__planner_last_step_time = time.time()
                    self.local_planner.replan()
                    self.planner_fps = 1.0 / dt

                # with self.lock_controller:
                self.controller.tj = self.local_planner.get_local_plan()

                # with self.lock_world:
                state = self.world.get_ego_state()
                self.local_planner.step(state)

                t2 = time.time()
                log.debug(f"Planner iteration: dt={dt:.3f}s, execution time={t2-t1:.3f}s")
            except Exception as e:
                log.error(f"Error in planner worker: {e}", exc_info=True)
                time.sleep(0.1)

    def worker_control(self):
        log.info(f"Controller Worker Started")
        while not self.__kill_flag and self.call_control:
            try:
                t1 = time.time()
                dt = t1 - self.__controller_last_step_time

                if dt > 10 * self.control_dt:  # probably its the first iteration
                    self.__controller_last_step_time = t1

                elif dt > self.control_dt:
                    with self.lock_controller:
                        self.__controller_last_step_time = t1

                    with self.lock_world:
                        state = self.world.ego_state
                        cmd = self.controller.control(state, control_dt=self.sim_dt)
                        self.world.control_ego_state(cmd, dt=self.sim_dt)

                    self.control_fps = 1.0 / dt

                t2 = time.time()
                sleep_time = max(0, self.control_dt - (t2 - t1))
                time.sleep(sleep_time)
                log.debug(f"Controller iteration actual step time {t2-t1:.3f} -> sleep time: {sleep_time:.2f} s")
            except Exception as e:
                log.error(f"Error in controller worker: {e}", exc_info=True)
                time.sleep(0.1)

            if self.call_perceive:
                try:
                    self._perception_step()
                except Exception as e:
                    log.error(f"Error in perception step: {e}")

    def _perception_step(self):
        if not self.perception:
            raise RuntimeError("Perception strategy is not set. Skipping perception step.")

        # elif self.perception.supports_detection == False and self.world.supports_ground_truth_detection:
        elif self.perception.requirements.issubset(self.world.capabilities): 
            self.pm = self.world.get_ground_truth_perception_model()
            # perception_output = self.perception.perceive(perception_model=self.pm)
            perception_output = self.perception.perceive(perception_model=self.pm, rgb_img=self.world.get_rgb_image(),
                                                         depth_img=self.world.get_depth_image(),
                                                         lidar_data=self.world.get_lidar_data())

            # log.debug(f"[Executer] Perception output: {perception_output.shape if not isinstance(perception_output, list) else len(perception_output)}")
            log.debug(f"type of perception_output: {type(perception_output)}")
            # log.warning(f"occupancy grid: {self.pm.occupancy_flow}")
            log.debug(f"occupancy grid sizes: {self.pm.grid_bounds}")

        else:
            raise RuntimeError(f"Perception strategy {self.perception.__class__.__name__} requirements {self.perception.requirements} not satisfied by capabilities: {self.world.capabilities}. Skipping perception step.")

    def worker_perception(self):
        while not self.__kill_flag and self.call_perceive:
            try:
                t1 = time.time()
                if self.perception and self.call_perceive:
                    self._perception_step()
                t2 = time.time()
                log.debug(f"Perception iteration: dt={t2-t1:.3f}s")
            except Exception as e:
                log.error(f"Error in perception worker: {e}")
                time.sleep(0.1)

    def stop(self):
        count = 0
        for t in self.threads:
            if t and t.is_alive():
                log.info(f"Stopping thread {t.name}")
                count += 1
                self.__kill_flag = True

        # Wait for threads to terminate
        for t in self.threads:
            if t and t.is_alive():
                t.join(timeout=1.0)
                if t.is_alive():
                    log.warning(f"Thread {t.name} is still running after stop request")

        log.info(f"Async Executer Threads Stopped. {count}/{len(self.threads)} threads signaled to stop.")
        self.threads = []
        self.planner_thread = None
        self.controller_thread = None
        self.perception_thread = None
        self.threads_started = False

    def create_threads(self):
        log.info(f"Creating threads...")
        # Make threads daemon so they exit when main thread exits
        self.threads = []

        if self.planner_thread is None or self.planner_thread.is_alive():
            self.planner_thread = threading.Thread( target=self.worker_planning, name="Planner", daemon=True,  )
            self.threads.append(self.planner_thread)
            log.info(f"Planner thread created: {self.planner_thread.name}")

        if self.controller_thread is None or not self.controller_thread.is_alive():
            self.controller_thread = threading.Thread(target=self.worker_control, name="Controller", daemon=True)
            self.threads.append(self.controller_thread)
            log.info(f"Controller thread created: {self.controller_thread.name}")   

        # self.perception_thread = threading.Thread(target=self.worker_perception, name="Perception", daemon=True)
        # self.threads = [self.planner_thread, self.controller_thread, self.perception_thread]
        log.info(f"{len(self.threads)} threads created.")


    def start_threads(self):
        if self.threads_started:
            log.warning("Threads already started. Call stop() to restart.")
            return
        if len(self.threads) == 0:
            log.warning("No threads created to start. Call create_threads() first.")
            return

        self.__kill_flag = False

        t1 = time.time()
        log.info(f"Starting Planner Thread...")
        self.__planner_start_time = time.time()
        if self.planner_thread:
            self.planner_thread.start()

        log.info(f"Starting Controller Thread...")
        if self.controller_thread:
            self.controller_thread.start()

        # if self.perception_thread:
        #     log.info(f"Starting Perception Thread...")
        #     self.perception_thread.start()

        self.threads_started = True
        log.info(f"Threads started in {time.time()-t1:.3f} s")

    def setup_process_logging(self):
        """Configure worker process to send logs to queue"""
        # Remove all handlers
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)

        # Add queue handler
        queue_handler = QueueHandler(self.__log_queue)
        root.addHandler(queue_handler)

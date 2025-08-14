from __future__ import annotations

from c10_perception.c12_perception_strategy import PerceptionModel
from c10_perception.c12_perception_strategy import PerceptionStrategy
from c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from c20_planning.c23_local_planning_strategy import LocalPlannerStrategy
from c30_control.c32_control_strategy import ControlStrategy
from c40_execution.c41_execution_model import Executer
from c40_execution.c42_sync_executer import SyncExecuter, WorldBridge

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
            self.create_threads()
            self.start_threads()
            return
        elif all(not t.is_alive() for t in self.threads) and self.threads_started:
            log.warning(f"All threads are dead. Recreating and starting threads.")
            self.stop()
            self.create_threads()
            self.start_threads()
            return
        elif any(t.is_alive() for t in self.threads) and not all(t.is_alive() for t in self.threads):
            log.error( f"Some Async Executer Threads are dead! Planner status: {self.planner_thread.is_alive() if self.planner_thread else 'None'}, Controller status: {self.controller_thread.is_alive() if self.controller_thread else 'None'} . Call stop() to terminate all threads.")
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
                if not self.perception:
                    log.error("Perception strategy is not set. Skipping perception step.")
                    continue
                # log.info(f"support detection: {self.perception.supports_detection}")
                if self.perception.supports_detection == False and self.world.supports_ground_truth_detection:
                    self.pm = self.world.get_ground_truth_perception_model()
                    perception_output = self.perception.perceive(perception_model=self.pm)
                    # log.debug(f"[Executer] Perception output: {perception_output} sum{sum(perception_output)}")
                else:
                    perception_output = self.perception.perceive( rgb_img=self.world.get_rgb_image() 
                        if self.world.supports_rgb_image else None, depth_img=self.world.get_depth_image() 
                        if self.world.supports_depth_image else None, lidar_data=self.world.get_lidar_data() 
                        if self.world.supports_lidar_data else None)
                # log.debug(f"Support detection: {self.perception.supports_detection}")
                # log.debug(f"Perception output: {perception_output}")

    def worker_perception(self):
        while not self.__kill_flag and self.call_perceive:
            try:
                # if self.world.support_ground_truth_perception:
                #     self.pm = self.world.get_ground_truth_perception_model()

                # log.info(f"support detection: {self.perception.supports_detection}")
                if self.perception.supports_detection == False and self.world.supports_ground_truth_detection:
                    self.pm = self.world.get_ground_truth_perception_model()
                    perception_output = self.perception.perceive(perception_model=self.pm)
                    # log.debug(f"[Executer] Perception output: {perception_output} sum{sum(perception_output)}")
                else:
                    perception_output = self.perception.perceive( rgb_img=self.world.get_rgb_image() 
                        if self.world.supports_rgb_image else None, depth_img=self.world.get_depth_image() 
                        if self.world.supports_depth_image else None, lidar_data=self.world.get_lidar_data() 
                        if self.world.supports_lidar_data else None)
                # log.debug(f"Support detection: {self.perception.supports_detection}")
                # log.debug(f"Perception output: {perception_output}")
            except Exception as e:
                log.error(f"Error in perception worker: {e}", exc_info=True)
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
        log.info(f"Creating threads for Async Executer")
        self.planner_thread = None
        self.controller_thread = None
        self.perception_thread = None
        # Make threads daemon so they exit when main thread exits
        self.planner_thread = threading.Thread( target=self.worker_planning, name="Planner", daemon=True,  )
        self.controller_thread = threading.Thread(target=self.worker_control, name="Controller", daemon=True)
        # self.perception_thread = threading.Thread(target=self.worker_perception, name="Perception", daemon=True)
        # self.threads = [self.planner_thread, self.controller_thread, self.perception_thread]

        self.threads = [self.planner_thread, self.controller_thread]

        log.info(f"Created {len(self.threads)} threads: {[t.name for t in self.threads]}")

    def start_threads(self):
        if self.threads_started:
            log.warning("Threads already started. Call stop() to restart.")
            return
        if len(self.threads) < 2:
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

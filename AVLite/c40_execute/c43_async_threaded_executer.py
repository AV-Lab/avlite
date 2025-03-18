from __future__ import annotations
from c10_perceive.c11_base_perception import PerceptionModel
from c20_plan.c24_base_local_planner import BaseLocalPlanner
from c30_control.c31_base_controller import BaseController
from c40_execute.c41_base_executer import BaseExecuter, WorldInterface

from logging.handlers import QueueHandler, QueueListener
from queue import Queue

import threading
import time
import logging

log = logging.getLogger(__name__)


class AsyncThreadedExecuter(BaseExecuter):
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

        self.call_replan = call_replan
        self.call_control = call_control
        self.call_perceive = call_perceive

        self.__log_queue = Queue()
        self.__queue_listener = QueueListener(self.__log_queue, logging.getLogger().handlers[0])
        self.__queue_listener.start()
        self.setup_process_logging()

        self.threads = []
        self.threads_started = False

        self.planner_thread = None
        self.controller_thread = None

        self.create_threads()

    def step(
        self, control_dt=0.01, replan_dt=0.01, sim_dt=0.01, call_replan=True, call_control=True, call_perceive=False
    ):
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
            log.error(
                f"Some Async Executer Threads are dead! Planner status: {self.planner_thread.is_alive() if self.planner_thread else 'None'}, Controller status: {self.controller_thread.is_alive() if self.controller_thread else 'None'} . Call stop() to terminate all threads."
            )
            return




        
        # delta_t_exec = time.time() - self.__prev_exec_time if self.__prev_exec_time is not None else 0
        # self.__prev_exec_time = time.time()
        # self.elapsed_real_time += delta_t_exec

    def worker_planner(self):
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
                    self.planner.replan()
                    self.planner_fps = 1.0 / dt

                # with self.lock_controller:
                self.controller.tj = self.planner.get_local_plan()

                # with self.lock_world:
                state = self.world.get_ego_state()
                self.planner.step(state)

                t2 = time.time()
                log.debug(f"Planner iteration: dt={dt:.3f}s, execution time={t2-t1:.3f}s")
            except Exception as e:
                log.error(f"Error in planner worker: {e}", exc_info=True)
                time.sleep(0.1)

    def worker_controller(self):
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
                        state = self.world.get_ego_state()
                        cmd = self.controller.control(state)
                        self.world.update_ego_state(state, cmd, dt=self.sim_dt)
                    self.control_fps = 1.0 / dt

                t2 = time.time()
                sleep_time = max(0, self.control_dt - (t2 - t1))
                time.sleep(sleep_time)
                log.debug(f"Controller iteration actual step time {t2-t1:.3f} -> sleep time: {sleep_time:.2f} s")
            except Exception as e:
                log.error(f"Error in controller worker: {e}", exc_info=True)
                time.sleep(0.1)

    def worker_perceiver(self):
        raise NotImplementedError

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
        self.threads_started = False

    def create_threads(self):
        self.threads = []
        self.planner_thread = None
        self.controller_thread = None

        self.planner_thread = threading.Thread(
            target=self.worker_planner,
            name="Planner",
            daemon=True,  # Make threads daemon so they exit when main thread exits
        )
        self.threads.append(self.planner_thread)

        self.controller_thread = threading.Thread(target=self.worker_controller, name="Controller", daemon=True)
        self.threads.append(self.controller_thread)

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
        # root.setLevel(logging.INFO)

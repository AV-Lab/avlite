from __future__ import annotations

from avlite.c10_perception.c12_perception_strategy import PerceptionModel
from avlite.c10_perception.c12_perception_strategy import PerceptionStrategy
from avlite.c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from avlite.c20_planning.c23_local_planning_strategy import LocalPlanningStrategy
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c40_execution.c41_world_bridge import WorldBridge
from avlite.c40_execution.c42_execution_strategy import ExecutionStrategy
from avlite.c40_execution.c43_task_strategy import TaskStrategy

import threading
import time
import logging

log = logging.getLogger(__name__)

# TODO: Perception to be moved to a separate thread
class AsyncThreadedExecuter(ExecutionStrategy):
    def __init__(
        self,
        perception_model: PerceptionModel,
        perception: PerceptionStrategy = None,
        global_planner: GlobalPlannerStrategy = None,
        local_planner: LocalPlanningStrategy = None,
        controller: ControlStrategy = None,
        world: WorldBridge = None,
        localization=None,
        mapping=None,
        perception_dt=0.5,
        replan_dt=0.5,
        control_dt=0.05,
        localization_dt=0.1,
        combined_perception_planning: bool = True,
        tasks: list[TaskStrategy] | None = None,
    ):
        super().__init__(perception_model, perception, global_planner, local_planner, controller, world,
                         localization=localization, mapping=mapping, perception_dt=perception_dt,
                         replan_dt=replan_dt, control_dt=control_dt, localization_dt=localization_dt,
                         tasks=tasks)

        # When True, perception runs inside the planner thread (lower overhead).
        # When False, perception gets its own dedicated thread.
        self._combined_perception_planning = combined_perception_planning

        # Thread-specific attributes - no need for shared Values
        self.__planner_last_step_time = time.time()
        self.__planner_elapsed_time = 0.0
        self.__planner_start_time = time.time()
        self.__controller_last_step_time = 0.0

        # Locks for thread safety
        self.lock_planner = threading.Lock()
        self.lock_controller = threading.Lock()
        self.lock_world = threading.Lock()

        self.call_replan = True
        self.call_control = True
        self.call_perceive = True
        self.call_localize = True

        self.threads = []
        self.threads_started = False

        self.planner_thread = None
        self.controller_thread = None
        self.perception_thread = None

        self.create_threads()

    def step( self, perception_dt=0.01, control_dt=0.01, replan_dt=0.01, localization_dt=0.01, sim_dt=0.01, call_replan=True, call_control=True, call_perceive=False, call_localize=True):
        self.perception_dt = perception_dt
        self.control_dt = control_dt
        self.replan_dt = replan_dt
        self.localization_dt = localization_dt
        self.sim_dt = sim_dt
        self.call_replan = call_replan
        self.call_control = call_control
        self.call_perceive = call_perceive
        self.call_localize = call_localize

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
        __localize_last_t = time.time()
        __planner_step_last_t = time.time()

        while not self.stopped and self.call_replan:
            try:
                t1 = time.time()
                dt = t1 - self.__planner_last_step_time
                self.__planner_elapsed_time += time.time() - self.__planner_start_time

                if dt > 10 * self.replan_dt:
                    self.__planner_last_step_time = t1
                elif dt > self.replan_dt:
                    self.__planner_last_step_time = time.time()
                    self._replan_step()

                if self.local_planner and self.controller:
                    self.controller.set_plan(self.local_planner.get_local_plan())

                # Rate-limit local_planner.step to replan_dt — avoids flooding the GIL
                # with continuous KD-tree queries that starve the controller thread
                if self.local_planner and t1 - __planner_step_last_t >= self.replan_dt:
                    state = self.world.get_ego_state()
                    self.local_planner.step(state)
                    __planner_step_last_t = t1

                t2 = time.time()
                log.debug("Planner iteration: dt=%.3fs, execution time=%.3fs", dt, t2 - t1)

                # Localization: rate-limited by localization_dt
                if self.call_localize:
                    if t1 - __localize_last_t >= self.localization_dt:
                        try:
                            self._localization_step()
                            __localize_last_t = t1
                        except Exception as e:
                            log.error(f"Error in localization step: {e}", exc_info=True)

                # Perception runs alongside planning, rate-limited by perception_dt.
                # Only active when combined mode is on; in separate-thread mode the
                # dedicated worker_perception thread handles this instead.
                if self.call_perceive and self._combined_perception_planning:
                    dt_p = time.time() - self._perception_fps_tracker.last
                    if dt_p > 10 * self.perception_dt:
                        self._perception_fps_tracker.last = time.time()
                    elif dt_p >= self.perception_dt:
                        try:
                            self._perception_step()
                        except Exception as e:
                            log.error(f"Error in perception step: {e}", exc_info=True)

                # Sleep for the remainder of the replan cycle so this thread
                # does not busy-wait between replans and starve the UI + controller.
                sleep_time = max(0, self.replan_dt - (time.time() - t1))
                time.sleep(sleep_time)

            except Exception as e:
                log.error(f"Error in planner worker: {e}", exc_info=True)
                time.sleep(0.1)

    def worker_control(self):
        log.info(f"Controller Worker Started")
        while not self.stopped and self.call_control:
            try:
                t1 = time.time()
                dt = t1 - self.__controller_last_step_time

                if dt > 10 * self.control_dt:  # probably its the first iteration
                    self.__controller_last_step_time = t1

                elif dt > self.control_dt:
                    with self.lock_controller:
                        self.__controller_last_step_time = t1

                    with self.lock_world:
                        if self.controller and self._can_actuate():
                            sensors = self.world.get_sensor_frame()
                            local_plan = (
                                self.local_planner.get_local_plan()
                                if self.local_planner else None
                            )
                            cmd = self.controller.control(
                                self.world.ego_state, local_plan,
                                control_dt=self.sim_dt,
                                perception_model=self.pm, sensors=sensors,
                            )
                            self.world.control_ego_state(cmd, dt=self.sim_dt)

                    self.control_fps = self._control_fps_tracker.tick(floor_dt=self.sim_dt)
                    self.elapsed_sim_time += self.control_dt
                    self.elapsed_real_time += dt
                    # EVERY_CYCLE means every control cycle under async (not UI poll).
                    self.task_runner.step(self)

                t2 = time.time()
                sleep_time = max(0, self.control_dt - (t2 - t1))
                time.sleep(sleep_time)
                log.debug("Controller iteration actual step time %.3f -> sleep time: %.2f s", t2 - t1, sleep_time)
            except Exception as e:
                log.error(f"Error in controller worker: {e}", exc_info=True)
                time.sleep(0.1)

    def worker_perception(self):
        while not self.stopped and self.call_perceive:
            try:
                t1 = time.time()
                if self.perception and self.call_perceive:
                    self._perception_step()
                t2 = time.time()
                log.debug("Perception iteration: dt=%.3fs", t2 - t1)
            except Exception as e:
                log.error(f"Error in perception worker: {e}")
                time.sleep(0.1)

    @property
    def ui_poll_delay(self):
        # step() is nearly instant — all work runs in background threads.
        # Tell the UI to poll at 20 Hz rather than burning the event loop.
        return 0.05

    def stop(self):
        # Safe to call from a worker: set stopped, join peers, never join self.
        super().stop()
        current = threading.current_thread()
        threads = list(self.threads)
        count = sum(1 for t in threads if t and t.is_alive())
        for t in threads:
            if t and t.is_alive():
                log.info(f"Stopping thread {t.name}")

        try:
            for t in threads:
                if t and t.is_alive() and t is not current:
                    t.join(timeout=1.0)
                    if t.is_alive():
                        log.warning(f"Thread {t.name} is still running after stop request")
        finally:
            log.info(
                f"Async Executer Threads Stopped. {count}/{len(threads)} threads signaled to stop."
            )
            self.threads = []
            self.planner_thread = None
            self.controller_thread = None
            self.perception_thread = None
            self.threads_started = False

    def create_threads(self):
        log.info(f"Creating threads...")
        # Make threads daemon so they exit when main thread exits
        self.threads = []

        if self.planner_thread is None or not self.planner_thread.is_alive():
            self.planner_thread = threading.Thread( target=self.worker_planning, name="Planner", daemon=True,  )
            self.threads.append(self.planner_thread)
            log.info(f"Planner thread created: {self.planner_thread.name}")

        if self.controller_thread is None or not self.controller_thread.is_alive():
            self.controller_thread = threading.Thread(target=self.worker_control, name="Controller", daemon=True)
            self.threads.append(self.controller_thread)
            log.info(f"Controller thread created: {self.controller_thread.name}")

        if not self._combined_perception_planning:
            self.perception_thread = threading.Thread(target=self.worker_perception, name="Perception", daemon=True)
            self.threads.append(self.perception_thread)
            log.info(f"Perception thread created: {self.perception_thread.name}")

        log.info(f"{len(self.threads)} threads created.")


    def start_threads(self):
        if self.threads_started:
            log.warning("Threads already started. Call stop() to restart.")
            return
        if len(self.threads) == 0:
            log.warning("No threads created to start. Call create_threads() first.")
            return

        self.stopped = False

        t1 = time.time()
        log.info(f"Starting Planner Thread...")
        self.__planner_start_time = time.time()
        if self.planner_thread:
            self.planner_thread.start()

        log.info(f"Starting Controller Thread...")
        if self.controller_thread:
            self.controller_thread.start()

        if not self._combined_perception_planning and self.perception_thread:
            log.info(f"Starting Perception Thread...")
            self.perception_thread.start()

        self.threads_started = True
        log.info(f"Threads started in {time.time()-t1:.3f} s")

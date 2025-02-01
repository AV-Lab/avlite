from __future__ import annotations
from c10_perceive.c11_perception_model import PerceptionModel
from c20_plan.c21_base_planner import BasePlanner
from c20_plan.c24_trajectory import Trajectory
from c30_control.c31_base_controller import BaseController, ControlComand
from c10_perceive.c12_state import EgoState
from c40_execute.c41_executer import Executer, WorldInterface

import multiprocessing as mp
from multiprocessing.managers import BaseManager
from multiprocessing import Value, Lock
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
    ):
        super().__init__(pm, pl, cn, world)
        # BaseManager.register('Trajectory', Trajectory, exposed=())
        # BaseManager.register('EgoState', EgoState, exposed=())
        # BaseManager.register('ControlComand', ControlComand, exposed=())
        # self.manager = BaseManager()
        # self.manager.start()
        #
        # self.shared_cmd = self.manager.ControlComand()
        # self.shared_ego_state = self.manager.EgoState()
        # self.shared_ego_state = self.manager.ControlComand()
        # self.__time_since_last_replan = Value('d', 0.0)  # Shared double variable
        # self.lock = Lock()
        
        self.log_queue = mp.Queue()
        self.queue_listener = QueueListener(
            self.log_queue,
            logging.getLogger().handlers[0]  # Forward to default handler
        )
        self.queue_listener.start()

        self.processes = []
        self.queue = mp.Queue()
    
    def step(self, control_dt=0.01, replan_dt=None, call_replan=True, call_control=True, call_perceive=False):
        log.info(f"Async Executer Step Called")
        raise NotImplementedError
    
    def setup_worker_logging(self):
        """Configure worker process to send logs to queue"""
        # Remove all handlers
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        
        # Add queue handler
        queue_handler = QueueHandler(self.log_queue)
        root.addHandler(queue_handler)
        # root.setLevel(logging.INFO)

    def worker(self,name):
        # Simulate some work done in a separate process
        self.setup_worker_logging()
        logger = logging.getLogger(__name__)
        for i in range(100):
            msg = f"{name} - iteration {i}"
            print(msg)
            logger.info(msg)

            # self.queue.put(msg)
            time.sleep(.2)
        msg2 = f"{name} - finished"
        print(msg2)
        logger.info(msg2)
        # self.queue.put(msg2)

    def start_processes(self):
        for i in range(10):
            p = mp.Process(target=self.worker, args=(f"Process {i}",))
            self.processes.append(p)
        for p in self.processes:
            p.start()


    def run(self, replan_dt=0.5, control_dt=0.01, call_replan=True, call_control=True, call_perceive=False):
        processes = []
        log.info(f"Async Executer Run Called")
        if all(not p.is_alive() for p in self.processes):
            self.start_processes()

        # while not self.queue.empty():
        #     message = self.queue.get()
        #     log.info(message)
        # log.info(f"Queue is empty")
        # checkign if processes alive or not 
        
        # if call_replan:
        #     p = mp.Process(target=self._planner_process, args=(replan_dt,))
        #     processes.append(p)
        #     p.start()
        #     log.info(f"Planner Process Started")
        #
        # if call_control:
        #     p = mp.Process(target=self._controller_process, args=(control_dt,))
        #     processes.append(p)
        #     p.start()
        #
        # if call_perceive:
        #     p = mp.Process(target=self._perceiver_process)
        #     processes.append(p)
        #     p.start()

    def stop(self):
        log.info(f"Async Executer Stop Called")
        for p in self.processes:
            if p.is_alive():
                p.terminate()  # Force terminate the process
                p.join()      # Wait for process to finish
        log.info(f"Async Executer Processes Stopped")
        self.processes = []

    def _planner_process(self, replan_dt):
        self.__time_since_last_replan += replan_dt
        if self.__time_since_last_replan > replan_dt:
            self.__time_since_last_replan = 0
            self.planner.replan()
        self.planner.step(self.shared_ego_state.value)

    def _controller_process(self, control_dt):
        local_tj = self.planner.get_local_plan()
        cmd = self.controller.control(self.shared_ego_state.value, local_tj)
        self.world.update_ego_state(self.shared_ego_state.value, cmd, dt=control_dt)
        self.shared_cmd.value = cmd

    def _perceiver_process(self):
        raise NotImplementedError

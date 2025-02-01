import multiprocessing as mp
from multiprocessing.managers import BaseManager
import copy

class NestedCounter:
    def __init__(self, initial_value=0):
        self.value = initial_value

    def increment(self):
        self.value += 1

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = value

class Counter:
    def __init__(self, initial_value=0):
        self.nested_counter = NestedCounter(initial_value)

    def increment(self):
        self.nested_counter.increment()

    def get_value(self):
        return self.nested_counter.get_value()

    def set_value(self, value):
        self.nested_counter.set_value(value)
    
    def get_copy(self):
        return self


BaseManager.register('Counter', Counter, exposed=('increment', 'get_value', 'set_value', 'get_copy'))

def increment_counter(id, shared_counter, lock):
    print(f"Process {id} started with counter value {shared_counter.get_value()}")
    
    # counter = shared_counter.get_copy() if id % 2 == 0 else shared_counter
    counter = shared_counter
    for _ in range(100):
        with lock:
            counter.increment()
    print(f"Process {id} end with counter value {counter.get_value()}")

if __name__ == "__main__":
    manager = BaseManager()
    manager.start()
    shared_counter = manager.Counter(0)  # Initialize Counter object
    lock = mp.Lock()  # Use multiprocessing.Lock()

    processes = [mp.Process(target=increment_counter, args=(id, shared_counter, lock)) for id in range(10)]

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print(f"Final counter value: {shared_counter.get_value()}")

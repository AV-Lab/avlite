from planner import planner
import yaml
from pynput import keyboard
import time
import queue

class dummy_planner(planner):
    def __init__(self, path_to_track, frenet_zoom=15, xy_zoom=30):
        super().__init__(path_to_track, frenet_zoom, xy_zoom)
        self.step_idx = 0

    def step(self):
        # x_current = self.reference_x[self.step_idx]
        # y_current = self.reference_y[self.step_idx]

        # self.step_idx = (self.step_idx + 1) % len(self.reference_path)
        # print("=== Step: ", self.step_idx)
        
        # super().step_at_fixed_loc(x_current, y_current)
        super().step()
        



def load():
    config_file = "/home/mkhonji/workspaces/A2RL_Integration/config/config.yaml"
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
        path_to_track = config_data["path_to_track"]

    pl = planner(path_to_track)
    return pl




def animate_from(frame=1100):
    pl = load()
    # pl.step_idx = len(pl.tj.reference_path) - 20
    pl.step_idx = frame


    i = 0 
    while True:
        pl.step()
        if i % 10 == 0:
            pl.replan()
        i += 1
        pl.plot(pause_interval=0.01)

def static_at(frame=1100):
    pl = load()   
    pl.step_idx = frame
    # pl.step_idx = len(pl.tj.reference_path) - 20


    pl.step()
    pl.replan()
    pl.plot(pause_interval=0.01)
    pl.plt_show()


def keyboard_from(frame=1100):
    pl = load()
    key_queue = queue.Queue()

    def _on_press(key):
        try:
            if key == keyboard.Key.space:
                key_queue.put('s')
            elif key == keyboard.KeyCode.from_char('r'):
                key_queue.put('r')
        except AttributeError:
            pass

    listener = keyboard.Listener(on_press=_on_press)
    listener.start()


    print("Press space to step, and R to replan")

    while True:
        try:
            key = key_queue.get(timeout=0.01)
            if key == 's':
                t1 = time.time()
                pl.step()
                t2 = time.time()
                print("Step time (milli-seconds): ", (t2 - t1) * 1000)
            elif key == 'r':
                t1 = time.time()
                pl.replan()
                t2 = time.time()
                print("Replan time (milli-seconds): ", (t2 - t1) * 1000)
            pl.plot(pause_interval=0.01)
        except queue.Empty:
            pass


def main():
    # animate_from(frame=1100) 
    static_at(frame=1100)
    # keyboard_from(frame=1100)

if __name__ == '__main__':
    main()

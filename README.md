# High Level AV Stack
The stack provides a clear logical abstraction of the autonomous vehicle driving components, isolating the developer
from lower-level technical details when not needed. The code also includes tools for hot reloading and debugging, 
making it easier to develop and test the code while running the stack.

## Installation

To install the package from source 
```bash
pip install -r requirements.txt
```
To run the package from source:
```bash
python race_plan_control/main.py
```

To install the package system wide:
```bash
pip install .
```
To run the package system wide:
```bash
race_plan_control
```
## Project structure 
The project is structured as follows:
- Core components are prevised with `ci_` prefix, where `i` is the component number. For examplle `c30_control` is the controllers.
Utility.
- Within each coponent, each module has also number prefix. For example `c32_pid_controller` is the PID controller module `c30_control`.

The goal is to be able to quicly navigate to the desired module by using the search function of the editor. It also provide a quick understanding on where the code belongs to.


## Local Planner
![](docs/imgs/tk_visualizer.png)







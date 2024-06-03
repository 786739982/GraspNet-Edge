# Arm Controll Python Interface

[toc]

## 1. Install

### 1.1 Finish the installation of arm-control C++ interface first

### 1.2 Get and enter the source code folder

```shell
git clone --recursive https://git.qiuzhi.tech:20000/airbot-play/control/python-interface.git
cd python-interface/
```

### 1.3 Build from sources on host & install

##### Compile into shared libraries

```bash
mkdir build && cd build
cmake .. && make -j8
```

##### Install airbot python package

```bash
pip install -e .
```

## 2. Usage

The names of all the python APIs are the same as that of the C++ version. Here is an example of how to use:

```python
import airbot
import time

# modify the path to the airbot_play urdf file
urdf_path = "path/to/your/arm-control/models/airbot_play_v2_1/urdf/airbot_play_v2_1_with_gripper.urdf"
# specify the fk/ik/dk classes to use
fk = airbot.ChainFKSolver(urdf_path)
ik = airbot.ChainIKSolver(urdf_path)
id = airbot.ChainIDSolver(urdf_path, "down")
# instance the airbot player
airbot_player = airbot.create_agent(fk, ik, id, "can0", 1.0, "newteacher", False, False)
# wait for the robot move to the initial zero pose
time.sleep(2)
# get current joint positions(q), velocities(v) and torques(t)
# all are six-elements tuple containing current values of joint1-6
cp = list(airbot_player.get_current_joint_q())
cv = airbot_player.get_current_joint_v()
ct = airbot_player.get_current_joint_t()
# set target joint positions (no blocking)
# all are six-elements tuple/list/np.ndarray containing target values of joint1-6
cp[5] = 1.5
airbot_player.set_target_joint_q(cp)
# airbot_player.set_target_pose([(0, 0, 0), (0, 0, 0, 1)])
# wait for the movement to be done
time.sleep(2)
# enter the gravity compensation mode
airbot_player.gravity_compensation()
time.sleep(5)
# exit the gravity compensation mode by setting joint target (usually current value)
airbot_player.set_target_joint_q(airbot_player.get_current_joint_q())

# wait key
while True:
    key = input("Press Enter to exit...")
    if key == "":
        del airbot_player
        break
```

Before run the example code above, you should first setup the robot (power on and connect it to your computer) and run the cmd below to start the communication of target can bus.
```bash
sudo ip link set up can0 type can bitrate 1000000
./airbot_play_node <urdf_path> <1st_can_device> [<2st_can_device>]
```

Then you can run the example codes. For more instructions on using these APIs, please refer to the README.md file of arm-control C++ interface project.
import airbot
import time

# modify the path to the airbot_play urdf file
urdf_path = (
    "/home/ghz/Work/airbot_play/arm-control/models/airbot_play_v2_1/urdf/airbot_play_v2_1_with_gripper.urdf"
)

# specify the fk/ik/dk classes to use
fk = airbot.AnalyticFKSolver(urdf_path)
ik = airbot.ChainIKSolver(urdf_path)
id = airbot.ChainIDSolver(urdf_path, "down")
# instance the airbot player
airbot_player = airbot.create_agent(fk, ik, id, "vcan0", 1.0, "gripper", False, False)

# wait for the robot move to the initial zero pose
time.sleep(2)
# get current joint positions(q), velocities(v) and torques(t)
# all are six-elements tuple containing current values of joint1-6
cp = airbot_player.get_current_joint_q()
cv = airbot_player.get_current_joint_v()
ct = airbot_player.get_current_joint_t()
print(cp)
# set target joint positions (no blocking)
# all are six-elements tuple/list/np.ndarray containing target values of joint1-6
cp[5] = 1.5
airbot_player.set_target_joint_q(cp)
# airbot_player.set_target_pose([(0, 0, 0), (0, 0, 0, 1)])
# wait for the movement to be done
time.sleep(2)
cp = airbot_player.get_current_joint_q()
print(cp)
cep = airbot_player.set_target_end(0.5)
time.sleep(2)
cep = airbot_player.get_current_end()
print(cep)
# enter the gravity compensation mode
# airbot_player.gravity_compensation()
# time.sleep(5)
# exit the gravity compensation mode by setting joint target (usually current value)
# airbot_player.set_target_joint_q(airbot_player.get_current_joint_q())

# wait key
while True:
    key = input("Press Enter to exit...")
    if key == "":
        # del airbot_player
        break
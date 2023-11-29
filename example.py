import time

import mujoco.viewer
from MujocoRobot import *

if __name__ == "__main__":

    lower_body_mjcf = "robot_descriptions/kyon/kyon_lowerbody_position_ctrl.xml"

    model = mujoco.MjModel.from_xml_path(lower_body_mjcf)
    data = mujoco.MjData(model)

    robot = MujocoRobot(model, data)

    # set control referrence from keyframe
    robot.set_ctrl_from_keyframe()

    print(robot.joint_names)

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        start = time.monotonic()
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

             
        while viewer.is_running():
            t = time.monotonic()
            st = np.sin(t)

            # set control reference relative to initial(homing) control setpoints
            robot.set_relative_ctrl(['hip_pitch_1','knee_pitch_1'],[ st*0.2, -st*0.4])
            robot.set_relative_ctrl(['hip_pitch_2','knee_pitch_2'],[ st*0.2, -st*0.4])
            robot.set_relative_ctrl(['hip_pitch_3','knee_pitch_3'],[-st*0.2, +st*0.4])
            robot.set_relative_ctrl(['hip_pitch_4','knee_pitch_4'],[-st*0.2, +st*0.4])

            # set absolute control reference 
            # robot.set_ctrl(['hip_pitch_1','hip_pitch_3','hip_pitch_2','hip_pitch_4'],[-st*0.15]*4)
            # robot.set_ctrl(['knee_pitch_1','knee_pitch_3','knee_pitch_2','knee_pitch_4'],[st*0.3]*4)

            # step physics
            robot.step()


            # syncronize viewer
            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.monotonic() - t)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
                

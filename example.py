import time

from MujocoRobot import *

if __name__ == "__main__":

    import mujoco.viewer
    import time

    model = mujoco.MjModel.from_xml_path("robot_descriptions/kyon/kyon_position_ctrl.xml")
    data = mujoco.MjData(model)

    robot = MujocoRobot(model, data)
    robot.set_ctrl_from_keyframe()

    print(robot.joint_names)

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        start = time.time()
        while viewer.is_running():
            t = time.time()


            robot.step()

            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - t)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
                

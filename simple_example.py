import time
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("robot_descriptions/kyon/kyon_lowerbody_position_ctrl.xml")
data = mujoco.MjData(model)


with mujoco.viewer.launch_passive(model, data) as viewer:
    wall = time.monotonic()

    mujoco.mj_resetDataKeyframe(model,data,0)
    data.ctrl = data.qpos[7:]

    while viewer.is_running():
        t = time.monotonic() - wall

        while (data.time <= t):
            mujoco.mj_step(model, data)

        viewer.sync()
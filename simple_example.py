import time
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path(
    "robot_descriptions/kyon/kyon_lowerbody_position_ctrl.xml"
)
data = mujoco.MjData(model)


with mujoco.viewer.launch_passive(model, data) as viewer:

    with viewer.lock():
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.ctrl = data.qpos[7:]

    wall = time.monotonic()
    while viewer.is_running():
        t = time.monotonic() - wall

        while data.time <= t:
            mujoco.mj_step(model, data)

        viewer.sync()

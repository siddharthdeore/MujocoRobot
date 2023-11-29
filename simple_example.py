import time
import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path("robot_descriptions/kyon/kyon_position_ctrl.xml")
d = mujoco.MjData(m)


with mujoco.viewer.launch_passive(m, d) as viewer:
    wall = time.monotonic()

    mujoco.mj_resetDataKeyframe(m,d,0)
    d.ctrl = d.qpos[7:]

    while viewer.is_running():
        t = time.monotonic() - wall

        while (d.time <= t):
            mujoco.mj_step(m, d)

        viewer.sync()
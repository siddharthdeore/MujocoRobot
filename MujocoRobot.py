import mujoco
import time
import numpy as np


def extract_mj_names(model, num_obj, obj_type):
    id2name = {i: None for i in range(num_obj)}
    name2id = {}

    for i in range(num_obj):
        name = mujoco.mj_id2name(model, obj_type, i)
        name2id[name] = i
        id2name[i] = name
    
    return [id2name[nid] for nid in sorted(name2id.values())], name2id, id2name

class MujocoRobot:
    def __init__(self, model, data) -> None:

        self.data = data
        self.model = model
        self.joint_names = None
        self.__make_mappings()
        mujoco.mj_resetDataKeyframe(self.model,self.data,0)

        self.revolute_jnt_index = np.where(self.model.jnt_type==mujoco.mjtJoint.mjJNT_HINGE)[0].astype(np.int32)
        self.floating_jnt_index = np.where(self.model.jnt_type==mujoco.mjtJoint.mjJNT_FREE)[0].astype(np.int32)
        self.set_ctrl_from_keyframe()

        self.init_ctrl_ref = {}

        itr = 0
        for i in self.revolute_jnt_index:
            self.init_ctrl_ref[self._joint_id2name[i]] = self.data.ctrl[itr]
            itr = itr + 1



        self.wall = time.monotonic()

    def set_ctrl_from_keyframe(self):
        mujoco.mj_resetDataKeyframe(self.model,self.data,0)
        if(self.model.joint(0).type == mujoco.mjtJoint.mjJNT_FREE):
            self.data.ctrl = self.data.qpos[7:]
        else:
            self.data.ctrl = self.data.qpos
        
    def step(self):
        t = time.monotonic() - self.wall
        while (self.data.time <= t):
            mujoco.mj_step(self.model, self.data, nstep=1)

    def get_body_positon(self, body_name):
        """
        Get body position
        """
        return self.data.body(body_name).xpos

    def get_body_rotation(self, body_name):
        """
        Get body rotation matrix
        """
        return self.data.body(body_name).xmat.reshape([3, 3])
    def set_relative_ctrl(self,name,val):
        if isinstance(name, str):
            index = self._actuator_name2id[name]
            self.data.ctrl[index] = self.init_ctrl_ref[name] + val
        elif isinstance(name, list) and isinstance(val, list):
            for n, v in zip(name, val):
                index = self._actuator_name2id[n]
                self.data.ctrl[index] = self.init_ctrl_ref[n] + v

    def set_ctrl(self,name,val):
        if isinstance(name, str):
            index = self._actuator_name2id[name]
            self.data.ctrl[index] = val
        elif isinstance(name, list) and isinstance(val, list):
            for n, v in zip(name, val):
                index = self._actuator_name2id[n]
                self.data.ctrl[index] = v

    def __make_mappings(self):
        m = self.model
        (
            self.body_names,
            self._body_name2id,
            self._body_id2name,
        ) = extract_mj_names(m, m.nbody, mujoco.mjtObj.mjOBJ_BODY)
        (
            self.joint_names,
            self._joint_name2id,
            self._joint_id2name,
        ) = extract_mj_names(m, m.njnt, mujoco.mjtObj.mjOBJ_JOINT)
        (
            self.actuator_names,
            self._actuator_name2id,
            self._actuator_id2name,
        ) = extract_mj_names(m,  m.nu, mujoco.mjtObj.mjOBJ_ACTUATOR)
        (
            self.sensor_names,
            self._sensor_name2id,
            self._sensor_id2name,
        ) = extract_mj_names(m, m.nsensor, mujoco.mjtObj.mjOBJ_SENSOR)


if __name__ == "__main__":

    import mujoco.viewer
    import time

    model = mujoco.MjModel.from_xml_path("robot_descriptions/kyon/kyon_position_ctrl.xml")
    data = mujoco.MjData(model)

    robot = MujocoRobot(model, data)
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
                

import os
import jax
import jax.numpy as jnp
import numpy as np
import mink
import mujoco
from tqdm import trange

from loco_mujoco.environments import UnitreeG1, AgibotX2
from loco_mujoco.task_factories import RLFactory, ImitationFactory, DefaultDatasetConf
from loco_mujoco.trajectory import Trajectory, TrajectoryInfo, TrajectoryModel, TrajectoryData
from loco_mujoco.core.utils.mujoco import mj_jntname2qposid



# UnitreeG1 to AgibotX2 joint mapping
JOINT_MAPPING = { 
        "root": "floating_base_joint",

        # --- LEFT LEG ---
        "left_hip_pitch_joint": "left_hip_pitch_joint",
        "left_hip_roll_joint": "left_hip_roll_joint",
        "left_hip_yaw_joint": "left_hip_yaw_joint",
        "left_knee_joint": "left_knee_joint",
        "left_ankle_pitch_joint": "left_ankle_pitch_joint",
        "left_ankle_roll_joint": "left_ankle_roll_joint",
        
        # --- RIGHT LEG ---
        "right_hip_pitch_joint": "right_hip_pitch_joint",
        "right_hip_roll_joint": "right_hip_roll_joint",
        "right_hip_yaw_joint": "right_hip_yaw_joint",
        "right_knee_joint": "right_knee_joint",
        "right_ankle_pitch_joint": "right_ankle_pitch_joint",
        "right_ankle_roll_joint": "right_ankle_roll_joint",
        
        # --- WAIST ---
        "waist_yaw_joint": "waist_yaw_joint",
        "waist_pitch_joint": "waist_pitch_joint",
        "waist_roll_joint": "waist_roll_joint",
        
        # --- ARMS (Map these if the G1 dataset has arm motion) ---
        "left_shoulder_pitch_joint": "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint": "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint": "left_shoulder_yaw_joint",
        "left_elbow_joint": "left_elbow_joint",
        "right_shoulder_pitch_joint": "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint": "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint": "right_shoulder_yaw_joint",
        "right_elbow_joint": "right_elbow_joint"
    }

 

def generate_G1_trajectories():
    N_steps = 5000
    env = UnitreeG1(init_state_type="DefaultInitialStateHandler")
    
    # reset the env
    key = jax.random.PRNGKey(0)
    env.reset(key)

    # get the model and data of the environment
    model = env.get_model()
    data = env.get_data()

    # get the initial qpos and qvel of the environment
    qpos = data.qpos
    qvel = data.qvel

    # stack qpos and qvel to a trajectory
    qpos = np.tile(qpos, (N_steps, 1))
    qvel = np.tile(qvel, (N_steps, 1))

    # add a sine wave to the elbow joint to make it more interesting
    elbow_joint_left_ind = mj_jntname2qposid("left_elbow_joint", model)
    test = 0.1 * np.sin(np.linspace(0, 2 * np.pi, N_steps))
    tes3 = qpos[:, elbow_joint_left_ind]
    qpos[:, elbow_joint_left_ind] += 0.5 * np.sin(np.linspace(0, 20 * np.pi, N_steps)).reshape(-1, 1)

    # since the elbow qpos is updated, qvel needs to be updated as well
    qvel = qvel[1:-1]
    qvel[:, elbow_joint_left_ind] = (qpos[2:, elbow_joint_left_ind] - qpos[:-2, elbow_joint_left_ind]) / (2 * env.dt)
    qpos = qpos[1:-1]

    # create a trajectory info -- this stores basic information about the trajectory
    njnt = model.njnt
    jnt_type = model.jnt_type
    jnt_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(njnt)]
    traj_info = TrajectoryInfo(jnt_names, model=TrajectoryModel(njnt, jnp.array(jnt_type)), frequency=1 / env.dt)

    # create a trajectory data -- this stores the actual trajectory data
    traj_data = TrajectoryData(jnp.array(qpos), jnp.array(qvel), split_points=jnp.array([0, N_steps]))

    # combine them to a trajectory
    traj = Trajectory(traj_info, traj_data)

    # example: save the trajectory
    #traj.save("trajectory.npz")

    # example: load the trajectory
    #traj = Trajectory.load("trajectory.npz")

    # add the trajectory to the environment
    env.load_trajectory(traj)

    # replay the trajectory
    env.play_trajectory(n_steps_per_episode=N_steps)
    

# ---- keypoints to track: adjust names to match YOUR xml's site/body names ----
# These should be sites (or bodies) placed at anatomically equivalent spots
# on both the G1 and X2 MJCF (feet, hands/wrists, pelvis, torso).
# KEYPOINTS = {
#     "pelvis":     dict(g1="pelvis",      x2="pelvis",      type="body"),
#     "torso":      dict(g1="torso_link",  x2="torso_link",  type="body"),
#     "left_foot":  dict(g1="left_foot",   x2="left_foot",   type="site"),
#     "right_foot": dict(g1="right_foot",  x2="right_foot",  type="site"),
#     "left_hand":  dict(g1="left_wrist",  x2="left_wrist",  type="site"),
#     "right_hand": dict(g1="right_wrist", x2="right_wrist", type="site"),
# }

KEYPOINTS = {
    #"pelvis":     dict(g1="pelvis",      x2="pelvis",      type="body"),
    #"torso":      dict(g1="torso_link",  x2="torso_link",  type="body"),
    "left_foot":    dict(g1="left_foot_mimic",  x2="left_foot",   type="site"),
    "right_foot":   dict(g1="right_foot_mimic", x2="right_foot",  type="site"),
    "left_wrist":   dict(g1="left_hand_mimic",  x2="left_wrist",  type="site"),
    "right_wrist":  dict(g1="right_hand_mimic", x2="right_wrist", type="site"),
    "torso":        dict(g1="torso_link",       x2="torso_link",  type="body"),
}


# def get_frame_pose(model, data, name, ftype):
#     if ftype == "site":
#         sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
#         pos = data.site_xpos[sid].copy()
#         mat = data.site_xmat[sid].reshape(3, 3).copy()
#     else:
#         bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
#         pos = data.xpos[bid].copy()
#         mat = data.xmat[bid].reshape(3, 3).copy()
#     return pos, mat

def build_qpos_address_map(model: mujoco.MjModel):
    """Map joint name -> slice into qpos for that joint (handles free + hinge joints)."""
    addr_map = {}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        jtype = model.jnt_type[j]
        adr = model.jnt_qposadr[j]
        if jtype == mujoco.mjtJoint.mjJNT_FREE:
            addr_map[name] = slice(adr, adr + 7)   # xyz + quat(wxyz)
        elif jtype == mujoco.mjtJoint.mjJNT_BALL:
            addr_map[name] = slice(adr, adr + 4)   # quat
        else:  # hinge / slide
            addr_map[name] = slice(adr, adr + 1)
    return addr_map


def get_frame_pose(model, data, name, ftype):
    """Safely retrieves frame pose and guards against silent -1 index lookups."""
    if ftype == "site":
        fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if fid == -1:
            raise ValueError(f"Site '{name}' not found in MuJoCo model!")
        pos = data.site_xpos[fid].copy()
        mat = data.site_xmat[fid].reshape(3, 3).copy()
    else:
        fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if fid == -1:
            raise ValueError(f"Body '{name}' not found in MuJoCo model!")
        pos = data.xpos[fid].copy()
        mat = data.xmat[fid].reshape(3, 3).copy()
    return pos, mat


# def compute_limb_scale(g1_model, g1_data, x2_model, x2_data):
#     """
#     Rough per-limb scale factors so keypoint targets are expressed relative to
#     X2's own proportions instead of G1's. Uses standing/keyframe pose distances.
#     Refine this per-limb (leg vs arm) if your robots differ a lot segment-wise.
#     """
#     mujoco.mj_forward(g1_model, g1_data)
#     mujoco.mj_forward(x2_model, x2_data)

#     def leg_length(model, data, foot_site):
#         pelvis, _ = get_frame_pose(model, data, "pelvis", "body")
#         foot, _ = get_frame_pose(model, data, foot_site, "site")
#         return np.linalg.norm(pelvis - foot)

#     def arm_length(model, data, hand_site):
#         pelvis, _ = get_frame_pose(model, data, "pelvis", "body")
#         hand, _ = get_frame_pose(model, data, hand_site, "site")
#         return np.linalg.norm(pelvis - hand)

#     leg_scale = leg_length(x2_model, x2_data, "left_foot") / leg_length(g1_model, g1_data, "left_foot_mimic")
#     arm_scale = arm_length(x2_model, x2_data, "left_wrist") / arm_length(g1_model, g1_data, "left_hand_mimic")
#     return {"left_foot": leg_scale, "right_foot": leg_scale,
#             "left_wrist": arm_scale, "right_wrist": arm_scale
#             }

def compute_limb_scale(g1_model, g1_data, x2_model, x2_data):
    """Computes exact 3D limb proportions between G1 and X2 to prevent pigeon-toeing and knee snapping."""
    mujoco.mj_forward(g1_model, g1_data)
    mujoco.mj_forward(x2_model, x2_data)

    def get_rel_pos(model, data, end_site):
        pelvis, _ = get_frame_pose(model, data, "pelvis", "body")
        end, _ = get_frame_pose(model, data, end_site, "site")
        return np.abs(end - pelvis)  # Absolute distance per axis in default pose

    # --- LEG SCALING (Anatomical Width and Height) ---
    g1_leg_vec = get_rel_pos(g1_model, g1_data, "left_foot_mimic")
    x2_leg_vec = get_rel_pos(x2_model, x2_data, "left_foot")
    
    z_scale = (x2_leg_vec[2] / g1_leg_vec[2])
    y_scale = x2_leg_vec[1] / g1_leg_vec[1]
    
    # We use Z-scale for X (stride length) so stride matches leg length proportions
    leg_scale = np.array([z_scale, y_scale, z_scale])

    # --- ARM SCALING (Overall length) ---
    g1_arm_vec = get_rel_pos(g1_model, g1_data, "left_hand_mimic")
    x2_arm_vec = get_rel_pos(x2_model, x2_data, "left_wrist")
    
    arm_len_scale = np.linalg.norm(x2_arm_vec) / np.linalg.norm(g1_arm_vec)
    arm_scale = np.array([arm_len_scale, arm_len_scale, arm_len_scale])

    return {
        "left_foot": leg_scale, "right_foot": leg_scale,
        "left_wrist": arm_scale, "right_wrist": arm_scale
    }


JOINT_TRANSFORMS = {
    # joint_name: (multiplier, offset)
    "left_elbow_joint":  (-1.0, 0.0),
    "right_elbow_joint": (-1.0, 0.0),
}

def retarget_one_to_one(g1_model, x2_model, g1_qpos_traj, joint_mapping):
    """
    g1_qpos_traj: (T, nq_g1) array of source qpos frames.
    Returns: (T, nq_x2) array for the target robot, direct joint-angle copy
             for every mapped joint, default (keyframe/zero) pose elsewhere.
    """
    g1_addr = build_qpos_address_map(g1_model)
    x2_addr = build_qpos_address_map(x2_model)

    T = g1_qpos_traj.shape[0]
    x2_qpos_traj = np.tile(x2_model.qpos0, (T, 1))  # default pose per frame

    for g1_name, x2_name in joint_mapping.items():
        if g1_name not in g1_addr or x2_name not in x2_addr:
            print(f"[skip] missing joint: {g1_name} -> {x2_name}")
            continue

        g1_slice, x2_slice = g1_addr[g1_name], x2_addr[x2_name]

        if g1_name == "root":
            # Free joint: copy orientation as-is, but do NOT trust the height/xy —
            # X2's pelvis-to-foot distance differs from G1's, so a straight copy
            # will float or clip the ground. Keep X2's own default root height,
            # only transfer orientation (quat) and horizontal (x, y) motion.
            g1_pos, g1_quat = g1_qpos_traj[:, g1_slice.start:g1_slice.start+3], \
                               g1_qpos_traj[:, g1_slice.start+3:g1_slice.start+7]
            x2_qpos_traj[:, x2_slice.start:x2_slice.start+2] = g1_pos[:, :2]   # x, y
            x2_qpos_traj[:, x2_slice.start+3:x2_slice.start+7] = g1_quat       # quat
            # leave z (height) at X2's default standing height
            continue

        # size mismatch guard (e.g. hinge vs ball)
        src = g1_qpos_traj[:, g1_slice]
        n = min(src.shape[1], x2_slice.stop - x2_slice.start)
        
        # Apply sign flip or offset if specified for this joint
        if g1_name in JOINT_TRANSFORMS:
            mult, offset = JOINT_TRANSFORMS[g1_name]
            x2_qpos_traj[:, x2_slice.start:x2_slice.start + n] = (src[:, :n] * mult) + offset
        else:
            x2_qpos_traj[:, x2_slice.start:x2_slice.start + n] = src[:, :n]

    return x2_qpos_traj


def retarget_ik(g1_model, x2_model, g1_qpos_traj, x2_qpos_prior, max_iters=15, dt=1.0/50):
    g1_data = mujoco.MjData(g1_model)
    x2_data = mujoco.MjData(x2_model)
    scale = compute_limb_scale(g1_model, g1_data, x2_model, x2_data)

    configuration = mink.Configuration(x2_model)

    tasks_by_name = {}
    tasks = []
    
    # Define tasks only for extremities (Feet & Hands)
    for name, spec in KEYPOINTS.items():
        is_foot = "foot" in name
        is_torso = "torso" in name
        
        pos_cost = 0.0 if is_torso else (10.0 if is_foot else 2.0)
        ori_cost = 10.0 if is_torso else (0.1 if is_foot else 0.0)
        
        t = mink.FrameTask(
            frame_name=spec["x2"],
            frame_type=spec["type"],
            position_cost=pos_cost,
            orientation_cost=ori_cost,
            lm_damping=1.0,
        )
        tasks_by_name[name] = t
        tasks.append(t)

    # Posture task pulls unconstrained joints toward mapped warm-start pose
    #posture_task = mink.PostureTask(model=x2_model, cost=1e-1)
    cost_weights = np.full(x2_model.nv, 1e-1) 
    
    # Heavily penalize the extra waist joints to stop the lateral hip sway
    for joint_name in ["waist_roll_joint", "waist_pitch_joint"]:
        jnt_id = mujoco.mj_name2id(x2_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jnt_id != -1:
            dof_id = x2_model.jnt_dofadr[jnt_id]
            cost_weights[dof_id] = 10.0  # Massive penalty for bending these
            
    posture_task = mink.PostureTask(model=x2_model, cost=cost_weights)
    tasks.append(posture_task)
    
    T = g1_qpos_traj.shape[0]
    nq_x2 = x2_model.nq
    x2_qpos_out = np.zeros((T, nq_x2))
    
    configuration.update(x2_qpos_prior[0])

    for t in trange(T, desc="Retargeting frames with IK"):
        # Forward kinematics for source G1
        g1_data.qpos[:] = g1_qpos_traj[t]
        mujoco.mj_forward(g1_model, g1_data)
        g1_pelvis_pos, g1_pelvis_mat = get_frame_pose(g1_model, g1_data, "pelvis", "body")

        # Set posture task prior to mapped state at frame t
        posture_task.set_target(x2_qpos_prior[t])
        #posture_task.set_target(x2_model.qpos0)

        # Forward kinematics for target X2 at prior state
        x2_data.qpos[:] = x2_qpos_prior[t]
        mujoco.mj_forward(x2_model, x2_data)
        x2_pelvis_pos, x2_pelvis_mat = get_frame_pose(x2_model, x2_data, "pelvis", "body")

        # Update Mink solver configuration
        # configuration.update(x2_qpos_prior[t])

        for name, spec in KEYPOINTS.items():
            g_pos, g_mat = get_frame_pose(g1_model, g1_data, spec["g1"], spec["type"])
            target_rot = mink.SO3.from_matrix(g_mat)
            
            if name == "torso":
                # We only care about orientation (pos_cost = 0), but we MUST set a valid SE3 target.
                # We pass the G1's rotation and a dummy position (zeros).
                tasks_by_name[name].set_target(
                    mink.SE3.from_rotation_and_translation(target_rot, np.zeros(3))
                )
                continue
            
            # 1. Express G1 extremity relative to G1 pelvis in Local Pelvis Frame
            rel_world_g1 = g_pos - g1_pelvis_pos
            rel_local_g1 = g1_pelvis_mat.T @ rel_world_g1

            # 2. Scale relative vector in local space
            rel_local_scaled = rel_local_g1 * scale[name]
            # rel_local_scaled = rel_local_g1.copy()
            # rel_local_scaled[0] *= scale[name]  # Scale X (Stride length)
            # rel_local_scaled[1] *= 1.0          # Keep Y (Step width) exactly 1:1
            # rel_local_scaled[2] *= scale[name]  # Scale Z (Leg extension/height)

            # 3. Transform back into World Space relative to X2 pelvis
            target_pos = x2_pelvis_pos + (x2_pelvis_mat @ rel_local_scaled)

            tasks_by_name[name].set_target(
                mink.SE3.from_rotation_and_translation(target_rot, target_pos)
            )

        # Solve IK for frame t
        for _ in range(max_iters):
            vel = mink.solve_ik(configuration, tasks, dt, solver="daqp", damping=1e-3)
            configuration.integrate_inplace(vel, dt)

        x2_qpos_out[t] = configuration.q.copy()

    return x2_qpos_out


def map_G1_to_X2_trajectory(dataset:str="walk", method:str="one_to_one", prior_qpos=None):
    env = ImitationFactory.make("UnitreeG1", default_dataset_conf=DefaultDatasetConf([dataset]))
    
    # reset the env
    key = jax.random.PRNGKey(0)
    env.reset(key)
    
    # Trajectory object (info + data) actually loaded
    traj = env.th.traj
    traj_info = traj.info
    traj_data = traj.data
    
    g1_model = mujoco.MjModel.from_xml_path("/Users/dave/miniconda3/envs/locomujoco/lib/python3.10/site-packages/loco_mujoco_models/unitree_g1/g1_23dof.xml")
    
    x2_path = "/Users/dave/miniconda3/envs/locomujoco/lib/python3.10/site-packages/loco_mujoco_models/agibot_x2/x2.xml"
    spec = mujoco.MjSpec.from_file(x2_path)
    spec.meshdir = "/Users/dave/miniconda3/envs/locomujoco/lib/python3.10/site-packages/loco_mujoco_models/agibot_x2/meshes"
    x2_model = spec.compile()

    g1_qpos = np.asarray(traj_data.qpos)
    
    if method == "one_to_one":
        x2_qpos = retarget_one_to_one(g1_model, x2_model, g1_qpos, JOINT_MAPPING)
    elif method == "ik":
        #x2_qpos_naive = retarget_one_to_one(g1_model, x2_model, g1_qpos, JOINT_MAPPING)
        x2_qpos_prior = np.asarray(prior_qpos)        
        x2_qpos = retarget_ik(g1_model, x2_model, g1_qpos, x2_qpos_prior)
    else:
        raise ValueError(f"Unknown retargeting method: {method}")

    # quick visual check
    #x2_data = mujoco.MjData(x2_model)
    x2_njnt = x2_model.njnt
    x2_jnt_names = [mujoco.mj_id2name(x2_model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(x2_njnt)]
    x2_traj_info = TrajectoryInfo(
        x2_jnt_names,
        model=TrajectoryModel(x2_njnt, jnp.array(x2_model.jnt_type)),
        frequency=traj_info.frequency,   # same frame rate as source, frame count is unchanged
    )
    
    n_frames = x2_qpos.shape[0]
    x2_qvel = np.zeros((n_frames, x2_model.nv))  # placeholder -- see note below

    x2_traj_data = TrajectoryData(
        jnp.array(x2_qpos),
        jnp.array(x2_qvel),
        split_points=jnp.array(traj_data.split_points),  # unchanged: same number of frames, same clip boundaries
    )
    x2_traj = Trajectory(x2_traj_info, x2_traj_data)
    
    x2_save_dir = f"/Users/dave/Library/CloudStorage/OneDrive-PolitecnicodiMilano/Projects/loco-mujoco/datasets/agibotx2/{dataset}"
    os.makedirs(x2_save_dir, exist_ok=True)
    if method == "one_to_one":
        x2_save_path = os.path.join(x2_save_dir, "agibotx2_retargeted_one_to_one.npz")
    elif method == "ik":
        x2_save_path = os.path.join(x2_save_dir, "agibotx2_retargeted_ik.npz")
    x2_traj.save(x2_save_path)
    
    
if __name__ == "__main__":
    dataset = "run"  # "walk" or "run"
    method = "one_to_one"  # "ik" or "one_to_one"
    x2_qpos_prior = None
    
    # if method == 'ik':
    #     # Load the prior trajectory for IK retargeting
    #     traj_prior = Trajectory.load(f"/Users/dave/Library/CloudStorage/OneDrive-PolitecnicodiMilano/Projects/loco-mujoco/datasets/agibotx2/{dataset}/agibotx2_retargeted_one_to_one.npz")
    #     x2_qpos_prior = traj_prior.data.qpos
    
    # map_G1_to_X2_trajectory(dataset=dataset, method=method, prior_qpos=x2_qpos_prior)
    
    env = AgibotX2()
    traj = Trajectory.load(f"/Users/dave/Library/CloudStorage/OneDrive-PolitecnicodiMilano/Projects/loco-mujoco/datasets/agibotx2/{dataset}/agibotx2_retargeted_{method}.npz")
    env.load_trajectory(traj)
    env.play_trajectory(n_episodes=3, n_steps_per_episode=5000, render=True)

from typing import Tuple, List, Union
import mujoco
from mujoco import MjSpec
import numpy as np

import loco_mujoco
from loco_mujoco.core import ObservationType, Observation
from loco_mujoco.environments.humanoids.base_robot_humanoid import BaseRobotHumanoid
from loco_mujoco.core. utils import info_property


class AgibotX2(BaseRobotHumanoid):
    def __init__(self, 
                 disable_arms: bool = False,
                 disable_back_joint: bool = False,
                 spec: Union[str, MjSpec] = None,
                 observation_spec: List[Observation] = None,
                 actuation_spec: List[str] = None,
                 **kwargs) -> None:
        """
        Constructor.

        Args:
            disable_arms (bool): Whether to disable arm joints.
            disable_back_joint (bool): Whether to disable the back joint.
            spec (Union[str, MjSpec]): Specification of the environment. Can be a path to the XML file or an MjSpec object.
                If none is provided, the default XML file is used.
            observation_spec (List[Observation], optional): List defining the observation space. Defaults to None.
            actuation_spec (List[str], optional): List defining the action space. Defaults to None.
            **kwargs: Additional parameters for the environment.
        """
        self._disable_arms = disable_arms
        self._disable_back_joint = disable_back_joint

        if spec is None:
            spec = self.get_default_xml_file_path()

        # load the model specification
        spec = mujoco.MjSpec.from_file(spec) if not isinstance(spec, MjSpec) else spec

        # get the observation and action specification
        if observation_spec is None:
            # get default
            observation_spec = self._get_observation_specification(spec)
        else:
            # parse
            observation_spec = self.parse_observation_spec(observation_spec)
        if actuation_spec is None:
            actuation_spec = self._get_action_specification(spec)

        # modify the specification if needed
        # if self.mjx_enabled:
        #     spec = self._modify_spec_for_mjx(spec)
        # if disable_arms or disable_back_joint:
        #     joints_to_remove, motors_to_remove, equ_constr_to_remove = self._get_xml_modifications()
        #     obs_to_remove = ["q_" + j for j in joints_to_remove] + ["dq_" + j for j in joints_to_remove]
        #     observation_spec = [elem for elem in observation_spec if elem.name not in obs_to_remove]
        #     actuation_spec = [ac for ac in actuation_spec if ac not in motors_to_remove]
        #     spec = self._delete_from_spec(spec, joints_to_remove,
        #                                   motors_to_remove, equ_constr_to_remove)
        #     if disable_arms:
        #         spec = self._reorient_arms(spec)

        super().__init__(spec=spec, actuation_spec=actuation_spec, observation_spec=observation_spec, **kwargs)

    # @staticmethod
    # def get_default_xml_file_path():
    #     # Make sure this points to your actual XML
    #     return "/Users/dave/.loco-mujoco-caches/models/agibotx2/agibotx2.xml"
    
    def _get_xml_modifications(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Specifies which joints, motors, and equality constraints should be removed from the Mujoco XML.

        Returns:
            Tuple[List[str], List[str], List[str]]: A tuple containing lists of joints to remove, motors to remove,
            and equality constraints to remove.
        """

        joints_to_remove = []
        motors_to_remove = []
        equ_constr_to_remove = []

        if self._disable_arms:
            joints_to_remove += ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                                 "right_elbow_pitch_joint", "right_elbow_roll_joint",
                                 "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                                 "left_elbow_pitch_joint", "left_elbow_roll_joint"]
            # actuators are named without the \"_joint\" suffix in the XML
            motors_to_remove += ["right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
                                 "right_elbow_pitch", "right_elbow_roll",
                                 "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
                                 "left_elbow_pitch", "left_elbow_roll"]

        if self._disable_back_joint:
            joints_to_remove += ["torso_joint"]
            motors_to_remove += ["torso_joint"]

        return joints_to_remove, motors_to_remove, equ_constr_to_remove

    @staticmethod
    def _get_observation_specification(spec: MjSpec) -> List[Observation]:
        """
        Returns the observation specification of the environment.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            List[Observation]: A list of observations.
        """

        observation_spec = [# ------------- JOINT POS -------------
                            ObservationType.FreeJointPosNoXY("q_floating_base_joint", xml_name="floating_base_joint"),
                            ObservationType.JointPos("q_left_hip_pitch_joint", xml_name="left_hip_pitch_joint"),
                            ObservationType.JointPos("q_left_hip_roll_joint", xml_name="left_hip_roll_joint"),
                            ObservationType.JointPos("q_left_hip_yaw_joint", xml_name="left_hip_yaw_joint"),
                            ObservationType.JointPos("q_left_knee_joint", xml_name="left_knee_joint"),
                            ObservationType.JointPos("q_left_ankle_pitch_joint", xml_name="left_ankle_pitch_joint"),
                            ObservationType.JointPos("q_left_ankle_roll_joint", xml_name="left_ankle_roll_joint"),
                            
                            ObservationType.JointPos("q_right_hip_pitch_joint", xml_name="right_hip_pitch_joint"),
                            ObservationType.JointPos("q_right_hip_roll_joint", xml_name="right_hip_roll_joint"),
                            ObservationType.JointPos("q_right_hip_yaw_joint", xml_name="right_hip_yaw_joint"),
                            ObservationType.JointPos("q_right_knee_joint", xml_name="right_knee_joint"),
                            ObservationType.JointPos("q_right_ankle_pitch_joint", xml_name="right_ankle_pitch_joint"),
                            ObservationType.JointPos("q_right_ankle_roll_joint", xml_name="right_ankle_roll_joint"),
                            
                            ObservationType.JointPos("q_waist_yaw_joint", xml_name="waist_yaw_joint"),
                            ObservationType.JointPos("q_waist_pitch_joint", xml_name="waist_pitch_joint"),
                            ObservationType.JointPos("q_waist_roll_joint", xml_name="waist_roll_joint"),
                            
                            ObservationType.JointPos("q_left_shoulder_pitch_joint", xml_name="left_shoulder_pitch_joint"),
                            ObservationType.JointPos("q_left_shoulder_roll_joint", xml_name="left_shoulder_roll_joint"),
                            ObservationType.JointPos("q_left_shoulder_yaw_joint", xml_name="left_shoulder_yaw_joint"),
                            ObservationType.JointPos("q_left_elbow_joint", xml_name="left_elbow_joint"),
                            ObservationType.JointPos("q_left_wrist_yaw_joint", xml_name="left_wrist_yaw_joint"),
                            ObservationType.JointPos("q_left_wrist_pitch_joint", xml_name="left_wrist_pitch_joint"),
                            ObservationType.JointPos("q_left_wrist_roll_joint", xml_name="left_wrist_roll_joint"),
                            
                            ObservationType.JointPos("q_right_shoulder_pitch_joint", xml_name="right_shoulder_pitch_joint"),
                            ObservationType.JointPos("q_right_shoulder_roll_joint", xml_name="right_shoulder_roll_joint"),
                            ObservationType.JointPos("q_right_shoulder_yaw_joint", xml_name="right_shoulder_yaw_joint"),
                            ObservationType.JointPos("q_right_elbow_joint", xml_name="right_elbow_joint"),
                            ObservationType.JointPos("q_right_wrist_yaw_joint", xml_name="right_wrist_yaw_joint"),
                            ObservationType.JointPos("q_right_wrist_pitch_joint", xml_name="right_wrist_pitch_joint"),
                            ObservationType.JointPos("q_right_wrist_roll_joint", xml_name="right_wrist_roll_joint"),

                            # ------------- JOINT VEL -------------
                            ObservationType.FreeJointVel("dq_floating_base_joint", xml_name="floating_base_joint"),
                            ObservationType.JointVel("dq_left_hip_pitch_joint", xml_name="left_hip_pitch_joint"),
                            ObservationType.JointVel("dq_left_hip_roll_joint", xml_name="left_hip_roll_joint"),
                            ObservationType.JointVel("dq_left_hip_yaw_joint", xml_name="left_hip_yaw_joint"),
                            ObservationType.JointVel("dq_left_knee_joint", xml_name="left_knee_joint"),
                            ObservationType.JointVel("dq_left_ankle_pitch_joint", xml_name="left_ankle_pitch_joint"),
                            ObservationType.JointVel("dq_left_ankle_roll_joint", xml_name="left_ankle_roll_joint"),
                            
                            ObservationType.JointVel("dq_right_hip_pitch_joint", xml_name="right_hip_pitch_joint"),
                            ObservationType.JointVel("dq_right_hip_roll_joint", xml_name="right_hip_roll_joint"),
                            ObservationType.JointVel("dq_right_hip_yaw_joint", xml_name="right_hip_yaw_joint"),
                            ObservationType.JointVel("dq_right_knee_joint", xml_name="right_knee_joint"),
                            ObservationType.JointVel("dq_right_ankle_pitch_joint", xml_name="right_ankle_pitch_joint"),
                            ObservationType.JointVel("dq_right_ankle_roll_joint", xml_name="right_ankle_roll_joint"),
                            
                            ObservationType.JointVel("dq_waist_pitch_joint", xml_name="waist_pitch_joint"),
                            ObservationType.JointVel("dq_waist_roll_joint", xml_name="waist_roll_joint"),
                            ObservationType.JointVel("dq_waist_yaw_joint", xml_name="waist_yaw_joint"),
                            
                            ObservationType.JointVel("dq_left_shoulder_pitch_joint", xml_name="left_shoulder_pitch_joint"),
                            ObservationType.JointVel("dq_left_shoulder_roll_joint", xml_name="left_shoulder_roll_joint"),
                            ObservationType.JointVel("dq_left_shoulder_yaw_joint", xml_name="left_shoulder_yaw_joint"),
                            ObservationType.JointVel("dq_left_elbow_joint", xml_name="left_elbow_joint"),
                            ObservationType.JointVel("dq_left_wrist_yaw_joint", xml_name="left_wrist_yaw_joint"),
                            ObservationType.JointVel("dq_left_wrist_pitch_joint", xml_name="left_wrist_pitch_joint"),
                            ObservationType.JointVel("dq_left_wrist_roll_joint", xml_name="left_wrist_roll_joint"),
                            
                            ObservationType.JointVel("dq_right_shoulder_pitch_joint", xml_name="right_shoulder_pitch_joint"),
                            ObservationType.JointVel("dq_right_shoulder_roll_joint", xml_name="right_shoulder_roll_joint"),
                            ObservationType.JointVel("dq_right_shoulder_yaw_joint", xml_name="right_shoulder_yaw_joint"),
                            ObservationType.JointVel("dq_right_elbow_joint", xml_name="right_elbow_joint"),
                            ObservationType.JointVel("dq_right_wrist_yaw_joint", xml_name="right_wrist_yaw_joint"),
                            ObservationType.JointVel("dq_right_wrist_pitch_joint", xml_name="right_wrist_pitch_joint"),
                            ObservationType.JointVel("dq_right_wrist_roll_joint", xml_name="right_wrist_roll_joint"),]

        return observation_spec

    @staticmethod
    def _get_action_specification(spec: MjSpec) -> List[str]:
        """
        Returns the action space specification.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            List[str]: A list of actuator names.
        """
        return [actuator.name for actuator in spec.actuators]

    @staticmethod
    def _reorient_arms(spec: MjSpec) -> MjSpec:
        """
        Reorients the arms to prevent collision with the hips when the arms are disabled.

        Args:
            spec (MjSpec): Mujoco specification.

        Returns:
            MjSpec: Modified Mujoco specification.
        """
        # modify the arm orientation
        left_shoulder_pitch_link = [body for body in spec.bodies if body.name == "left_shoulder_pitch_link"][0]
        left_shoulder_pitch_link.quat = [1.0, 0.25, 0.1, 0.0]
        right_elbow_link = [body for body in spec.bodies if body.name == "right_elbow_link"][0]
        right_elbow_link.quat = [1.0, 0.0, 0.25, 0.0]
        right_shoulder_pitch_link = [body for body in spec.bodies if body.name == "right_shoulder_pitch_link"][0]
        right_shoulder_pitch_link.quat = [1.0, -0.25, 0.1, 0.0]
        left_elbow_link = [body for body in spec.bodies if body.name == "left_elbow_link"][0]
        left_elbow_link.quat = [1.0, 0.0, 0.25, 0.0]

        return spec

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        """
        Returns the default XML file path for the Agibot X2 environment.
        """
        return "/Users/dave/miniconda3/envs/locomujoco/lib/python3.10/site-packages/loco_mujoco_models/agibot_x2/x2_ultra.xml"

    @info_property
    def upper_body_xml_name(self) -> str:
        """
        Returns the name of the upper body in the Mujoco XML file.
        """
        return "torso_link"
    
    @info_property
    def root_free_joint_xml_name(self) -> str:
        """
        Returns the name of the root (free) joint in the Mujoco XML file.
        """
        return "floating_base_joint"

    @info_property
    def root_body_name(self) -> str:
        """
        Returns the name of the root body (typically the pelvis) in the Mujoco XML file.
        """
        return "pelvis"

    @info_property
    def root_height_healthy_range(self) -> Tuple[float, float]:
        """
        Returns the healthy range of the root height.

        Returns:
            Tuple[float, float]: The healthy height range (min, max).
        """
        return (0.5, 1.0)

# Register the environment
AgibotX2.register()
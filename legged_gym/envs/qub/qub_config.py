
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
#
# QUB cfg — Phase A + first-step stabilization + reward-scale rebalancing.
#
# Run history:
#   Phase A initial:    NaN at first physics step (actions clipped at 100
#                       sent >2000 Nm torque to PD controller).
#   clip_actions=5 fix: training survived 2 iterations but value loss
#                       grew to 10^19 -> 10^24, then NaN at iter 2 mid-step.
#                       Diagnosed: rew_ang_vel_xy = -1.9 at iter 1 means
#                       base_ang_vel ~38 rad/s; large penalties + small
#                       clip_single_reward (5) let value function explode.
# Rebalanced scales below (vs SF baseline) to keep per-step penalties under
# clip_single_reward=1.0 even when robot misbehaves.

from legged_gym.envs.base.base_config import BaseConfig


class BipedCfgQUB(BaseConfig):
    class env:
        num_envs = 4096
        # 3 (ang_vel) + 3 (gravity) + 13 (dof_pos) + 13 (dof_vel)
        # + 13 (actions) + 2 (clock sin/cos) + 4 (gaits) = 51
        num_observations = 51
        num_critic_observations = 3 + num_observations  # +lin_vel
        num_height_samples = 117
        num_actions = 13  # 12 leg + 1 torso_yaw
        env_spacing = 3.0
        send_timeouts = True
        episode_length_s = 20
        obs_history_length = 5
        dof_vel_use_pos_diff = True
        fail_to_terminal_time_s = 0.5

    class terrain:
        mesh_type = "plane"
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25
        curriculum = True
        static_friction = 0.4
        dynamic_friction = 0.4
        restitution = 0.8
        measure_heights = False
        critic_measure_heights = True
        measured_points_x = [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
                             0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 5 + 4
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10
        num_cols = 20
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        slope_treshold = 0.75
        simplify_grid = False
        edge_width_thresh = 0.01
        high_horizontal_scale = 0.01
        edge_width_thresh_up = 0.18
        edge_width_thresh_down = 0.05

    class commands:
        curriculum = False
        smooth_max_lin_vel_x = 2.0
        smooth_max_lin_vel_y = 1.0
        non_smooth_max_lin_vel_x = 1.0
        non_smooth_max_lin_vel_y = 1.0
        max_ang_vel_yaw = 3.0
        curriculum_threshold = 0.75
        num_commands = 3 + 2
        resampling_time = 5.0
        heading_command = False
        min_norm = 0.1
        zero_command_prob = 0.8

        class ranges:
            lin_vel_x = [-0.5, 1.0]
            lin_vel_y = [-0.5, 0.5]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14159, 3.14159]
            base_height = [0.72, 0.80]
            stand_still = [0, 1]

    class gait:
        num_gait_params = 4
        resampling_time = 5
        touch_down_vel = 0.0

        class ranges:
            frequencies = [1.0, 1.5]
            offsets = [0.5, 0.5]
            durations = [0.5, 0.5]
            swing_height = [0.08, 0.15]

    class init_state:
        pos = [0.0, 0.0, 0.88]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]

        default_joint_angles = {
            "torso_yaw_joint":      0.0,
            "L_hip_pitch_joint":    0.0,
            "L_hip_roll_joint":     0.0,
            "L_hip_yaw_joint":      0.0,
            "L_knee_pitch_joint":   0.0,
            "L_ankle_pitch_joint":  0.0,
            "L_ankle_roll_joint":   0.0,
            "R_hip_pitch_joint":    0.0,
            "R_hip_roll_joint":     0.0,
            "R_hip_yaw_joint":      0.0,
            "R_knee_pitch_joint":   0.0,
            "R_ankle_pitch_joint":  0.0,
            "R_ankle_roll_joint":   0.0,
        }

        init_stand_joint_angles = {
            "torso_yaw_joint":      0.0,

            "L_hip_pitch_joint":    -0.30,
            "L_hip_roll_joint":      0.0,
            "L_hip_yaw_joint":       0.0,
            "L_knee_pitch_joint":   -0.60,
            "L_ankle_pitch_joint":   0.30,
            "L_ankle_roll_joint":    0.0,

            "R_hip_pitch_joint":     0.30,
            "R_hip_roll_joint":      0.0,
            "R_hip_yaw_joint":       0.0,
            "R_knee_pitch_joint":    0.60,
            "R_ankle_pitch_joint":   0.30,
            "R_ankle_roll_joint":    0.0,
        }

    class control:
        action_scale = 0.25
        control_type = "P"

        stiffness = {
            "torso_yaw_joint":      40.0,

            "L_hip_pitch_joint":    40.0,
            "L_hip_roll_joint":     40.0,
            "L_hip_yaw_joint":      40.0,
            "L_knee_pitch_joint":   60.0,
            "L_ankle_pitch_joint":  20.0,
            "L_ankle_roll_joint":   20.0,

            "R_hip_pitch_joint":    40.0,
            "R_hip_roll_joint":     40.0,
            "R_hip_yaw_joint":      40.0,
            "R_knee_pitch_joint":   60.0,
            "R_ankle_pitch_joint":  20.0,
            "R_ankle_roll_joint":   20.0,
        }
        damping = {
            "torso_yaw_joint":       2.0,

            "L_hip_pitch_joint":     2.0,
            "L_hip_roll_joint":      2.0,
            "L_hip_yaw_joint":       2.0,
            "L_knee_pitch_joint":    3.0,
            "L_ankle_pitch_joint":   1.0,
            "L_ankle_roll_joint":    1.0,

            "R_hip_pitch_joint":     2.0,
            "R_hip_roll_joint":      2.0,
            "R_hip_yaw_joint":       2.0,
            "R_knee_pitch_joint":    3.0,
            "R_ankle_pitch_joint":   1.0,
            "R_ankle_roll_joint":    1.0,
        }

        decimation = 8
        user_torque_limit = 120.0
        max_power = 1000.0

        pull_off_robots = False
        pull_interval_s = 6
        max_pull_vel_z = 0.25
        force_duration_s = 3.0

    class asset:
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/QUB/urdf/QUB.urdf"
        name = "qub"
        foot_name = "foot"
        foot_radius = 0.0
        penalize_contacts_on = ["thigh", "calf", "hip"]
        terminate_after_contacts_on = ["base", "pelvis"]
        disable_gravity = False
        collapse_fixed_joints = True
        fix_base_link = False
        default_dof_drive_mode = 3
        self_collisions = 0
        replace_cylinder_with_capsule = True
        flip_visual_attachments = False
        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 1000.0
        max_linear_velocity = 1000.0
        armature = 0.0
        thickness = 0.01

    class domain_rand:
        randomize_friction = False
        friction_range = [0.0, 1.6]
        randomize_restitution = False
        restitution_range = [0.0, 1.0]
        randomize_base_mass = False
        added_mass_range = [-0.5, 5]
        randomize_base_com = False
        rand_com_vec = [0.03, 0.02, 0.03]
        randomize_inertia = False
        randomize_inertia_range = [0.8, 1.2]
        push_robots = False
        push_interval_s = 7
        max_push_vel_xy = 1.0
        rand_force = False
        force_resampling_time_s = 15
        max_force = 50.0
        rand_force_curriculum_level = 0
        randomize_Kp = False
        randomize_Kp_range = [0.8, 1.2]
        randomize_Kd = False
        randomize_Kd_range = [0.8, 1.2]
        randomize_motor_torque = False
        randomize_motor_torque_range = [0.8, 1.2]
        randomize_default_dof_pos = False
        randomize_default_dof_pos_range = [-0.05, 0.05]
        randomize_action_delay = False
        randomize_imu_offset = False
        delay_ms_range = [0, 20]

    class rewards:
        class scales:
            # Tracking (positive rewards)
            keep_balance = 1.0
            tracking_lin_vel_x = 1.5
            tracking_lin_vel_y = 1.5
            tracking_ang_vel = 1.0

            # ====== Regulation (negative rewards) ======
            # Iter-1 diagnostics showed rew_ang_vel_xy=-1.9 dominating loss.
            # Scaled down 5-10x to keep per-step penalties below clip=1.0.
            base_height = -1.0       # was -10
            lin_vel_z = -0.1         # was -0.5
            ang_vel_xy = -0.01       # was -0.05
            torques = -0.00008
            dof_acc = -2.5e-7
            action_rate = -0.01
            dof_pos_limits = -2.0
            collision = -10.0        # was -100
            action_smooth = -0.01
            orientation = -1.0       # was -5.0
            feet_distance = -10.0    # was -100
            feet_regulation = -0.05
            tracking_contacts_shaped_force = -2.0
            tracking_contacts_shaped_vel = -2.0
            tracking_contacts_shaped_height = -2.0
            feet_contact_forces = -0.002
            ankle_torque_limits = -0.1
            power = -2e-4
            relative_feet_height_tracking = 1.0
            zero_command_nominal_state = -1.0   # was -10
            keep_ankle_pitch_zero_in_air = 1.0
            foot_landing_vel = -2.0  # was -10

        only_positive_rewards = False
        clip_reward = 100
        # ====== Per-step single-reward clip ======
        # Iter-0/1 showed rew_ang_vel_xy = -1.9 (well above 1.0). Now with
        # scales reduced, this should comfortably fit under clip=1.0 and
        # avoid value-function explosion.
        clip_single_reward = 1.0   # was 5
        tracking_sigma = 0.2
        ang_tracking_sigma = 0.25
        height_tracking_sigma = 0.01
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 0.8
        base_height_target = 0.78
        feet_height_target = 0.08
        min_feet_distance = 0.18
        max_contact_force = 100.0
        kappa_gait_probs = 0.05
        gait_force_sigma = 25.0
        gait_vel_sigma = 0.25
        gait_height_sigma = 0.005
        about_landing_threshold = 0.05

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            dof_acc = 0.0025
            height_measurements = 5.0
            contact_forces = 0.01
            torque = 0.05
            base_z = 1.0 / 0.78

        clip_observations = 100.0
        clip_actions = 5.0

    class noise:
        add_noise = True
        noise_level = 1.5

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class viewer:
        ref_env = 0
        pos = [5, -5, 3]
        lookat = [0, 0, 0]
        realtime_plot = True

    class sim:
        dt = 0.0025
        substeps = 1
        gravity = [0.0, 0.0, -9.81]
        up_axis = 1

        class physx:
            num_threads = 0
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23
            default_buffer_size_multiplier = 5
            contact_collection = 2


class BipedCfgPPOQUB(BaseConfig):
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class MLP_Encoder:
        output_detach = True
        num_input_dim = BipedCfgQUB.env.num_observations * BipedCfgQUB.env.obs_history_length
        num_output_dim = 3
        hidden_dims = [256, 128]
        activation = "elu"
        orthogonal_init = False
        encoder_des = "Base linear velocity"

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"
        orthogonal_init = False
        fix_std_noise_value = None

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.0e-3
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        # ====== Stricter gradient clipping ======
        # Iter-1 value loss = 10^24 indicates massive gradients leaking
        # into the policy. Tightened from 1.0 to 0.5 for stability.
        max_grad_norm = 0.5

        est_learning_rate = 1.0e-3
        ts_learning_rate = 1.0e-4
        critic_take_latent = True

    class runner:
        encoder_class_name = "MLP_Encoder"
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24
        max_iterations = 10000

        logger = "tensorboard"
        exptid = ""
        wandb_project = "legged_gym_QUB"
        save_interval = 500
        experiment_name = "QUB"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None

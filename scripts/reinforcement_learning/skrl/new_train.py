import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)

parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import random
from datetime import datetime

import skrl
from packaging import version
# Add to your imports near the top
import numpy as np
import cv2
import requests

# For Detecron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import get_model_zoo_configs
from detectron2.model_zoo.model_zoo import get_checkpoint_url
import torch

# check for minimum supported skrl version
SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"

# --- Detecron2 Model Initialization ---
# This part should be moved outside the main function or initialized once.
# For simplicity, we'll initialize a generic Detecron2 model here.
# You would replace this with your specific trained model.
def setup_detectron2_predictor():
    cfg = get_cfg()
    # Add project-specific config (e.g., if you have a custom dataset or model architecture)
    # from detectron2.projects.<your_project_name> import add_your_project_config
    # add_your_project_config(cfg)

    # Use a pre-trained model for demonstration. Replace with your model.
    # For example, a COCO-pretrained Mask R-CNN:
    cfg.merge_from_file(get_model_zoo_configs()["COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"])
    cfg.MODEL.WEIGHTS = get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Use GPU if available

    predictor = DefaultPredictor(cfg)
    return predictor

# Initialize the predictor once
detectron2_predictor = None
try:
    import detectron2
    detectron2_predictor = setup_detectron2_predictor()
    print("[INFO] Detecron2 predictor initialized.")
except ImportError:
    print("[WARNING] Detectron2 is not installed. Image processing will be skipped.")
    print("Please install Detectron2 to enable this feature: pip install 'detectron2@git+https://github.com/facebookresearch/detectron2.git'")
except Exception as e:
    print(f"[ERROR] Failed to initialize Detectron2 predictor: {e}")
    detectron2_predictor = None


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    runner = Runner(env, agent_cfg)

    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    # --- Custom Training Loop for Camera and Detecron2 Integration ---
    # We need to manually control the simulation steps to access camera data.
    # The default skrl runner.run() will abstract this.
    # So, we'll mimic the runner's behavior and add our logic.

    # Retrieve camera sensor from the environment
    # Assuming 'front_cam' is the name given in your scene configuration
    camera = env.unwrapped.scene["camera"]# For now assuming only one camera in env

    # Calculate steps per second for image capture
    sim_dt = env_cfg.sim.dt
    decimation = env_cfg.decimation
    simulation_steps_per_second = 1.0 / (sim_dt * decimation)
    print(f"[INFO] Simulation steps per second: {simulation_steps_per_second}")
    # We want to capture every 1 second, so capture_interval_steps is simulation_steps_per_second
    capture_interval_steps = int(simulation_steps_per_second)
    if capture_interval_steps == 0: # Ensure at least 1 step if sim_dt or decimation is very large
        capture_interval_steps = 1
    print(f"[INFO] Capturing image every {capture_interval_steps} simulation steps.")

    # Initialize environment and get initial observations
    observations, info = env.reset()
    runner.agent.set_initial_states(observations=observations)

    # Training loop
    current_step = 0
    while current_step < agent_cfg["trainer"]["timesteps"]:
        # Sample an action from the agent
        actions = runner.agent.act(observations, info["truncation"], inference=False)

        # Apply action to environment and step simulation
        observations, rewards, terminations, truncations, info = env.step(actions)

        # Update agent and train
        runner.agent.record(observations, actions, rewards, terminations, truncations, info)
        runner.agent.post_step()

        # --- Image Capture and Detecron2 Inference ---
        if camera and detectron2_predictor and current_step % capture_interval_steps == 0:
            print(f"--- Processing camera data at simulation step: {current_step} ---")
            for env_idx in range(env_cfg.scene.num_envs):
                # Access the RGB image data for the current environment
                # The data is typically in (H, W, C) format, and already a numpy array
                rgb_image = camera.data.output["rgb"][env_idx]
                rgb_image_np = rgb_image.numpy() if isinstance(rgb_image, torch.Tensor) else rgb_image

                # Convert from RGB to BGR for Detecron2 (OpenCV convention)
                bgr_image = cv2.cvtColor(rgb_image_np, cv2.COLOR_RGB2BGR)

                # Perform Detecron2 inference
                try:
                    outputs = detectron2_predictor(bgr_image)
                    instances = outputs["instances"].to("cpu")

                    if len(instances) > 0:
                        # Extract bounding boxes (x1, y1, x2, y2)
                        # And optionally scores and class_ids
                        bboxes = instances.pred_boxes.tensor.numpy()
                        scores = instances.scores.numpy()
                        class_ids = instances.pred_classes.numpy()

                        print(f"Env {env_idx}: Detected {len(instances)} objects.")
                        for i in range(len(instances)):
                            x1, y1, x2, y2 = bboxes[i]
                            score = scores[i]
                            class_id = class_ids[i]
                            print(f"  Object {i}: Class ID: {class_id}, Score: {score:.2f}, BBox: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

                            # --- PLACEHOLDER: Use the coordinates to "print" an object ---
                            # This is where you would implement your logic to use the detected coordinates.
                            # For example, you might:
                            # 1. Convert pixel coordinates to 3D world coordinates.
                            #    This is complex and would involve camera intrinsics and extrinsics,
                            #    and potentially depth data if available from the camera.
                            #    You would need to use `isaaclab.sensors.Camera` helper functions or
                            #    OmniGraph nodes to achieve this.
                            # 2. Spawn a new object in the simulation at the inferred 3D position.
                            # 3. Apply a force/torque to an existing object based on detection.
                            # 4. Trigger some other simulation event.

                            # Example: If you wanted to roughly draw a rectangle on the image for visualization (not in sim)
                            # cv2.rectangle(bgr_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            # cv2.putText(bgr_image, f"{class_id}: {score:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    else:
                        print(f"Env {env_idx}: No objects detected.")

                except Exception as e:
                    print(f"[ERROR] Detecron2 inference failed for Env {env_idx}: {e}")

        current_step += 1

    # Finalize runner (save model, etc.)
    runner.finalize()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
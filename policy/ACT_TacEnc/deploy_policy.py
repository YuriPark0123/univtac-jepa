import sys
import json
from pathlib import Path

from .._base_policy import BasePolicy
from .paths import UNIVTAC_POLICY, ckpt_dir as get_ckpt_dir

import os
import cv2
import yaml
import numpy as np
import torch
from .act_policy import ACT
# from act_policy import ACT
from torchvision import transforms

class Policy(BasePolicy):
# class Policy:
    def __init__(self, args):
        """Initialize ACT policy for TacArena deployment"""
        # Construct checkpoint directory path
        self.train_config_name = os.environ.get('TRAIN_CONFIG', 'train_config')
        self.ep_num = os.environ.get('EP_NUM', '50')
        # checkpoints are keyed by the training data version (isaac45 / isaac51), which must match the eval simulator
        self.data_version = os.environ.get('DATA_VERSION', args.get('data_version', 'isaac45'))
        ckpt_dir = get_ckpt_dir(args['task_name'], self.data_version, self.ep_num, self.train_config_name)
 
        self.task_name = args['task_name']
        with open(UNIVTAC_POLICY / 'task_settings.json', 'r') as f:
            task_settings = json.load(f)
        assert self.task_name in task_settings, f"Task '{self.task_name}' not found in task_settings.json"
        self.camera_type = task_settings[self.task_name].get('camera_type', 'head')
        print(f"Using camera type '{self.camera_type}' for task '{self.task_name}'")

        with open(Path(__file__).parent / f'{self.train_config_name}.yml', 'r') as f:
            train_config = yaml.load(f, Loader=yaml.FullLoader)
        
        # tactile frame history: data offsets -> eval steps (one recorded data frame = eval_steps_per_data_frame policy calls)
        from tactile_encoders import get_frame_offsets
        from .utils import load_tactile_background
        self.tactile_names = train_config.get('tactile_names', [])
        self.frame_offsets = get_frame_offsets(train_config['tactile_encoder'], **(train_config.get('tactile_encoder_kwargs') or {}))
        self.eval_steps_per_data_frame = int(args.get('eval_steps_per_data_frame', 2))
        self.tactile_background = load_tactile_background(
            train_config.get('tactile_background'), train_config.get('tactile_background_files'), self.tactile_names)
        self.tactile_history = {}
        print(f"Tactile encoder '{train_config['tactile_encoder']}' frame offsets {self.frame_offsets}, "
              f"{self.eval_steps_per_data_frame} eval steps per data frame")

        train_config.update({
            'task_name': f"sim-{args['task_name']}-{self.data_version}-{self.ep_num}",
            'task_config': args['task_config'],
            'ckpt_dir': str(ckpt_dir),
            "seed": args.get('seed', 0),
            "num_epochs": 1
        })
        
        # Initialize ACT model (RoboTwin_Config=None for TacArena)
        self.model = ACT(train_config)
        print(f"ACT policy loaded from {ckpt_dir}")

    def encode_obs(self, observation):
        """
        Encode TacArena observation to ACT input format
        
        Input (TacArena):
            observation = {
                "observation": {"head": {"rgb": torch.Tensor([H, W, 3])}},  # HWC, 0-255
                "joint_action": torch.Tensor([9])  # [arm(7), gripper(1), extra(1)]
            }
            camera: 480x270
            tactile: 320x240
        
        Output (ACT):
            obs = {
                "qpos": torch.Tensor([8])  # [arm(7), gripper(1)]
                "cam_high": torch.Tensor([3, 256, 256]),  # CHW, 0-1
                "tac_left": torch.Tensor([3, 256, 256]),  # CHW, 0-1
                "tac_right": torch.Tensor([3, 256, 256]),  # CHW, 0-1
            }
        """
        # Debug: observation structure validated
        # observation['embodiment']['joint'] contains joint state
        def camera_transform(img: torch.Tensor):
            img = transforms.Resize((256, 256))(img.permute(2, 0, 1))  # HWC -> CHW
            img = img / 255.0  # Normalize to [0, 1]
            img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
            return img
        
        def tactile_transform(img: torch.Tensor):
            # raw resolution, [0, 1]; the tactile encoder resizes / normalises (same as training)
            return img.permute(2, 0, 1).float() / 255.0  # HWC -> CHW

        def tactile_frames(name: str, img: torch.Tensor):
            history = self.tactile_history.setdefault(name, [])
            history.append(img)
            frames = []
            for off in self.frame_offsets:
                if off == "first":
                    frame = history[0] if self.tactile_background is None \
                        else torch.from_numpy(self.tactile_background[name]).to(img.device)
                else:
                    frame = history[max(0, len(history) - 1 + off * self.eval_steps_per_data_frame)]
                frames.append(tactile_transform(frame))
            return torch.stack(frames, dim=0)  # (num_frames, 3, H, W)

        if self.camera_type == 'all':
            cam_high = camera_transform(observation["observation"]["head"]["rgb"])
            cam_wrist = camera_transform(observation["observation"]["wrist"]["rgb"])
        else:
            cam_high = camera_transform(observation["observation"][self.camera_type]["rgb"])

        left_tac = tactile_frames("tac_left", observation["tactile"]["left_tactile"]["rgb_marker"])
        right_tac = tactile_frames("tac_right", observation["tactile"]["right_tactile"]["rgb_marker"])
        
        # Extract joint positions (8D: 7 arm + 1 gripper)
        qpos = observation["embodiment"]["joint"][:8]

        ret = {
            "cam_high": cam_high,
            "tac_left": left_tac,
            "tac_right": right_tac,
            "qpos": qpos.cpu().numpy()
        }
        if self.camera_type == 'all':
            ret["cam_wrist"] = cam_wrist
        return ret

    def eval(self, task, observation):
        """
        Evaluate ACT policy on TacArena task
        
        Args:
            task: TacArena BaseTask instance
            observation: Current observation from environment
        """
        
        # Get action from ACT model (returns (1, 8) numpy array)
        obs = self.encode_obs(observation)
        if self.model.t % 10 == 0:
            self.save(task.get_frame_shot(observation), task.take_action_cnt)
        action = self.model.get_action(obs).reshape(-1)
        action = torch.from_numpy(action).to(task.device).float()
        exec_succ, eval_succ = task.take_action(action, action_type='qpos')

    def reset(self):
        """Reset ACT model state (temporal aggregation and timestep counter) and the tactile frame history"""
        self.tactile_history = {}
        if hasattr(self.model, 'reset'):
            self.model.reset()

    def save(self, img, t):
        from PIL import Image
        from PIL import ImageDraw, ImageFont
        
        obs = Image.fromarray(img.cpu().numpy())

        draw = ImageDraw.Draw(obs)
        font = ImageFont.load_default()

        draw.text((obs.width-100, obs.height-60), f'{t:03d}', fill=(255, 0, 0), font=font)
        obs.save(f'ACT_{self.task_name}_{self.train_config_name}.png')

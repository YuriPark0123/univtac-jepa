"""
Convert released UniVTAC demos (data/<data_version>/<task>/hdf5) into ACT_TacEnc training episodes.

Differences from policy/ACT/process_data.py:
  * reads data/<data_version>/<task> (e.g. data/isaac45/insert_hole) instead of data/<task>/<task_config>
  * tactile images are kept at their raw resolution (240x320); each tactile encoder resizes them itself
  * output folder and SIM_TASK_CONFIGS key carry the data version: sim-<task>-<data_version>-<N>
  * outputs go to <UniVTAC>/data/act_tacenc (paths.py), not next to the code
  * the head camera is saved as 'cam_high' (the name the ACT configs and deploy code read)
  * joints are cut to the first 8 dims (state_dim 8 = 7 arm + gripper, as in deploy_policy and the public stats)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import UNIVTAC_POLICY, OUT_ROOT, SIM_TASK_CONFIGS_PATH
sys.path.append(str(UNIVTAC_POLICY))

from _base_data_preprocessor import *


class VersionedDataPreprocessor(BaseDataPreprocessor):

    def __init__(self, task_name: str, data_version: str):
        super().__init__(task_name, data_version)
        self.raw_root_path = DATA_ROOT_PATH / data_version / task_name
        self.raw_hdf5_path = sorted(self.raw_root_path.rglob('*.hdf5'), key=lambda x: int(x.stem))
        # train_config*.yml and deploy_policy.py use 'cam_high'; the base preprocessor saves the head camera as 'cam_head'
        self.camera_key_map['visual/head']['save_key'] = 'cam_high'

    def joint_transform(self, joints: np.ndarray) -> np.ndarray:
        # raw embodiment/joint is 9-D; ACT uses state_dim 8 and deploy_policy feeds joint[:8]
        # (the public checkpoints' dataset_stats.pkl are 8-D and match the first 8 dims of isaac45 data)
        return joints[..., :8]

    def tactile_transform(self, images: np.ndarray) -> np.ndarray:
        return np.stack(images, axis=0).astype(np.uint8)


def main(task_name, data_version, expert_data_num):
    output_path = OUT_ROOT / f"sim-{task_name}" / f"{data_version}-{expert_data_num}"

    with open(UNIVTAC_POLICY / 'task_settings.json', 'r') as f:
        task_settings = json.load(f)
    camera_type = task_settings.get(task_name, {}).get('camera_type', 'head')
    visual_cameras = ['head', 'wrist'] if camera_type == 'all' else [camera_type]

    processor = VersionedDataPreprocessor(task_name, data_version)
    assert len(processor.raw_hdf5_path) > 0, f"No hdf5 found under {processor.raw_root_path}"
    metadata = processor.run(
        save_root_path=output_path,
        visual_cameras=visual_cameras,
        tactile_cameras=['left', 'right'],
        downsample_factor=1,
        episode_num=expert_data_num,
        random_select=False,
    )

    try:
        with open(SIM_TASK_CONFIGS_PATH, "r") as f:
            SIM_TASK_CONFIGS = json.load(f)
    except Exception:
        SIM_TASK_CONFIGS = {}

    SIM_TASK_CONFIGS[f"sim-{task_name}-{data_version}-{expert_data_num}"] = {
        "dataset_dir": metadata['dataset_dir'],
        "num_episodes": metadata['num_episodes'],
        "episode_len": metadata['episode_len'],
        "camera_names": metadata['camera_names'],
        "data_version": data_version,
        "raw_episodes": [str(p) for p in processor.selected_raw_hdf5_paths],
    }

    with open(SIM_TASK_CONFIGS_PATH, "w") as f:
        json.dump(SIM_TASK_CONFIGS, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process released UniVTAC episodes for ACT_TacEnc training.")
    parser.add_argument("task_name", type=str, help="e.g. insert_hole")
    parser.add_argument("data_version", type=str, choices=["isaac45", "isaac51"], help="must match the eval simulator")
    parser.add_argument("expert_data_num", type=int, help="Number of episodes to process")
    args = parser.parse_args()
    main(args.task_name, args.data_version, args.expert_data_num)

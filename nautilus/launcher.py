"""
Template code to launch several nautilus tasks with a hyperparameter sweep; from Stone Tao
"""
import json
import logging
import os
import os.path as osp
import random
import string
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from dacite import from_dict
from omegaconf import OmegaConf
from sklearn.model_selection import ParameterGrid

try:
    user = os.environ["NAUTILUS_USER"]
    wandb_key = os.environ["NAUTILUS_WANDB_KEY"]
    github_token = os.environ["NAUTILUS_GITHUB_TOKEN"] # username:token
except:
    logging.error(
        "You are missing some environment variables. Please add NAUTILUS_USER, NAUTILUS_WANDB_KEY, and NAUTILUS_GITHUB_TOKEN"
    )
    exit()


def generate_job_file(template, replace_dict=dict(), dest="job.yml"):
    # Read in the file
    Path(osp.dirname(dest)).mkdir(parents=True, exist_ok=True)
    with open(template, "r") as file:
        filedata = file.read()

    # Replace the target string
    for k, v in replace_dict.items():
        filedata = filedata.replace(f"${k}$", str(v))

    # Write the file out again
    with open(dest, "w") as file:
        file.write(filedata)


def launch_exps(template: str, dest: str, job_prefix: str):
    
    # 5th seed: 7050
    pgrid = {"seed": [4796, 2044, 8946, 9689],  
        "env" : ['PickCube-v1'],
        "obs" : ['state']}
          
    # demos=-1 means using all demos
    parameters = ParameterGrid(pgrid)

    CPUS = 16
    MEM = "20Gi"
    STORAGE = "50Gi"
    GPUS = 1
    print(f"Launch from template {template}. Creating {len(parameters)} jobs and temp yml files to {dest}")
    params_json = dict()
    replication_json = dict(
        template=template, pgrid=pgrid, job_prefix=job_prefix, dest=dest, jobs=dict()
    )
    for i, p in enumerate(parameters):
        seed = p["seed"]
        env = p["env"]
        obs = p["obs"]
        replace_dict = {
            "job_name": f"{user}-{job_prefix}-{i}",
            "wandb_key": wandb_key,
            "github_token": github_token,
            "cpu_request": CPUS,
            "mem_request": MEM,
            "gpu_request": GPUS,
            "storage_request": STORAGE,
            "cpu_request_limit": CPUS,
            "mem_request_limit": MEM,
            "gpu_request_limit": GPUS,
            "command": f""" 
                XLA_PYTHON_CLIENT_PREALLOCATE=false python train_sac.py configs/base_sac_ms3.yml \
                    sac.replay_buffer_capacity=1_000_000 \
                    seed={seed} train.steps=1_000_000 env.env_id={env} env.num_envs=32 env.env_type="gym:gpu" env.env_kwargs.obs_mode={obs} \
                    logger.exp_name="sac-{env}-{obs}-{seed}-walltime_efficient" \
                    logger.wandb=True logger.project_name="sac_jax" logger.wandb_cfg.group="{env}-{obs}"
            """,
        }
        # XLA_PYTHON_CLIENT_PREALLOCATE=false python train_sac.py configs/base_sac_ms3.yml \
        #             sac.replay_buffer_capacity=100_000 \
        #             seed={seed} train.steps=1_000_000 env.env_id={env} env.num_envs=32 env.env_type="gym:gpu" env.env_kwargs.obs_mode={obs} env.render_size=64 \
        #             logger.exp_name="sac-{env}-{obs}-{seed}-walltime_efficient" \
        #             logger.wandb=True logger.project_name="sac_jax" logger.wandb_cfg.group="{env}-{obs}"
        generate_job_file(
            template, replace_dict=replace_dict, dest=osp.join(dest, f"job_{i}.yml")
        )
        params_json[i] = p
        replication_json["jobs"][i] = replace_dict

    with open(osp.join(dest, "params.json"), "w") as f:
        json.dump(dict(params=params_json, indent=2), f)
    with open(osp.join(dest, "config.json"), "w") as f:
        json.dump(replication_json, f, indent=2)
    return dest

def launch_from_json(json_path, dest=None):
    """
    Launch from a config.json file generated with launch_exps function. Useful for launching experiments again and changing user / wandbi key
    """
    with open(json_path, "r") as f:
        config = json.load(f)
    if dest is None: dest = config["dest"]
    job_prefix = config["job_prefix"]
    template = config["template"]
    print(f"Launch from json {json_path}. Creating {len(config['jobs'])} jobs and temp yml files to {dest}")
    replication_json = dict(
        template=template, pgrid=config["pgrid"], job_prefix=job_prefix, dest=dest, jobs=dict()
    )
    for k in config["jobs"]:
        replace_dict = config["jobs"][k]
        replace_dict["job_name"] = f"{user}-{job_prefix}-{k}"
        replace_dict["wandb_key"] = wandb_key
        replace_dict["github_token"] = github_token
        generate_job_file(
            template,
            replace_dict=config["jobs"][k],
            dest=osp.join(dest, f"job_{k}.yml"),
        )
        replication_json["jobs"][k] = replace_dict
    with open(osp.join(dest, "config.json"), "w") as f:
        json.dump(replication_json, f, indent=2)
    return dest

from typing import Union
@dataclass
class LauncherConfig:
    json_path: Union[str, None] = None
    template: Union[str, None] = None
    dest: Union[str, None] = None
    job_prefix: Union[str, None] = None
    dry_run: bool = False


def parse_args():
    cfg = OmegaConf.from_cli()
    return from_dict(data_class=LauncherConfig, data=cfg)

import shutil
if __name__ == "__main__":
    """
    Run python nautilus/launcher.py template

    To regenerate jobs
    if dry_run=True, then jobs will not automatically be launched onto Nautilus, only the job.yml files will be created

    ## Example use

    Generating your own templates
    ```
    python nautilus/launcher.py template=nautilus/job_d4rl_template.yml dest=nautilus/my_experiment job_prefix=my_exp
    ```

    Reusing a template written by someone else. YOU MUST do this (or else you use their username and keys!)
    ```
    python nautilus/launcher.py json_path=nautilus/my_experiment/config.json
    ```
    This will regenerate the job template using your wandb API key and nautilus username

    Note that all job_*.yml files are not commited to git. These are specific to you!
    
    """
    cfg = parse_args()
    
    if cfg.dest is not None:
        if "nautilus" not in osp.abspath(cfg.dest):
            raise ValueError(f"dest: {cfg.dest} invalid. The absolute path must be a subdirectory of nautilus")
        if osp.exists(cfg.dest):
            res = input(f"{cfg.dest} exists, delete and replace? Y/N ")
            if res.lower() == "y":
                print("replacing...")
                shutil.rmtree(cfg.dest)
            else:
                exit()
    dest = cfg.dest
    if cfg.json_path is not None:
        dest = launch_from_json(cfg.json_path, cfg.dest)
    elif cfg.template is not None:
        assert cfg.dest is not None
        assert cfg.job_prefix is not None
        launch_exps(cfg.template, cfg.dest, cfg.job_prefix)
    else:
        raise ValueError("Missing json_path or template argument")
    print(f"Created job templates. Example at {osp.join(dest, 'job_0.yml')}")
    cmd = f"kubectl create $(ls {dest}/*.yml | awk ' {{ print \" -f \" $1 }} ')"
    print(f"To run experiment run\n{cmd}")
    cmd = f"kubectl delete $(ls {dest}/*.yml | awk ' {{ print \" -f \" $1 }} ')"
    print(f"To delete experiment run\n{cmd}")
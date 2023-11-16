import hydra, os
from omegaconf import DictConfig, OmegaConf
from fmbase.util.ops import fmbdir
from graphcast import checkpoint
from typing import Any, Mapping, Sequence, Tuple, Union
from graphcast.graphcast import ModelConfig, TaskConfig, CheckPoint

def cfg() -> DictConfig:
    return Configuration.instance().cfg

def configure(config_name: str):
    Configuration.init( config_name )

class Configuration:
    _instance = None
    _instantiated = None

    def __init__(self, config_name: str ):
        self.cfg: DictConfig = hydra.compose( config_name, return_hydra_config=True )

    @classmethod
    def init(cls, config_name: str ):
        if cls._instance is None:
            inst = cls(config_name)
            cls._instance = inst
            cls._instantiated = cls

    @classmethod
    def instance(cls) -> "Configuration":
        return cls._instance

def config_files() -> Tuple[ModelConfig,TaskConfig]:
    root = fmbdir('model')
    params_file = cfg().task.params
    pfilepath = f"{root}/params/{params_file}.npz"
    with open(pfilepath, "rb") as f:
        ckpt = checkpoint.load(f, CheckPoint)
        model_config = ckpt.model_config
        task_config = ckpt.task_config
        print("Model description:\n", ckpt.description, "\n")
        print(f" >> model_config: {model_config}")
        print(f" >> task_config:  {task_config}")
    return model_config, task_config
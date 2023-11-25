from omegaconf import DictConfig, OmegaConf
import sys, hydra

def cfg() -> DictConfig:
    return Configuration.instance().cfg

def configure(config_name: str):
    Configuration.init( config_name )

class Configuration:
    _instance = None
    _instantiated = None

    def __init__(self, config_name: str ):
        self.config_name = config_name
        self.cfg: DictConfig = hydra.compose( self.config_name, return_hydra_config=True )

    @classmethod
    def init(cls, config_name: str ):
        if cls._instance is None:
            inst = cls( config_name )
            cls._instance = inst
            cls._instantiated = cls

    @classmethod
    def instance(cls) -> "Configuration":
        return cls._instance


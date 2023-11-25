from omegaconf import DictConfig, OmegaConf
import sys

def cfg() -> DictConfig:
    return Configuration.instance().cfg

def configure(config_name: str, config_path: str = None):
    Configuration.init( config_name, config_path )

class Configuration:
    _instance = None
    _instantiated = None

    def __init__(self, config_name: str, config_path: str ):
        self.config_name = config_name
        self.config_path = config_path
        self.cfg: DictConfig = self.get_config()

    def get_config(self) -> DictConfig:
        if 'hydra' in sys.modules:
            import hydra
            dcfg: DictConfig = hydra.compose( self.config_name, return_hydra_config=True )
        else:
            dcfg: DictConfig = OmegaConf.load(f"{self.config_path}/{self.config_name}.yaml")
        return dcfg

    @classmethod
    def init(cls, config_name: str, config_path: str ):
        if cls._instance is None:
            inst = cls( config_name, config_path )
            cls._instance = inst
            cls._instantiated = cls

    @classmethod
    def instance(cls) -> "Configuration":
        return cls._instance


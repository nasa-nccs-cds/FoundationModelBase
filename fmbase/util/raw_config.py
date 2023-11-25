from omegaconf import DictConfig, OmegaConf

def cfg() -> DictConfig:
    return Configuration.instance().cfg

def configure(config_name: str, config_path: str):
    Configuration.init( config_name, config_path )

class Configuration:
    _instance = None
    _instantiated = None

    def __init__(self, config_name: str, config_path: str ):
        self.cfg: DictConfig = OmegaConf.load( f"{config_path}/{config_name}.yaml" )

    @classmethod
    def init(cls, config_name: str ):
        if cls._instance is None:
            inst = cls(config_name)
            cls._instance = inst
            cls._instantiated = cls

    @classmethod
    def instance(cls) -> "Configuration":
        return cls._instance


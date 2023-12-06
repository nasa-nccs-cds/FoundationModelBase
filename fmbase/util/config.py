from omegaconf import DictConfig, OmegaConf
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Type, Optional, Union
from dataclasses import dataclass
import hydra

def cfg() -> DictConfig:
    return Configuration.instance().cfg

def configure(config_name: str):
    Configuration.init( config_name )

@dataclass(frozen=True, eq=True, order=True)
class Date:
	day: int
	month: int
	year: int

	@property
	def kw(self):
		return dict( day=self.day, month=self.month, year=self.year )

	@property
	def skw(self):
		return dict( year = str(self.year), month = f"{self.month+1:0>2}", day = f"{self.day+1:0>2}" )

	def __str__(self):
		return f'{self.year:04}{self.month:02}{self.day:02}'

	def __repr__(self):
		return f'{self.year}-{self.month}-{self.day}'

	@classmethod
	def get_dates( cls, **kwargs ) -> List["Date"]:
		years = kwargs.get( 'years', list( range(*cfg().preprocess.year_range) ) )
		months = kwargs.get( 'months', list(range(0, 12, 1)) )
		days = kwargs.get( 'days', list(range(0, 31, 1)) )
		return [ Date(day=day, month=month, year=year) for year in years for month in months for day in days ]
class ConfigBase(ABC):
    _instance = None
    _instantiated = None

    def __init__(self, config_name: str, **kwargs ):
        self.config_name = config_name
        self.cfg: DictConfig = self.get_parms(**kwargs)

    @abstractmethod
    def get_parms(self, **kwargs) -> DictConfig:
        return None

    @classmethod
    def init(cls, config_name: str ):
        if cls._instance is None:
            inst = cls( config_name )
            cls._instance = inst
            cls._instantiated = cls

    @classmethod
    def instance(cls) -> "Configuration":
        return cls._instance


class Configuration(ConfigBase):
    def get_parms(self, **kwargs) -> DictConfig:
        return hydra.compose(self.config_name, return_hydra_config=True)

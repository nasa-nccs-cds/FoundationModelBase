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
		return dict( year = self.syear, month = self.smonth , day = self.sday )

	@property
	def smonth(self):
		return f"{self.month + 1:0>2}"

	@property
	def sday(self):
		return f"{self.day + 1:0>2}"

	@property
	def syear(self):
		return str(self.year)

	def __str__(self):
		return self.syear + self.smonth + self.sday

	def __repr__(self):
		return f'{self.year}-{self.month+1}-{self.day+1}'

	@classmethod
	def get_dates( cls, **kwargs ) -> List["Date"]:
		years = kwargs.get( 'years', list( range(*cfg().preprocess.year_range) ) )
		months = kwargs.get( 'months', list(range(0, 12, 1)) )
		days = kwargs.get( 'days', list(range(0, 31, 1)) )
		return [ Date(day=day, month=month, year=year) for year in years for month in months for day in days ]

	@classmethod
	def days( cls, num_days: int )-> List["Date"]:
		year, month, day = cfg().model.year, cfg().model.month, cfg().model.day
		return [Date(year=year, month=month, day=day1) for day1 in range(day, day + num_days)]
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

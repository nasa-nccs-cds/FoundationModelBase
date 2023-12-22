import numpy as np
from typing import List, Optional, Dict, Type
import os, datetime
from fmbase.util.base import FMSingleton
from enum import Enum
from functools import wraps
from time import time
from datetime import datetime
import threading, time, logging, sys, traceback

def lgm() -> "LogManager":
    return LogManager.instance()

def exception_handled(func):
    def wrapper( *args, **kwargs ):
        try:
            return func( *args, **kwargs )
        except:
            lgm().exception( f" Error in {func}:" )
    return wrapper

def log_timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        try:
            ts = time.time()
            result = f(*args, **kw)
            te = time.time()
            lgm().log( f'EXEC {f.__name__} took: {te-ts:3.4f} sec' )
            return result
        except:
            lgm().exception( f" Error in {f}:" )
    return wrap

class LogManager(FMSingleton):

    def __init__(self):
        super(LogManager, self).__init__()
        self._wGui = None
        self._name = None
        self._lid = None
        self._level = logging.INFO
        self._log_stream = None
        self._keras_logger = None
        self.log_dir = None
        self.log_file  = None
        self.init_logging()

    def close(self):
        if self._log_stream  is not None:
            self._log_stream.flush()
            self._log_stream.close()
            self._log_stream = None

    @classmethod
    def pid(cls):
        return os.getpid()

    def set_level(self, level ):
        self._level = level

    def init_logging(self):
        from fmbase.util.config import cfg
        self.log_dir = f"{cfg().platform.cache}/logs"
        os.makedirs( self.log_dir, 0o777, exist_ok=True )
        overwrite = False
        self._lid = "" if overwrite else f"-{os.getpid()}"
        self.log_file = f'{self.log_dir}/main{self._lid}.log'
        self._log_stream = open(self.log_file, 'w')
        print( f"Opening log file:  '{self.log_file}'" )

    @property
    def ctime(self):
        return datetime.now().strftime("%H:%M:%S")

    def log( self,  msg, **kwargs ):
        if kwargs.get( 'print', False ): print( msg, flush=True )
        self._log_stream.write(f"[{self.ctime}] {msg}\n")
        self._log_stream.flush()

    def fatal(self, msg: str, status: int = 1 ):
        print( msg )
        self._log_stream.write(msg + "\n")
        self._log_stream.flush()
        sys.exit( status )

    def debug(self, msg, **kwargs ):
        if self._level == logging.DEBUG:
            self.log( msg,  **kwargs )

    def exception(self,  msg, **kwargs ):
        self._log_stream.write(f"\n{msg}\n{traceback.format_exc()}\n")
        self._log_stream.flush()

    def trace(self,  msg, **kwargs ):
        strace = "".join(traceback.format_stack())
        self._log_stream.write(f"\n{msg}\n{strace}\n")
        self._log_stream.flush()
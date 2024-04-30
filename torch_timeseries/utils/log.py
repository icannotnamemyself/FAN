#!/usr/bin/env python
import json
import logging
import logging.handlers
from decimal import Decimal
import math
import os
import threading
import time
import sys
from typing import Any, Dict, Optional




class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs)
        return cls._instances[cls]


class JupyterFormatter(logging.Formatter):
    def get_extra_fields(self, record):
        # The list contains all the attributes listed in
        # http://docs.python.org/library/logging.html#logrecord-attributes
        skip_list = (
            'args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
            'funcName', 'id', 'levelname', 'levelno', 'lineno', 'module',
            'msecs', 'msecs', 'message', 'msg', 'name', 'pathname', 'process',
            'processName', 'relativeCreated', 'thread', 'threadName', 'extra',
            'auth_token', 'password', 'stack_info')

        easy_types = (str, bool, dict, int, list, type(None))

        fields = {}

        for key, value in record.__dict__.items():
            if key not in skip_list:
                if isinstance(value, easy_types):
                    fields[key] = value
                elif isinstance(value, float):
                    fields[key] = None if math.isnan(value) else value
                else:
                    fields[key] = repr(value)

        return fields

    def format(self, record):
        s = super(JupyterFormatter, self).format(record)
        extras = json.dumps(self.get_extra_fields(record))
        if extras != '{}':
            s = f'{s} extras: {extras}'
        return s



class LoggerPool(metaclass=Singleton):
    def __init__(self):
        self.pool: Dict[str, logging.Logger] = {}

    def get(self, name: Optional[str], path=None, streaming=False) -> logging.Logger:
        if name is None:
            name = 'root'
        if name not in self.pool:
            host = '172.17.0.1'
            port = 5000
            logger = logging.getLogger(name)
            logger.setLevel(logging.DEBUG)

            formatter = JupyterFormatter(
                '%(asctime)s Jupyter [%(process)d]: %(message)s',
                '%b %d %H:%M:%S')
            formatter.converter = time.gmtime  # if you want UTC time
            if path is not None:
                file_handler = logging.handlers.RotatingFileHandler(
               os.path.join(path,f'{name}.log'), maxBytes=1024 * 1024 * 12, backupCount=5,mode='w+')
            else:
                file_handler = logging.handlers.RotatingFileHandler(
                f'./logs/{name}.log', maxBytes=1024 * 1024 * 12, backupCount=5,mode='w+')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)


            if streaming:
                consoleHandler = logging.StreamHandler()
                consoleHandler.setFormatter(formatter)
                logger.addHandler(consoleHandler)

            # file_handler.setFormatter(formatter)
            # logger.addHandler(file_handler)
            # logstash_handler = AsynchronousLogstashHandler(
            #     host, port, None)
            # logstash_handler.setFormatter(MyLogstashFormatter())
            #logger.addHandler(logstash_handler)
            self.pool[name] = logger
        return self.pool[name]


#LoggerPool().get('asyncio')
#LoggerPool().get('websockets')


class Log():
    def __init__(self, name: Optional[str] = None):
        self.logger = LoggerPool().get(name)
        self.name = name

    def debug(self, line: str, extra: Optional[Dict[Any, Any]] = None):
        self.logger.debug(line, extra=extra)

    def info(self, line: str, extra: Optional[Dict[Any, Any]] = None):
        self.logger.info(line, extra=extra)
    def _info(self, line: str, extra: Optional[Dict[Any, Any]] = None):
        self.logger.info(line, extra=extra)
    def error(self, line: str, extra: Optional[Dict[Any, Any]] = None):
        self.logger.error(line, extra=extra)

    def warn(self, line: str, extra: Optional[Dict[Any, Any]] = None):
        self.logger.warning(line, extra=extra)


if __name__ == '__main__':
    log = Log()
    log.debug('debug')
    log.info('info', extra={'foo': math.nan, 'bar': Decimal('1.000'), 'kar': Decimal(math.nan)})
    log.warn('warn')
    log.error('error')

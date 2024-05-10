from typing import Type, Union
from .DishTS import DishTS
from .RevIN import RevIN
from .SAN import SAN
from .No import No
from .CovidsV1 import CovidsV1
from .CovidsV2 import CovidsV2
from .CovidsV3 import CovidsV3
from .CovidsV4 import CovidsV4
from .SNv1 import SNv1
from .SNv2 import SNv2
from .SNv3 import SNv3
from .SNv4 import SNv4
from .PeriodV1 import PeriodV1
from .PeriodV2 import PeriodV2

from .PeriodFDV3 import PeriodFDV3
from .PeriodFDV4 import PeriodFDV4
from .PeriodFDV5 import PeriodFDV5
from .FAN import FAN
from .FANCP1 import FANCP1
from .FANNO import FANNO
from .FANRON import FANRON
from .FANGRU import FANGRU
from .FAN3MLP import FAN3MLP
from .FANMixer import FANMixer


__all__ = [
    'DishTS',
    'No',
    'RevIN',
    'SAN',
    'CovidsV1',
    'CovidsV2',
    'CovidsV3',
    'CovidsV4',
    'SNv1',
    'SNv2',
    'SNv3',
    'SNv4',
    'PeriodV1',
    'PeriodV2',
    'PeriodFDV3',
    'PeriodFDV4',
    'PeriodFDV5',
    'FAN',
    'FANCP1',
    'FANNO',
    'FANRON',
    'FANGRU',
    'FAN3MLP',
    'FANMixer',
]

# def _parse_type(str_or_type: Union[Type, str]) -> Type:
#     if isinstance(str_or_type, str):
#         return eval(str_or_type)
#     elif isinstance(str_or_type, type):
#         return str_or_type
#     else:
#         raise RuntimeError(f"{str_or_type} should be string or type")


# def norm_factory(N, T, O, config):
#     mode_type = _parse_type(config["type"])
    
    
#     if config["type"] == "DishTS":
#         return DishTS(N, T, **config)
    
#     elif config["type"] == "No":
#         return No()
#     else:
#         raise ValueError("Unsupported connection type")

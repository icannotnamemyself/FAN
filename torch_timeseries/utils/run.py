# Import the W&B Python Library and log into W&B
from dataclasses import dataclass
from typing import Type, Tuple, List
import wandb
from torch_timeseries.experiments.experiment import Experiment
from dataclasses import dataclass, asdict, field

from libtmux import Server



@dataclass
class Config:
    device: str
    horizons: List[int]
    # pred_lens : List[int] 
    
    datasets: List[Tuple[str, int]]
    batch_size : int = 128 # 32 default, recommended 64
    epochs: int = 100
    patience: int = 5
    
    model_paramaeters: dict = field(default_factory=lambda:{})


def run(exp_type: Type[Experiment], config: Config, project:str="", name:str=""):
    for dataset_type, windows in config.datasets:
        for horizon in config.horizons:
            exp = exp_type(
                epochs=config.epochs,
                patience=config.patience,
                windows=windows,
                horizon=horizon,
                batch_size=config.batch_size,
                dataset_type=dataset_type,
                device=config.device,
                **config.model_paramaeters
            )
            if project != "" and name != "":
                exp.config_wandb(project, name)
            exp.runs()
            wandb.finish()


def run_seed(exp_type: Type[Experiment], config: Config, seed:int):
    
    for dataset_type, windows in config.datasets:
        for horizon in config.horizons:
            exp = exp_type(
                epochs=config.epochs,
                patience=config.patience,
                windows=windows,
                horizon=horizon,
                batch_size=config.batch_size,
                dataset_type=dataset_type,
                device=config.device,
                **config.model_paramaeters
            )
            exp.run(seed)
            wandb.finish()






def print_id(exp_type: Type[Experiment], config: Config, seeds: List[int] = [42,233,666,19971203,19980224]):
    for dataset_type, windows in config.datasets:
        for horizon in config.horizons:
            exp = exp_type(
                epochs=config.epochs,
                patience=config.patience,
                windows=windows,
                horizon=horizon,
                dataset_type=dataset_type,
                device=config.device,
                **config.model_paramaeters
            )
            
            for seed in seeds:
                print(f"{seed} {exp._run_identifier(seed)}")





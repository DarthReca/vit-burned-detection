from typing import Union, Dict, Any, List

import torch
from importlib import import_module
import pandas as pd


def config_to_object(
    module_name: str, class_name: str, parameters: Union[Dict[str, Any], List[Any]]
) -> Any:
    """
    Create an instance of class `class_name` with parameters `parameters` imported from `module_name`.

    Parameters
    ----------
    module_name : str
        The name of the module in which the class is present.
    class_name: str
        The name of the class to instantiate.
    parameters:
        The arguments to pass to the constructor as a list or dictionary.

    Returns
    -------
    instance: Any
        The instance of `module_name`.`class_name`(`parameters`)
    """
    if "." in class_name:
        module_name, _, class_name = class_name.rpartition(".")
    mod = import_module(module_name)
    tr = getattr(mod, class_name)
    if isinstance(parameters, list):
        instance = tr(*parameters)
    elif isinstance(parameters, dict):
        try:
            instance = tr(**parameters)
        except TypeError:
            parameters = {
                k: torch.tensor(v) if isinstance(v, (int, float)) else v
                for k, v in parameters.items()
            }
            instance = tr(**parameters)
    else:
        raise Exception("Only dict and list are supported")
    return instance


def trainer_converter(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply standard conversion for trainer configuration."""
    if "callbacks" in config:
        config["callbacks"] = [
            config_to_object("pytorch_lightning.callbacks", k, v)
            for k, v in config["callbacks"].items()
        ]
    return config


def create_satellite_groups(config: Dict[str, Any]) -> Dict[str, Any]:
    if "fold_separation_csv" not in config:
        return config
    config["groups"] = read_groups(config["fold_separation_csv"])
    return config


def read_groups(satellite_folds: str) -> Dict[str, List[str]]:
    """
    Read folds (i.e., colors) - for each fold get the corresponding input folders of Sentinel-2 dataset.
    Copyright Simone Monaco. Copy from https://github.com/dbdmg/rescue.

    Returns
    -------
    dictionary
        key = fold color, value = list of dataset folders in this fold
    """
    groups = {}
    df = pd.read_csv(satellite_folds)
    for key, grp in df.groupby("fold"):
        folder_list = grp["folder"].tolist()
        groups[key] = folder_list
    return groups

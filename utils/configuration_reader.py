# Copyright 2022 Daniele Rege Cambrin

from typing import Dict, Any, Callable
import yaml


class ConfigurationParser:
    def __init__(self, config_file: str):
        with open(config_file, "r") as f:
            all_configs = yaml.safe_load(f)

        print("Loading configuration " + all_configs.pop("name"))
        self.configs = {}
        for k, v in all_configs.items():
            with open(v, "r") as f:
                self.configs[k] = yaml.safe_load(f)

    def get_configuration(self, name: str) -> Dict[str, Any]:
        assert name in self.configs
        return self.configs[name]

    def get_config_and_apply(
        self, name: str, func: Callable[[Dict[str, Any]], Dict[str, Any]]
    ) -> Dict[str, Any]:
        return func(self.get_configuration(name))

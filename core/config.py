import yaml
from functools import cache

class Config:
    DEFAULT_CONFIG_FILE = "default.yml"

    def __init__(self, config_file="config.yml"):
        self.cfg = self.load_config(config_file)
        self.default = self.load_config(self.DEFAULT_CONFIG_FILE)

    def load_config(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            return config

    def check(self):
        print(self.cfg)
        print(self.default)

    def get(self, key, default=None):
        keys = key.split(".")
        value = self.cfg

        # Try to get the value from the config file
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                value = None
                break

        # If the value is None, try to get the value from the default config file
        if value is None:
            value = self.default
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default

        return value
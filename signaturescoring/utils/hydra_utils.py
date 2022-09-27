"""For some scenarios with very simple configuration,
hydra needs somewhat verbose code.
We wrap this code into two decorators.
Use case:
@hy.config
@dataclasses.dataclass
class ConnConfig:
    host: str = "some-website.com"
    port: int = 1234
@hy.main
def connect(config: ConnConfig):
    request = some_function(config.host, config.port)
    if request.status == something:
        do_something
    etc.
"""
import hydra
import omegaconf
from hydra.core.config_store import ConfigStore

_CS = ConfigStore.instance()
_MAIN_CONFIG: str = "MainConfig"


MISSING = omegaconf.MISSING


def config(config_type):
    _CS.store(name=_MAIN_CONFIG, node=config_type)
    return config_type


def main(fn):
    return hydra.main(config_path=None, config_name=_MAIN_CONFIG)(fn)

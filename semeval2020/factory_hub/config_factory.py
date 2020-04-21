from typing import List, Dict
import copy


__dispatcher: Dict[str, Dict] = {}


def get_config(method_name: str) -> Dict:
    return copy.deepcopy(__dispatcher[method_name])


def get_selection() -> List[str]:
    return list(__dispatcher.keys())


def register(method_name: str, config: Dict):
    if method_name in __dispatcher:
        raise ValueError(f'Config with the name {method_name} already exists.')
    __dispatcher[method_name] = config

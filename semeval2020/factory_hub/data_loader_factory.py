from abc import ABCMeta
from typing import List, Dict

from semeval2020.factory_hub import abstract_data_loader


__dispatcher: Dict[str, ABCMeta] = {}


def create_data_loader(method_name: str, **kwargs) -> abstract_data_loader.AbstractDataLoader:
    return __dispatcher[method_name](**kwargs)


def get_selection() -> List[str]:
    return list(__dispatcher.keys())


def register(method_name: str, data_loader_class: ABCMeta):
    if method_name in __dispatcher:
        raise ValueError(f'{str(type(data_loader_class))} with this name already exists.')
    __dispatcher[method_name] = data_loader_class

from abc import ABCMeta
from typing import List, Dict

from semeval2020.factory_hub import abstract_model


__dispatcher: Dict[str, ABCMeta] = {}


def create_model(method_name: str, **kwargs) -> abstract_model.AbstractModel:
    return __dispatcher[method_name](**kwargs)


def get_selection() -> List[str]:
    return list(__dispatcher.keys())


def register(method_name: str, model_class: ABCMeta):
    if method_name in __dispatcher:
        raise ValueError(f'{str(type(model_class))} with this name already exists.')
    __dispatcher[method_name] = model_class

from abc import ABCMeta
from typing import List, Dict

from semeval2020.factory_hub import abstract_preprocessor


__dispatcher: Dict[str, ABCMeta] = {}


def create_preprocessor(method_name: str, **kwargs) -> abstract_preprocessor.AbstractPreprocessor:
    return __dispatcher[method_name](**kwargs)


def get_selection() -> List[str]:
    return list(__dispatcher.keys())


def register(method_name: str, preprocessor_class: ABCMeta):
    if method_name in __dispatcher:
        raise ValueError(f'{str(type(preprocessor_class))} with this name already exists.')
    __dispatcher[method_name] = preprocessor_class

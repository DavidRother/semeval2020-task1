from abc import ABCMeta
from typing import List, Dict

from semeval2020.factory_hub import abstract_evaluation_suite


__dispatcher: Dict[str, ABCMeta] = {}


def create_evaluation_suite(method_name: str, **kwargs) -> abstract_evaluation_suite.AbstractEvaluationSuite:
    return __dispatcher[method_name](**kwargs)


def get_selection() -> List[str]:
    return list(__dispatcher.keys())


def register(method_name: str, evaluation_class: ABCMeta):
    if method_name in __dispatcher:
        raise ValueError(f'{str(type(evaluation_class))} with this name already exists.')
    __dispatcher[method_name] = evaluation_class

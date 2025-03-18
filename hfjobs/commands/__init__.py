from abc import ABC, abstractmethod
from argparse import _SubParsersAction


class BaseCommand(ABC):
    @staticmethod
    @abstractmethod
    def register_subcommand(parser: _SubParsersAction) -> None:
        raise NotImplementedError()

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError()

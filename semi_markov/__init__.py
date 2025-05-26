import importlib.metadata

__version__: str = importlib.metadata.version("semi_markov")

__all__ = ["__version__"]
del importlib
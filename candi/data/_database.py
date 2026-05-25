from typing import Any


class CancerDataNamespace:
    """Reusable namespace for attribute-style dataset access under `.data`."""

    __slots__ = ("_parent",)

    def __init__(self, parent: Any) -> None:
        object.__setattr__(self, "_parent", parent)

    def __getattr__(self, name: str) -> Any:
        if name in self._parent._datasets:
            return self._parent._datasets[name]
        if name in self._parent._paths:
            raise AttributeError(
                f"Dataset '{name}' is available but not loaded. Call `.load('{name}')` first."
            )
        raise AttributeError(f"No dataset named '{name}' defined.")

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_parent":
            object.__setattr__(self, name, value)
            return
        self.add(name=name, dataset=value)

    def __dir__(self):
        return sorted(
            set(super().__dir__())
            | set(self._parent._paths)
            | set(self._parent._datasets)
        )

    def add(self, name: str, dataset: Any, overwrite: bool = False) -> None:
        """Add a dataset to this namespace."""
        if not name or not isinstance(name, str):
            raise ValueError("Dataset name must be a non-empty string.")
        if not name.isidentifier():
            raise ValueError(
                f"Dataset name '{name}' is not a valid Python identifier for attribute access."
            )
        if name in object.__dir__(self):
            raise ValueError(
                f"Dataset name '{name}' conflicts with an existing namespace attribute."
            )
        if name in self._parent._datasets and not overwrite:
            raise ValueError(
                f"Dataset '{name}' is already loaded. Pass overwrite=True to replace it."
            )
        if name in self._parent._paths and not overwrite:
            raise ValueError(
                f"Dataset '{name}' is already defined as an available built-in dataset. "
                f"Pass overwrite=True to replace it."
            )
        self._parent._datasets[name] = dataset

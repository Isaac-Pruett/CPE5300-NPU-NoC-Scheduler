"""Discover trusted, local scheduler implementations without a central registry."""

import importlib
import inspect
import pkgutil
import re

from schedulers.scheduler import Scheduler


def discover_schedulers(package_name: str = "schedulers") -> dict[str, type[Scheduler]]:
    """Load concrete policies from public modules directly inside a package.

    Discovery imports Python modules: only trusted project code belongs here.
    Class-local opt-outs and names deliberately do not propagate to subclasses.
    No policy is instantiated during discovery.
    """
    try:
        package = importlib.import_module(package_name)
    except Exception as exc:
        raise ValueError(
            f"cannot import scheduler package {package_name}: {exc}"
        ) from exc
    policies: dict[str, type[Scheduler]] = {}
    modules = sorted(pkgutil.iter_modules(package.__path__), key=lambda item: item.name)
    for info in modules:
        if info.ispkg or info.name.startswith("_"):
            continue
        module_name = f"{package_name}.{info.name}"
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            raise ValueError(
                f"cannot import scheduler module {module_name}: {exc}"
            ) from exc
        # A class may have multiple aliases in its defining module.
        seen = set()
        for _, candidate in inspect.getmembers(module, inspect.isclass):
            if candidate in seen:
                continue
            seen.add(candidate)
            if (
                candidate is Scheduler
                or candidate.__module__ != module_name
                or not issubclass(candidate, Scheduler)
                or inspect.isabstract(candidate)
                or not candidate.__dict__.get("discoverable", True)
            ):
                continue
            name = candidate.__dict__.get("policy_name", info.name.replace("_", "-"))
            if not isinstance(name, str) or not re.fullmatch(
                r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", name
            ):
                raise ValueError(
                    f"{module_name}.{candidate.__name__}: invalid policy name {name!r}"
                )
            if name == "none":
                raise ValueError(
                    f"{module_name}.{candidate.__name__}: policy name 'none' is reserved"
                )
            if name in policies:
                previous = policies[name]
                raise ValueError(
                    f"duplicate policy name {name!r}: {previous.__module__}.{previous.__name__} "
                    f"and {module_name}.{candidate.__name__}; set a unique policy_name"
                )
            try:
                inspect.signature(candidate).bind()
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{module_name}.{candidate.__name__}: scheduler constructor must accept no arguments"
                ) from exc
            policies[name] = candidate
    return dict(sorted(policies.items()))

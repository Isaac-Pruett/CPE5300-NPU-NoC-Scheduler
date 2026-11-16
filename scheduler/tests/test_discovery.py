"""Scheduler plugin discovery and simulator-free policy listing."""

import importlib
import inspect
import io
import re
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from schedulers.discovery import discover_schedulers
from schedulers.fifo import FIFO
from schedulers.ljf import LongestJobFirst
from schedulers.prio_heap_on_fifo import PrioHeapOnFIFO
from schedulers.scheduler import Scheduler
from schedulers.sjf import ShortestJobFirst


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.package_name = "discovery_test_plugins"
        self.package = self.root / self.package_name
        self.package.mkdir()
        (self.package / "__init__.py").write_text("")
        self.path_patch = patch.object(sys, "path", [str(self.root), *sys.path])
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)
        self.addCleanup(self.clear_modules)
        importlib.invalidate_caches()

    def clear_modules(self):
        for name in list(sys.modules):
            if name == self.package_name or name.startswith(self.package_name + "."):
                del sys.modules[name]
        importlib.invalidate_caches()

    def module(self, name, source):
        path = self.package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source))
        cache = path.parent / "__pycache__"
        if cache.exists():
            for bytecode in cache.glob(f"{path.stem}.*.pyc"):
                bytecode.unlink()
        importlib.invalidate_caches()

    def policy(self, filename, body="", class_name="Policy"):
        self.module(
            filename,
            "from schedulers.scheduler import Scheduler\n\n"
            f"class {class_name}(Scheduler):\n"
            "    def next(self):\n"
            "        return self.queue.pop(0)\n"
            + textwrap.indent(textwrap.dedent(body).strip(), "    ")
            + "\n",
        )

    def discover(self):
        return discover_schedulers(self.package_name)

    def test_builtin_policies_are_discovered_and_helper_is_excluded(self):
        policies = discover_schedulers()
        for name, policy in {
            "fifo": FIFO,
            "sjf": ShortestJobFirst,
            "ljf": LongestJobFirst,
        }.items():
            self.assertIs(policies[name], policy)
        self.assertNotIn(PrioHeapOnFIFO, policies.values())
        self.assertIs(PrioHeapOnFIFO.__dict__.get("discoverable"), False)

    def test_scheduler_is_an_abstract_base_class(self):
        self.assertTrue(inspect.isabstract(Scheduler))
        with self.assertRaises(TypeError):
            Scheduler()

    def test_public_modules_are_discovered_in_deterministic_order(self):
        for name in ("z_last", "a_first", "middle"):
            self.policy(name + ".py")
        expected = ["a-first", "middle", "z-last"]
        policies = self.discover()
        self.assertEqual(list(policies), expected)
        self.assertEqual(list(self.discover()), expected)
        for name, policy in policies.items():
            self.assertEqual(
                policy.__module__,
                f"{self.package_name}.{name.replace('-', '_')}",
            )
            self.assertIsInstance(policy(), Scheduler)

    def test_imported_classes_and_aliases_are_not_registered(self):
        self.policy("original.py", class_name="Original")
        self.module(
            "aliases.py",
            """
            from schedulers.scheduler import Scheduler
            from schedulers.fifo import FIFO as ImportedFIFO
            from .original import Original
            AnotherAlias = Original
            """,
        )
        policies = self.discover()
        original = importlib.import_module(f"{self.package_name}.original").Original
        self.assertEqual(policies, {"original": original})

    def test_private_modules_and_subpackages_are_not_imported(self):
        self.policy("visible.py")
        self.module("_private.py", "raise RuntimeError('private module imported')")
        self.module("nested/__init__.py", "raise RuntimeError('package imported')")
        self.policy("nested/hidden.py")
        self.assertEqual(set(self.discover()), {"visible"})
        self.assertNotIn(f"{self.package_name}._private", sys.modules)
        self.assertNotIn(f"{self.package_name}.nested", sys.modules)

    def test_abstract_classes_base_classes_and_unrelated_classes_are_skipped(self):
        self.module(
            "abstract.py",
            """
            from abc import abstractmethod
            from schedulers.scheduler import Scheduler

            class AbstractPolicy(Scheduler):
                @abstractmethod
                def next(self):
                    pass

            class Unrelated:
                pass
            """,
        )
        self.assertEqual(self.discover(), {})

    def test_opt_out_is_class_local_and_child_does_not_inherit_policy_name(self):
        self.module(
            "helper.py",
            """
            from schedulers.scheduler import Scheduler

            class Helper(Scheduler):
                discoverable = False
                policy_name = 'helper-policy'

                def next(self):
                    return self.queue.pop(0)
            """,
        )
        self.module(
            "child_policy.py",
            """
            from .helper import Helper

            class Child(Helper):
                pass
            """,
        )
        policies = self.discover()
        child = importlib.import_module(f"{self.package_name}.child_policy").Child
        self.assertEqual(policies, {"child-policy": child})

    def test_class_local_policy_name_overrides_module_basename(self):
        self.policy("some_module.py", "policy_name = 'custom-policy2'")
        self.assertEqual(set(self.discover()), {"custom-policy2"})

    def test_duplicate_names_report_name_and_both_conflicting_modules(self):
        self.policy("first.py", "policy_name = 'collision'", "First")
        self.policy("second.py", "policy_name = 'collision'", "Second")
        with self.assertRaises(ValueError) as caught:
            self.discover()
        message = str(caught.exception)
        for detail in ("collision", "first", "second"):
            self.assertIn(detail, message)

    def test_two_classes_in_one_module_cannot_share_the_default_name(self):
        self.module(
            "collision.py",
            """
            from schedulers.scheduler import Scheduler

            class First(Scheduler):
                def next(self):
                    return self.queue.pop(0)

            class Second(Scheduler):
                def next(self):
                    return self.queue.pop(0)
            """,
        )
        with self.assertRaisesRegex(ValueError, "collision"):
            self.discover()

    def test_invalid_and_reserved_policy_names_are_rejected(self):
        for name in (
            "",
            "Upper",
            "two_words",
            "two words",
            "-leading",
            "trailing-",
            "double--dash",
            "1first",
            "punctuation!",
            "nonascii-é",
            "line\n",
            "none",
        ):
            with self.subTest(name=name):
                self.clear_modules()
                self.policy("invalid.py", f"policy_name = {name!r}")

                with self.assertRaises(ValueError):
                    self.discover()

    def test_required_constructor_arguments_are_rejected(self):
        for signature in ("self, required", "self, *, required", "self, required, /"):
            with self.subTest(signature=signature):
                self.clear_modules()
                self.policy(
                    "needs_argument.py",
                    f"def __init__({signature}):\n    super().__init__()",
                )
                with self.assertRaisesRegex(ValueError, "needs_argument|Policy"):
                    self.discover()

    def test_optional_and_variadic_constructors_are_accepted_without_instantiation(
        self,
    ):
        self.policy(
            "optional.py",
            """
            def __init__(self, value=1, *args, **kwargs):
                raise AssertionError('discovery must inspect, not instantiate')
            """,
        )
        self.assertEqual(set(self.discover()), {"optional"})

    def test_module_import_errors_are_wrapped_with_module_name(self):
        self.module("broken.py", "raise RuntimeError('broken plugin import')")
        with self.assertRaisesRegex(
            ValueError, re.escape(f"{self.package_name}.broken")
        ):
            self.discover()


class PolicyListingTests(unittest.TestCase):
    def test_listing_policies_does_not_need_a_simulator(self):
        from astra.run import main

        output = io.StringIO()
        with (
            patch.object(sys, "argv", ["astra.run", "--list-policies"]),
            patch(
                "astra.run.shutil.which",
                side_effect=AssertionError(
                    "policy listing must not locate a simulator"
                ),
            ) as which,
            redirect_stdout(output),
        ):
            main()
        which.assert_not_called()
        listing = output.getvalue()
        for name in ("fifo", "sjf", "ljf"):
            self.assertRegex(listing, rf"\b{name}\b")
        self.assertNotIn("prio-heap-on-fifo", listing)


if __name__ == "__main__":
    unittest.main()

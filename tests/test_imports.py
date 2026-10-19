"""Verify that the project's required third-party packages are available."""

import importlib
import unittest


class TestImports(unittest.TestCase):
    """Tests for required project dependencies."""

    def test_required_packages_import(self):
        """Import each required package and report missing dependencies clearly."""
        required_packages = (
            "numpy",
            "pandas",
            "conllu",
            "sklearn",
            "matplotlib",
        )

        for package_name in required_packages:
            with self.subTest(package=package_name):
                importlib.import_module(package_name)


if __name__ == "__main__":
    unittest.main()
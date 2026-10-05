import importlib
import unittest


class TestImports(unittest.TestCase):
    def test_required_packages_import(self):
        packages = ("numpy", "pandas", "conllu", "sklearn","matplotlib")

        for package in packages:
            with self.subTest(package=package):
                importlib.import_module(package)


if __name__ == "__main__":
    unittest.main()
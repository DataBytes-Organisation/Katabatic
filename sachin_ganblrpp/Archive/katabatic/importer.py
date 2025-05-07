# Usage
# -----
# from aiko_services.utilities import *
# module_descriptor = "pathname/filename.py"  # or "package.module"
# module = load_module(module_descriptor)
# module.some_class()
# module.some_function()
#
# To Do
# -----
# - None, yet !

import importlib
import os
import sys

__all__ = ["load_module", "load_modules"]


# If the environment variable 'AIKO_IMPORTER_USE_CURRENT_DIRECTORY' is set, add
# the current working directory to the system path to allow module imports from it.
if os.environ.get("AIKO_IMPORTER_USE_CURRENT_DIRECTORY"):
    sys.path.append(os.getcwd())

# Cache for storing loaded modules to prevent reloading the same module multiple times.
MODULES_LOADED = {}


def load_module(module_descriptor):
    """
    Load a Python module from a given descriptor.

    Arguments:
        module_descriptor (str): The path to the module (e.g., "directory/file.py") 
        or the module name (e.g., "package.module").

    Returns:
        module: The loaded module object.

    Example:
        module = load_module("example_module.py")
        module.some_function()

    If the module has already been loaded, it is returned from the cache. Otherwise, 
    it is loaded from the file system or from installed packages.
    """
    if module_descriptor in MODULES_LOADED:
        module = MODULES_LOADED[module_descriptor]
    else:
        if module_descriptor.endswith(".py"):
            # Load module from Python source pathname, e.g "directory/file.py"
            module_pathname = module_descriptor
            module = importlib.machinery.SourceFileLoader(
                "module", module_pathname
            ).load_module()
        else:
            print(module_descriptor)
            # Load module from "installed" modules, e.g "package.module"
            module_name = module_descriptor
            module = importlib.import_module(module_name)
            print(module)
        MODULES_LOADED[module_descriptor] = module
    return module


def load_modules(module_pathnames):
    """
    Load multiple Python modules from a list of descriptors.

    Arguments:
        module_pathnames (list of str): A list of paths or module names to load.

    Returns:
        list: A list of loaded module objects.

    Example:
        modules = load_modules(["module1.py", "module2.py"])
        modules[0].some_function()

    Each module in the list is loaded and appended to the return list. 
    If a module descriptor is empty, None is appended in its place.
    """
    modules = []
    for module_pathname in module_pathnames:
        if module_pathname:
            modules.append(load_module(module_pathname))
        else:
            modules.append(None)
    return modules


__path__ = __import__('pkgutil').extend_path(__path__, __name__)

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

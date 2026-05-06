from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("avlite")
except PackageNotFoundError:
    __version__ = "unknown"

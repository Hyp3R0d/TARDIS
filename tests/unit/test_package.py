from importlib.metadata import version

import tardis


def test_package_version_matches_metadata() -> None:
    assert tardis.__version__ == version("tardis-video")

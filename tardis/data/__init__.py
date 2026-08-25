"""Bounded-memory dataset access for local archives and remote diagnostics."""

from tardis.data.assembly import RemoteDataLoaders, build_remote_dataloaders
from tardis.data.contracts import RemoteObject, VideoRecord
from tardis.data.http_range import RangeClient

__all__ = [
    "RangeClient",
    "RemoteDataLoaders",
    "RemoteObject",
    "VideoRecord",
    "build_remote_dataloaders",
]

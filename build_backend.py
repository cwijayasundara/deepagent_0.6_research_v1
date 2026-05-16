from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from typing import Any

from flit_core import buildapi


def build_wheel(wheel_directory: str, config_settings: dict[str, Any] | None = None, metadata_directory: str | None = None) -> str:
    return buildapi.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory: str, config_settings: dict[str, Any] | None = None) -> str:
    return buildapi.build_sdist(sdist_directory, config_settings)


def get_requires_for_build_wheel(config_settings: dict[str, Any] | None = None) -> list[str]:
    return buildapi.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_editable(config_settings: dict[str, Any] | None = None) -> list[str]:
    return buildapi.get_requires_for_build_editable(config_settings)


def prepare_metadata_for_build_wheel(metadata_directory: str, config_settings: dict[str, Any] | None = None) -> str:
    return buildapi.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def prepare_metadata_for_build_editable(metadata_directory: str, config_settings: dict[str, Any] | None = None) -> str:
    return buildapi.prepare_metadata_for_build_editable(metadata_directory, config_settings)


def build_editable(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        wheel_name = buildapi.build_editable(tmpdir, config_settings, metadata_directory)
        source = Path(tmpdir) / wheel_name
        target = Path(wheel_directory) / wheel_name
        with zipfile.ZipFile(source, "r") as src, zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as dst:
            for item in src.infolist():
                data = src.read(item.filename)
                if item.filename.endswith(".pth"):
                    if not data.endswith(b"\n"):
                        data += b"\n"
                    item.external_attr = 0o644 << 16
                    item.create_system = 3
                dst.writestr(item, data)
    return wheel_name

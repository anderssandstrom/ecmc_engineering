"""Create the files for a new IOC project without overwriting existing work."""
import json
from pathlib import Path
import re


def create_ioc_project(parent, name):
    name = name.strip()
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]*", name):
        raise ValueError("Use an IOC name containing letters, digits, underscores or hyphens.")
    directory = Path(parent).expanduser().resolve() / name
    # Fail for an existing directory, including an existing empty project.
    directory.mkdir()
    created = []
    startup = directory / (name + "_startup.cmd")
    parameters = directory / (name + "_parameters.yaml")
    content = (
        "ioc_name: {}\n"
        'ioc_host: ""\n'
        "ioc_port: 50001\n"
        "ioc_user: root\n"
        "epics_version: 7.0.10\n"
        "os: deb12\n"
        "cpu_architecture: x86_64\n"
    ).format(json.dumps(name))
    try:
        for path, text in [(startup, ""), (parameters, content)]:
            with path.open("x", encoding="utf-8") as stream:
                created.append(path)
                stream.write(text)
    except OSError:
        for path in reversed(created):
            path.unlink()
        directory.rmdir()
        raise
    return startup


def parameters_path(startup):
    suffix = "_startup.cmd"
    if startup.name.endswith(suffix):
        return startup.with_name(startup.name[:-len(suffix)] + "_parameters.yaml")
    return None

import os
import shlex
from pathlib import Path


def get_julia_project_dir():
    """
    Return the repository-local Julia project directory when available.

    For an editable/source checkout this resolves to <repo>/julia-env.
    Installed packages may not contain that directory, in which case no
    project argument is injected.
    """
    package_dir = Path(__file__).resolve().parent
    project_dir = package_dir.parent / "julia-env"

    if (project_dir / "Project.toml").is_file():
        return str(project_dir)

    return None


def julia_command(interpreter, script, *args):
    """
    Build a Julia command as an argv list.

    The configured interpreter is preserved, including optional command-line
    arguments.  When the repository-local Julia environment exists,
    --project=<repo>/julia-env is added unless the interpreter already
    specifies a Julia project.
    """
    command = shlex.split(interpreter)

    if not command:
        raise ValueError("Julia interpreter is empty")

    has_project = any(
        arg == "--project" or arg.startswith("--project=")
        for arg in command[1:]
    )

    project_dir = get_julia_project_dir()
    if project_dir is not None and not has_project:
        command.append(f"--project={project_dir}")

    command.append(os.fspath(script))
    command.extend(os.fspath(arg) for arg in args)

    return command

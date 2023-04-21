from pathlib import Path


def get_project_dir():
    """Returns the path to the project directory.

    Returns:
        project_dir (pathlib.Path): path to the project directory
    """
    return Path(__file__).parent.parent.parent


def get_data_dir():
    """Returns the path to the data directory.

    Returns:
        data_dir (pathlib.Path): path to the data directory
    """
    return get_project_dir() / "data"


def get_models_dir():
    """Returns the path to the models directory.

    Returns:
        models_dir (pathlib.Path): path to the models directory
    """
    return get_project_dir() / "models"

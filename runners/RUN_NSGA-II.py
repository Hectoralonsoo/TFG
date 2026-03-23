from runpy import run_path
from src.tfg.paths import REPO_ROOT


_MODULE_PATH = REPO_ROOT / "src" / "tfg" / "runners" / "RUN_NSGA-II.py"


if __name__ == "__main__":
    run_path(str(_MODULE_PATH), run_name="__main__")

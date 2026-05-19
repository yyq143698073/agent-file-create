import logging
from pathlib import Path


def setup_logging() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    log_dir = base_dir / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    log_path = log_dir / "app.log"
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # file: INFO for all modules
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(fmt))
    root.addHandler(fh)

    # console: WARNING+ for third-party libs
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter(fmt))
    root.addHandler(ch)

    # our modules: INFO on console as well
    for name in ("agent_file_create", "__main__"):
        lg = logging.getLogger(name)
        lg.propagate = False
        lg.addHandler(fh)
        ch2 = logging.StreamHandler()
        ch2.setLevel(logging.INFO)
        ch2.setFormatter(logging.Formatter(fmt))
        lg.addHandler(ch2)

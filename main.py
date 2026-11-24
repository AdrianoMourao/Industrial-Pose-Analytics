"""Application entry point."""

from gui.main_window import MainWindow
from utils.logger import configure_logging


def main() -> None:
    """Start the desktop application."""
    configure_logging()
    app = MainWindow()
    app.mainloop()


if __name__ == "__main__":
    main()

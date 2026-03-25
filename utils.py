import sqlite3, os
from pathlib import Path
from sqlite3 import Cursor, Connection


class SqliteManager:
    def __init__(self, db_path: str | Path, commit_on_exit: bool = True) -> None:
        self.db_path = db_path
        self.commit_on_exit = commit_on_exit

    def __enter__(self) -> tuple[Cursor, Connection]:
        self.connection = sqlite3.connect(self.db_path)
        self.db = self.connection.cursor()
        return self.db, self.connection

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.db.close()
        if self.commit_on_exit:
            self.connection.commit()
        self.connection.close()


class ProgressBarElements:
    PROGRESS_RATIO = "{progress_ratio}"
    PROGRESS_BAR = "{progress_bar}"
    PROGRESS_PERCENTAGE = "{progress_percentage}"


class ProgressBar:
    _DEFAULT_LAYOUT = (
        ProgressBarElements.PROGRESS_RATIO,
        " |",
        ProgressBarElements.PROGRESS_BAR,
        "| ",
        ProgressBarElements.PROGRESS_PERCENTAGE,
    )

    def __init__(
        self,
        total: int,
        start_at: int = 1,
        update_every: int = 1,
        decimals: int = 1,
        length: int = 50,
        void: str = " ",
        fill: str = "█",
        print_end: str = "\r",
        layout: list[str] = None,
    ) -> None:
        if update_every < 1:
            raise ValueError("update_every must be at least 1")

        self.start_at = start_at
        self.iteration = start_at
        self.total = total
        self.update_every = update_every
        self.decimals = decimals
        self.length = length
        self.void = void
        self.fill = fill
        self.print_end = print_end
        self._finished = False
        self.progress_bar_length = 0

        self.layout = layout or list(self._DEFAULT_LAYOUT)

    def start(self):
        self.update()

    def update(self):
        if self.iteration > self.total:
            if not self._finished:
                self.finish()
            self._finished = True
            return

        try:
            self.percent = f"{self.iteration / self.total * 100: .{self.decimals}f}"
        except ZeroDivisionError:
            raise ValueError("Cannot have total = 0")

        filled_length = int(self.length * self.iteration // self.total)

        bar = self.fill * filled_length + self.void * (self.length - filled_length)

        full_bar = "".join(self.layout).format_map(
            {
                "progress_ratio": f"{self.iteration}/{self.total}",
                "progress_bar": bar,
                "progress_percentage": f"{self.percent}%",
            }
        )

        progress_bar = f"{full_bar: <{os.get_terminal_size().columns}}"

        # This is necessary because all numbers are not the same length every time
        # But I use os.get_terminal_size().columns instead which deletes the whole line
        # So
        # self.progress_bar_length = len(progress_bar)

        print(f"\r{progress_bar}", end=self.print_end)

    def increment(self):
        self.iteration += 1
        if self.iteration > self.total or (self.iteration - self.start_at) % self.update_every == 0:
            self.update()

    def clear_line(self):
        # print("\r" + " " * self.progress_bar_length, end="\r")
        print("\r" + " " * os.get_terminal_size().columns, end="\r")

    def finish(self):
        if self._finished:
            return

        if self.iteration <= self.total:
            self.update()

        self._finished = True
        print()

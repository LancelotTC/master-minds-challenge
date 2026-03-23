import sqlite3
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

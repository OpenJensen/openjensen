"""Internal async persistence. Public operations are exposed by owning modules."""

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import JSON, Column, MetaData, String, Table
from sqlalchemy.engine import URL, Connection
from sqlalchemy.ext.asyncio import create_async_engine

metadata = MetaData()
projects = Table(
    "projects",
    metadata,
    Column("id", String, primary_key=True),
    Column("name", String, nullable=False),
    Column("created_at", String, nullable=False),
)
jobs = Table(
    "jobs",
    metadata,
    Column("id", String, primary_key=True),
    Column("project_id", String, nullable=False, index=True),
    Column("status", String, nullable=False),
    Column("record", JSON, nullable=False),
)


def upgrade(connection: Connection) -> None:
    config = Config()
    config.set_main_option("script_location", str(Path(__file__).parent / "migrations"))
    config.attributes["connection"] = connection
    command.upgrade(config, "head")


class Storage:
    def __init__(self, directory: Path):
        self.engine = create_async_engine(
            URL.create("sqlite+aiosqlite", database=str(directory / "workspace.sqlite3"))
        )

    async def initialize(self) -> None:
        async with self.engine.begin() as connection:
            await connection.run_sync(upgrade)

    async def close(self) -> None:
        await self.engine.dispose()

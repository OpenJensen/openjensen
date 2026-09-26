from uuid import uuid4

from sqlalchemy import insert, select

from vla_platform.contracts import Project, now
from vla_platform.storage import Storage, projects


class Projects:
    def __init__(self, storage: Storage):
        self.storage = storage

    async def create(self, name: str) -> Project:
        project = Project(id=str(uuid4()), name=name, created_at=now())
        async with self.storage.engine.begin() as connection:
            await connection.execute(insert(projects).values(**project.model_dump()))
        return project

    async def list(self) -> list[Project]:
        async with self.storage.engine.connect() as connection:
            rows = await connection.execute(select(projects).order_by(projects.c.created_at))
            return [Project.model_validate(dict(row)) for row in rows.mappings()]

    async def get(self, project_id: str) -> Project | None:
        async with self.storage.engine.connect() as connection:
            row = (
                (await connection.execute(select(projects).where(projects.c.id == project_id)))
                .mappings()
                .first()
            )
            return Project.model_validate(dict(row)) if row else None

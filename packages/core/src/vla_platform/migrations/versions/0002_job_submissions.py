"""Atomically bind explicit submission keys to their original accepted jobs."""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"


def upgrade() -> None:
    op.create_table(
        "job_submissions",
        sa.Column("project_id", sa.String(), primary_key=True),
        sa.Column("operation", sa.String(), primary_key=True),
        sa.Column("idempotency_key", sa.String(), primary_key=True),
        sa.Column("fingerprint_version", sa.Integer(), nullable=False),
        sa.Column("request_sha256", sa.String(), nullable=False),
        sa.Column("request_record", sa.JSON(), nullable=False),
        sa.Column("job_id", sa.String(), sa.ForeignKey("jobs.id"), nullable=False),
        sa.Column("accepted_response", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
    )
    op.create_index("ix_job_submissions_job_id", "job_submissions", ["job_id"])


def downgrade() -> None:
    # Removing durable keys could permit duplicate paid work after a downgrade.
    raise RuntimeError(
        "Submission identities must be preserved; automatic downgrade is unsupported"
    )

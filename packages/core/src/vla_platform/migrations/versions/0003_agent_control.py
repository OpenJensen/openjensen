"""Durable agent identity and evidence; historical schema is independent of live models."""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"


def upgrade():
    op.create_table(
        "agent_sessions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("project_id", sa.String(), sa.ForeignKey("projects.id"), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("record", sa.JSON(), nullable=False),
        sa.Column("private", sa.JSON(), nullable=False),
    )
    op.create_index("ix_agent_sessions_project_id", "agent_sessions", ["project_id"])
    op.create_table(
        "agent_observations",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("session_id", sa.String(), sa.ForeignKey("agent_sessions.id"), nullable=False),
        sa.Column("project_id", sa.String(), nullable=False),
        sa.Column("record", sa.JSON(), nullable=False),
    )
    op.create_index("ix_agent_observations_session_id", "agent_observations", ["session_id"])
    op.create_table(
        "agent_proposals",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("session_id", sa.String(), sa.ForeignKey("agent_sessions.id"), nullable=False),
        sa.Column("request_id", sa.String(), nullable=False),
        sa.Column("request_sha256", sa.String(), nullable=False),
        sa.Column("record", sa.JSON(), nullable=False),
    )
    op.create_index("ix_agent_proposals_session_id", "agent_proposals", ["session_id"])
    op.create_table(
        "agent_receipts",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("session_id", sa.String(), sa.ForeignKey("agent_sessions.id"), nullable=False),
        sa.Column("proposal_id", sa.String(), nullable=False, unique=True),
        sa.Column("record", sa.JSON(), nullable=False),
    )
    op.create_index("ix_agent_receipts_session_id", "agent_receipts", ["session_id"])
    op.create_table(
        "agent_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(), sa.ForeignKey("agent_sessions.id"), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.Column("record", sa.JSON(), nullable=False),
    )
    op.create_index("ix_agent_events_session_id", "agent_events", ["session_id"])


def downgrade():
    raise RuntimeError("Agent execution identities must be retained")

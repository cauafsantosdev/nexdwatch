"""Harden interaction and ingestion schema.

Revision ID: a7c2d9e41f30
Revises: f83c598356b3
Create Date: 2026-08-07 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "a7c2d9e41f30"
down_revision: str | Sequence[str] | None = "f83c598356b3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply nullable ratings, uniqueness, and queue lifecycle metadata."""
    op.execute("ALTER TYPE status ADD VALUE IF NOT EXISTS 'FILTERED'")

    op.alter_column(
        "logs",
        "rating",
        existing_type=sa.Float(),
        nullable=True,
    )
    op.alter_column(
        "logs_pending",
        "rating",
        existing_type=sa.Float(),
        nullable=True,
    )

    # Constraint creation aborts with actionable diagnostics instead of silently
    # deleting interactions or queue records whose canonical row is ambiguous.
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM logs
                GROUP BY user_id, film_id HAVING COUNT(*) > 1
            ) THEN
                RAISE EXCEPTION
                    'duplicate logs exist for (user_id, film_id); reconcile before migration';
            END IF;
            IF EXISTS (
                SELECT 1 FROM films_queue
                GROUP BY film_slug HAVING COUNT(*) > 1
            ) THEN
                RAISE EXCEPTION
                    'duplicate films_queue slugs exist; reconcile before migration';
            END IF;
        END
        $$
        """
    )
    op.create_unique_constraint(
        "uq_logs_user_id_film_id", "logs", ["user_id", "film_id"]
    )
    op.create_unique_constraint(
        "uq_films_queue_film_slug", "films_queue", ["film_slug"]
    )

    op.add_column(
        "films_queue",
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
    )
    op.add_column("films_queue", sa.Column("last_error", sa.Text(), nullable=True))
    op.add_column(
        "films_queue",
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.add_column(
        "films_queue",
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )


def downgrade() -> None:
    """Revert interaction and ingestion schema hardening."""
    op.drop_column("films_queue", "updated_at")
    op.drop_column("films_queue", "created_at")
    op.drop_column("films_queue", "last_error")
    op.drop_column("films_queue", "attempts")
    op.drop_constraint("uq_films_queue_film_slug", "films_queue", type_="unique")
    op.drop_constraint("uq_logs_user_id_film_id", "logs", type_="unique")

    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM logs WHERE rating IS NULL) THEN
                RAISE EXCEPTION
                    'unrated logs exist; resolve them before downgrading';
            END IF;
            IF EXISTS (SELECT 1 FROM logs_pending WHERE rating IS NULL) THEN
                RAISE EXCEPTION
                    'unrated pending logs exist; resolve them before downgrading';
            END IF;
        END
        $$
        """
    )
    op.alter_column(
        "logs_pending",
        "rating",
        existing_type=sa.Float(),
        nullable=False,
    )
    op.alter_column(
        "logs",
        "rating",
        existing_type=sa.Float(),
        nullable=False,
    )

    op.execute("UPDATE films_queue SET status = 'FAILED' WHERE status = 'FILTERED'")
    op.execute("UPDATE logs_pending SET status = 'FAILED' WHERE status = 'FILTERED'")
    op.execute(
        "CREATE TYPE status_without_filtered AS ENUM ('PENDING', 'PROCESSED', 'FAILED')"
    )
    op.execute(
        "ALTER TABLE films_queue ALTER COLUMN status DROP DEFAULT, "
        "ALTER COLUMN status TYPE status_without_filtered "
        "USING status::text::status_without_filtered"
    )
    op.execute(
        "ALTER TABLE logs_pending ALTER COLUMN status DROP DEFAULT, "
        "ALTER COLUMN status TYPE status_without_filtered "
        "USING status::text::status_without_filtered"
    )
    op.execute("DROP TYPE status")
    op.execute("ALTER TYPE status_without_filtered RENAME TO status")
    op.execute("ALTER TABLE films_queue ALTER COLUMN status SET DEFAULT 'PENDING'")
    op.execute("ALTER TABLE logs_pending ALTER COLUMN status SET DEFAULT 'PENDING'")

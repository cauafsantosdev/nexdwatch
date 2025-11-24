"""fixing ratings to floats

Revision ID: f83c598356b3
Revises: d21d72f5089f
Create Date: 2025-11-23 16:28:16.277323

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f83c598356b3'
down_revision: Union[str, Sequence[str], None] = 'd21d72f5089f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.alter_column('logs', 'rating',
                    existing_type=sa.Integer(),
                    type_=sa.Float(),
                    postgresql_using='rating::double precision')
    
    op.alter_column('logs_pending', 'rating',
                    existing_type=sa.Integer(),
                    type_=sa.Float(),
                    postgresql_using='rating::double precision')


def downgrade() -> None:
    """Downgrade schema."""
    pass

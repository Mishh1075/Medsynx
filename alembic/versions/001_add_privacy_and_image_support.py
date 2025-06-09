"""add privacy and image support

Revision ID: 001
Revises: 
Create Date: 2024-01-20 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Add new columns to users table
    op.add_column('users', sa.Column('full_name', sa.String(), nullable=True))
    op.add_column('users', sa.Column('is_superuser', sa.Boolean(), nullable=True, server_default='false'))
    op.add_column('users', sa.Column('created_at', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')))

    # Add new columns to datasets table
    op.add_column('datasets', sa.Column('file_type', sa.String(), nullable=True))

    # Add new columns to synthetic_jobs table
    op.add_column('synthetic_jobs', sa.Column('user_id', sa.Integer(), nullable=True))
    op.add_column('synthetic_jobs', sa.Column('model_type', sa.String(), nullable=True))
    op.add_column('synthetic_jobs', sa.Column('epsilon', sa.Float(), nullable=True))
    op.add_column('synthetic_jobs', sa.Column('delta', sa.Float(), nullable=True))
    op.add_column('synthetic_jobs', sa.Column('privacy_metrics', postgresql.JSON(), nullable=True))
    op.add_column('synthetic_jobs', sa.Column('utility_metrics', postgresql.JSON(), nullable=True))
    
    # Create foreign key for synthetic_jobs.user_id
    op.create_foreign_key(
        'synthetic_jobs_user_id_fkey',
        'synthetic_jobs', 'users',
        ['user_id'], ['id']
    )

    # Create image_generation_jobs table
    op.create_table(
        'image_generation_jobs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('original_file', sa.String(), nullable=True),
        sa.Column('status', sa.String(), nullable=True),
        sa.Column('num_images', sa.Integer(), nullable=True),
        sa.Column('epsilon', sa.Float(), nullable=True),
        sa.Column('delta', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('error_message', sa.String(), nullable=True),
        sa.Column('output_files', postgresql.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_image_generation_jobs_id', 'image_generation_jobs', ['id'], unique=False)

def downgrade():
    # Drop image_generation_jobs table
    op.drop_index('ix_image_generation_jobs_id', 'image_generation_jobs')
    op.drop_table('image_generation_jobs')

    # Remove foreign key from synthetic_jobs
    op.drop_constraint('synthetic_jobs_user_id_fkey', 'synthetic_jobs', type_='foreignkey')

    # Remove columns from synthetic_jobs
    op.drop_column('synthetic_jobs', 'utility_metrics')
    op.drop_column('synthetic_jobs', 'privacy_metrics')
    op.drop_column('synthetic_jobs', 'delta')
    op.drop_column('synthetic_jobs', 'epsilon')
    op.drop_column('synthetic_jobs', 'model_type')
    op.drop_column('synthetic_jobs', 'user_id')

    # Remove column from datasets
    op.drop_column('datasets', 'file_type')

    # Remove columns from users
    op.drop_column('users', 'created_at')
    op.drop_column('users', 'is_superuser')
    op.drop_column('users', 'full_name') 
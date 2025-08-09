#!/usr/bin/env python3
"""Fix missing UNIQUE constraint on prediction_outcomes table."""

from src.database.operations import db_ops
from src.utils.logger import get_logger
from sqlalchemy import text

logger = get_logger(__name__)

def add_unique_constraint():
    """Add UNIQUE constraint to prediction_id if it doesn't exist."""
    
    try:
        with db_ops.db.get_session() as session:
            # Check if constraint already exists
            check_query = text("""
                SELECT COUNT(*) 
                FROM information_schema.table_constraints 
                WHERE table_name = 'prediction_outcomes' 
                AND constraint_type = 'UNIQUE'
                AND constraint_name LIKE '%prediction_id%'
            """)
            
            result = session.execute(check_query).scalar()
            
            if result > 0:
                print("✅ UNIQUE constraint already exists on prediction_id")
                return True
            
            # Add the unique constraint
            print("Adding UNIQUE constraint to prediction_id...")
            alter_query = text("""
                ALTER TABLE prediction_outcomes 
                ADD CONSTRAINT prediction_outcomes_prediction_id_key 
                UNIQUE (prediction_id)
            """)
            
            session.execute(alter_query)
            session.commit()
            
            print("✅ Successfully added UNIQUE constraint on prediction_id")
            return True
            
    except Exception as e:
        if "already exists" in str(e):
            print("✅ Constraint already exists")
            return True
        else:
            logger.error(f"Error adding constraint: {e}")
            print(f"❌ Error: {e}")
            return False

if __name__ == "__main__":
    print("🔧 Fixing prediction_outcomes table constraint...")
    success = add_unique_constraint()
    
    if success:
        print("\n✅ Database fix applied successfully!")
        print("You can now run predictions and outcomes will be tracked correctly.")
    else:
        print("\n❌ Failed to apply database fix.")
        print("Please check the logs for details.")
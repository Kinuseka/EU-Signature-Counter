#!/usr/bin/env python3
"""
Migration script to update signature_history.json structure
From: [{"time": "...", "count": 123}, ...]
To: {"history": [...], "arima_pre": [], "sarima_pre": []}
"""

import json
import os
from datetime import datetime

def migrate_database():
    """Migrate signature_history.json to new structure"""
    
    filename = "signature_history.json"
    backup_filename = f"signature_history_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    print(f"Starting database migration...")
    
    # Check if file exists
    if not os.path.exists(filename):
        print(f"No existing {filename} found. Creating new structure.")
        new_data = {
            "history": [],
            "arima_pre": [],
            "sarima_pre": []
        }
        with open(filename, 'w') as f:
            json.dump(new_data, f, indent=2)
        print("Created new database structure.")
        return
    
    # Load existing data
    try:
        with open(filename, 'r') as f:
            old_data = json.load(f)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return
    
    # Check if already migrated
    if isinstance(old_data, dict) and "history" in old_data:
        print("Database already migrated. No action needed.")
        return
    
    # Create backup
    try:
        with open(backup_filename, 'w') as f:
            json.dump(old_data, f, indent=2)
        print(f"Created backup: {backup_filename}")
    except Exception as e:
        print(f"Error creating backup: {e}")
        return
    
    # Convert to new structure
    if isinstance(old_data, list):
        new_data = {
            "history": old_data,
            "arima_pre": [],
            "sarima_pre": []
        }
        
        # Save migrated data
        try:
            with open(filename, 'w') as f:
                json.dump(new_data, f, indent=2)
            print(f"Successfully migrated {len(old_data)} history records.")
            print("New structure: history, arima_pre, sarima_pre")
        except Exception as e:
            print(f"Error saving migrated data: {e}")
            return
    else:
        print(f"Unexpected data format in {filename}. Manual migration required.")
        return
    
    print("Migration completed successfully!")

if __name__ == "__main__":
    migrate_database() 
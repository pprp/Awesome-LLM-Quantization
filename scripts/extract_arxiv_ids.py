#!/usr/bin/env python3
"""
Extract arXiv IDs from README.md and create a tracking file
"""

import re
import json
import os
from typing import Set, List

def extract_arxiv_ids_from_readme(readme_path: str = "README.md") -> Set[str]:
    """Extract all arXiv IDs from README.md"""
    arxiv_ids = set()
    
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Find all arXiv URLs and extract IDs
        # Pattern matches: arxiv.org/abs/2508.04257 or arxiv.org/abs/2508.04257v1
        pattern = r'arxiv\.org/abs/(\d{4}\.\d{4,5})(?:v\d+)?'
        matches = re.findall(pattern, content, re.IGNORECASE)
        
        for match in matches:
            arxiv_ids.add(match)
            print(f"Found arXiv ID: {match}")
            
    except Exception as e:
        print(f"Error reading README: {e}")
        
    return arxiv_ids

def save_arxiv_ids(arxiv_ids: Set[str], output_file: str = "scripts/processed_papers.json"):
    """Save arXiv IDs to JSON file"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Convert set to sorted list for better readability
        ids_list = sorted(list(arxiv_ids))
        
        # Create data structure with metadata
        data = {
            "last_updated": "2025-08-12",
            "total_papers": len(ids_list),
            "processed_arxiv_ids": ids_list
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print(f"Saved {len(ids_list)} arXiv IDs to {output_file}")
        
    except Exception as e:
        print(f"Error saving arXiv IDs: {e}")

def main():
    """Main function"""
    print("Extracting arXiv IDs from README.md...")
    
    # Extract IDs from README
    arxiv_ids = extract_arxiv_ids_from_readme()
    
    if arxiv_ids:
        print(f"Found {len(arxiv_ids)} unique arXiv IDs")
        
        # Save to tracking file
        save_arxiv_ids(arxiv_ids)
        print("arXiv ID tracking file created successfully!")
    else:
        print("No arXiv IDs found in README.md")

if __name__ == "__main__":
    main()
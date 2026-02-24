#!/usr/bin/env python3
"""
Script to sort JSON file entries by (group_id, sample_id) tuple.
"""

import json
import argparse
import sys
from typing import List, Dict, Any


def read_json(file_path: str) -> List[Dict[str, Any]]:
    """Read JSON file and return list of entries."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Ensure data is a list
        if not isinstance(data, list):
            print(f"Error: JSON file must contain an array/list of objects.", file=sys.stderr)
            sys.exit(1)
            
        return data
        
    except FileNotFoundError:
        print(f"Error: Input file '{file_path}' not found.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {e}", file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        print(f"Error reading file '{file_path}': {e}", file=sys.stderr)
        sys.exit(1)


def sort_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort entries by (group_id, sample_id) tuple."""
    try:
        return sorted(entries, key=lambda x: (x['group_id'], x['sample_id']))
    except KeyError as e:
        print(f"Error: Missing required key {e} in one or more entries.", file=sys.stderr)
        print("Please ensure all entries have both 'group_id' and 'sample_id' fields.", file=sys.stderr)
        sys.exit(1)
    except TypeError as e:
        print(f"Error: Cannot sort entries - {e}", file=sys.stderr)
        print("Please ensure 'group_id' and 'sample_id' values are comparable types.", file=sys.stderr)
        sys.exit(1)


def write_json(entries: List[Dict[str, Any]], file_path: str, indent: int = 2) -> None:
    """Write entries to JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(entries, f, ensure_ascii=False, indent=indent)
    except IOError as e:
        print(f"Error writing to file '{file_path}': {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Sort JSON file entries by (group_id, sample_id) tuple"
    )
    parser.add_argument(
        'input_file',
        help='Path to input JSON file'
    )
    parser.add_argument(
        'output_file',
        help='Path to output JSON file'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print verbose output'
    )
    parser.add_argument(
        '--no-indent',
        action='store_true',
        help='Write compact JSON without indentation'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Reading entries from: {args.input_file}")
    
    # Read JSON file
    entries = read_json(args.input_file)
    
    if args.verbose:
        print(f"Read {len(entries)} entries")
    
    # Sort entries
    sorted_entries = sort_entries(entries)
    
    if args.verbose:
        print("Entries sorted by (group_id, sample_id)")
    
    # Write sorted entries
    indent = None if args.no_indent else 2
    write_json(sorted_entries, args.output_file, indent)
    
    if args.verbose:
        print(f"Sorted entries written to: {args.output_file}")
    else:
        print(f"Successfully sorted {len(entries)} entries and wrote to {args.output_file}")


if __name__ == "__main__":
    main() 
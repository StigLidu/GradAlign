#!/usr/bin/env python3
"""
Script to analyze similarity metrics from JSONL files.
Reads cosine similarity data where each line contains group_id and similarity,
creates visualizations, identifies top performers, and extracts corresponding 
lines from JSONL dataset files using group_id + 1 as line numbers.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import os
from pathlib import Path


def load_similarity_data(jsonl_file_path):
    """Load similarity data from JSONL file where each line contains group_id and similarity."""
    similarities = {}
    
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                
                # Extract group_id and similarity
                if 'group_id' in data and 'similarity' in data:
                    group_id = data['group_id']
                    similarity = data['similarity']
                    similarities[str(group_id)] = float(similarity)
                else:
                    print(f"Warning: Missing group_id or similarity at line {line_num}")
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
            except (KeyError, ValueError, TypeError) as e:
                print(f"Warning: Skipping malformed data at line {line_num}: {e}")
    
    # Calculate summary statistics
    if similarities:
        sim_values = [float(v) for v in similarities.values()]
        summary = {
            'num_groups': len(similarities),
            'mean_similarity': np.mean(sim_values),
            'std_similarity': np.std(sim_values),
            'max_similarity': max(sim_values),
            'min_similarity': min(sim_values)
        }
    else:
        summary = {}
    
    # Create the expected data structure
    result = {
        'cosine_similarities': similarities,
        'summary': summary
    }
    
    return result


def load_jsonl_dataset(jsonl_file_path):
    """Load dataset from JSONL file."""
    lines = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
                    lines.append(None)  # Keep index alignment
    return lines


def extract_similarities(data):
    """Extract similarity values and indices from the loaded data."""
    cosine_similarities = data.get('cosine_similarities', {})
    
    # Convert to lists for easier manipulation
    indices = []
    similarities = []
    
    for idx, sim in cosine_similarities.items():
        indices.append(int(idx))
        similarities.append(float(sim))
    
    # Sort by index to maintain order
    sorted_pairs = sorted(zip(indices, similarities))
    indices, similarities = zip(*sorted_pairs)
    
    return list(indices), list(similarities)


def get_top_n_similarities(indices, similarities, n=10):
    """Get top N highest similarity values."""
    # Create pairs and sort by similarity (descending)
    pairs = list(zip(indices, similarities))
    top_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:n]
    
    return top_pairs


def extract_corresponding_lines(top_pairs, dataset_lines):
    """Extract dataset lines corresponding to top similarity group_ids (using group_id + 1 as line number)."""
    corresponding_data = []
    
    for group_id, similarity in top_pairs:
        # group_id corresponds to line number - 1, so we use group_id directly as 0-based index
        line_idx = int(group_id)
        
        if line_idx >= len(dataset_lines):
            print(f"Warning: group_id {group_id} (line {line_idx + 1}) is out of range for dataset (size: {len(dataset_lines)})")
            corresponding_data.append({
                'group_id': group_id,
                'line_number': line_idx + 1,
                'similarity': similarity,
                'data': None
            })
        else:
            line_data = dataset_lines[line_idx]
            corresponding_data.append({
                'group_id': group_id,
                'line_number': line_idx + 1,
                'similarity': similarity,
                'data': line_data
            })
    
    return corresponding_data


def save_top_lines(corresponding_data, output_folder_path):
    """Save the top similarity lines to output folder in multiple formats."""
    output_folder = Path(output_folder_path)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Extract just the data for train files
    train_data = []
    for item in corresponding_data:
        if item['data'] is not None:
            train_data.append(item['data'])
    
    # Save as train.jsonl
    jsonl_path = output_folder / 'train.jsonl'
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for data in train_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    # Save as train.parquet
    if train_data:
        df = pd.DataFrame(train_data)
        parquet_path = output_folder / 'train.parquet'
        df.to_parquet(parquet_path, index=False)
        print(f"Top similarity data saved to: {parquet_path}")
    
    # Also save detailed analysis as JSON
    analysis_path = output_folder / 'similarity_analysis.json'
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(corresponding_data, f, indent=2, ensure_ascii=False)
    
    print(f"Top similarity data saved to: {jsonl_path}")
    print(f"Detailed analysis saved to: {analysis_path}")
    
    return jsonl_path, analysis_path


def create_similarity_plot(indices, similarities, top_pairs, output_path=None):
    """Create a distribution plot of similarity data."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Calculate statistics for binning
    min_sim = min(similarities)
    max_sim = max(similarities)
    
    # Create histogram with 20 bins from min to max
    n, bins, patches = ax.hist(similarities, bins=20, range=(min_sim, max_sim), 
                              alpha=0.7, color='lightblue', edgecolor='black', linewidth=0.5)
    
    # Highlight bins that contain top similarities
    top_similarities = [pair[1] for pair in top_pairs]
    
    # Color bins that contain top similarities
    for i, (left_edge, right_edge) in enumerate(zip(bins[:-1], bins[1:])):
        # Check if any top similarity falls in this bin
        bin_contains_top = any(left_edge <= sim < right_edge for sim in top_similarities)
        if bin_contains_top:
            patches[i].set_facecolor('red')
            patches[i].set_alpha(0.8)
    
    # Add vertical line at zero for reference
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
    
    # Formatting
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Cosine Similarities (20 bins)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = f'Total samples: {len(similarities)}\n'
    stats_text += f'Mean: {np.mean(similarities):.4f}\n'
    stats_text += f'Std: {np.std(similarities):.4f}\n'
    stats_text += f'Min: {min_sim:.4f}\n'
    stats_text += f'Max: {max_sim:.4f}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    return fig


def print_statistics(data, top_pairs, corresponding_data=None):
    """Print summary statistics and top similarities."""
    summary = data.get('summary', {})
    
    print("\n" + "="*50)
    print("SIMILARITY ANALYSIS RESULTS")
    print("="*50)
    
    if summary:
        print(f"Number of groups: {summary.get('num_groups', 'N/A')}")
        print(f"Mean similarity: {summary.get('mean_similarity', 'N/A'):.6f}")
        print(f"Std similarity: {summary.get('std_similarity', 'N/A'):.6f}")
        print(f"Max similarity: {summary.get('max_similarity', 'N/A'):.6f}")
        print(f"Min similarity: {summary.get('min_similarity', 'N/A'):.6f}")
    
    print(f"\nTop {len(top_pairs)} Highest Similarities:")
    print("-" * 40)
    for i, (group_id, sim) in enumerate(top_pairs, 1):
        print(f"{i:2d}. Group #{group_id:3d} (Line {int(group_id)+1:4d}): {sim:8.6f}")
    
    # Print preview of corresponding data if available
    if corresponding_data:
        print(f"\nPreview of Top {min(3, len(corresponding_data))} Dataset Lines:")
        print("-" * 50)
        for i, item in enumerate(corresponding_data[:3], 1):
            if item['data'] is not None:
                # Try to show a preview of the data
                data_str = json.dumps(item['data'], ensure_ascii=False)
                if len(data_str) > 100:
                    data_str = data_str[:97] + "..."
                print(f"{i}. Group #{item['group_id']} (Line {item['line_number']}) - sim: {item['similarity']:.4f}:")
                print(f"   {data_str}")
            else:
                print(f"{i}. Group #{item['group_id']} (Line {item['line_number']}) - sim: {item['similarity']:.4f}: [DATA NOT FOUND]")


def main():
    parser = argparse.ArgumentParser(description='Analyze similarity metrics and extract corresponding dataset lines')
    parser.add_argument('jsonl_file', help='Path to the JSONL file containing similarity data (group_id and similarity per line)')
    parser.add_argument('--dataset-folder', '-d', help='Path to the dataset folder containing train.jsonl')
    parser.add_argument('--output-folder', '-o', help='Output folder for extracted lines and plots (default: same dir as input)')
    parser.add_argument('--top-n', '-n', type=int, default=10, help='Number of top similarities to extract (default: 10)')
    parser.add_argument('--show', action='store_true', help='Display the plot interactively')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.jsonl_file):
        print(f"Error: File '{args.jsonl_file}' not found.")
        return 1
    
    try:
        # Load and process similarity data
        print(f"Loading similarity data from: {args.jsonl_file}")
        data = load_similarity_data(args.jsonl_file)
        indices, similarities = extract_similarities(data)
        top_pairs = get_top_n_similarities(indices, similarities, args.top_n)
        
        # Load dataset if provided
        corresponding_data = None
        if args.dataset_folder:
            dataset_folder = Path(args.dataset_folder)
            train_jsonl_path = dataset_folder / 'train.jsonl'
            
            if not dataset_folder.exists():
                print(f"Warning: Dataset folder '{args.dataset_folder}' not found. Skipping line extraction.")
            elif not train_jsonl_path.exists():
                print(f"Warning: train.jsonl not found in '{args.dataset_folder}'. Skipping line extraction.")
            else:
                print(f"Loading dataset from: {train_jsonl_path}")
                dataset_lines = load_jsonl_dataset(str(train_jsonl_path))
                corresponding_data = extract_corresponding_lines(top_pairs, dataset_lines)
                
                # Generate output folder path
                if args.output_folder:
                    output_folder = Path(args.output_folder)
                else:
                    input_path = Path(args.jsonl_file)
                    output_folder = input_path.parent / f"{input_path.stem}_top_{args.top_n}_selected"
                
                # Save the corresponding lines
                save_top_lines(corresponding_data, output_folder)
        
        # Generate output path for plot
        if args.output_folder:
            plot_output_folder = Path(args.output_folder)
            plot_output_folder.mkdir(parents=True, exist_ok=True)
            plot_output_path = plot_output_folder / "similarity_plot.png"
        else:
            input_path = Path(args.jsonl_file)
            plot_output_path = input_path.parent / f"{input_path.stem}_similarity_plot.png"
        
        # Create plot
        fig = create_similarity_plot(indices, similarities, top_pairs, str(plot_output_path))
        
        # Print statistics
        print_statistics(data, top_pairs, corresponding_data)
        
        # Show plot if requested
        if args.show:
            plt.show()
        
        print(f"\nAnalysis complete!")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 
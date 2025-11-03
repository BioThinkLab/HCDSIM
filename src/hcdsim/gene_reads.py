#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import os
import numpy as np
from scipy.stats import beta as beta_dist
import gzip
import subprocess
import tempfile
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simulate sequencing reads with coverage bias based on Lorenz curve using wgsim',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Example:
    python simulate_reads.py -f genome.fa -o output -cov 30 -l 150 -w 1000 -x 0.6 -y 0.3 -t 8
        '''
    )
    
    parser.add_argument('-f', '--fasta', required=True,
                        help='Input genome FASTA file (required)')
    parser.add_argument('-o', '--output', required=True,
                        help='Output prefix for simulated reads (required)')
    parser.add_argument('-cov', '--coverage', type=float, default=5.0,
                        help='Target average coverage depth (default: 30.0)')
    parser.add_argument('-l', '--read_length', type=int, default=150,
                        help='Read length (default: 150)')
    parser.add_argument('-w', '--window_size', type=int, default=50000,
                        help='Window size for coverage simulation (default: 1000)')
    parser.add_argument('-x', '--x0', type=float, default=0.5,
                        help='Lorenz curve parameter x0 (default: 0.5)')
    parser.add_argument('-y', '--y0', type=float, default=0.3,
                        help='Lorenz curve parameter y0 (default: 0.48)')
    parser.add_argument('-t', '--threads', type=int, default=1,
                        help='Number of threads to use (default: 1)')
    parser.add_argument('-s', '--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: None)')
    parser.add_argument('--wgsim', type=str, default='wgsim',
                        help='Path to wgsim executable (default: wgsim)')
    parser.add_argument('-e', '--error_rate', type=float, default=0.02,
                        help='Base error rate for wgsim (default: 0.02)')
    parser.add_argument('-d', '--insert_size', type=int, default=500,
                        help='Outer distance between the two ends for PE reads (default: 500)')
    parser.add_argument('-std', '--insert_std', type=int, default=50,
                        help='Standard deviation of insert size (default: 50)')
    parser.add_argument('--no-progress', action='store_true',
                        help='Disable progress bars')
    parser.add_argument('--chunk-size', type=int, default=1,
                        help='Number of windows to process per task (default: 10)')
    
    return parser.parse_args()


def check_wgsim(wgsim_path):
    """Check if wgsim is available"""
    try:
        result = subprocess.run([wgsim_path], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False


def read_fasta(fasta_file, show_progress=True):
    """Read all chromosomes from FASTA file"""
    chromosomes = {}
    current_chr = None
    current_seq = []
    
    # Handle gzipped files
    open_func = gzip.open if fasta_file.endswith('.gz') else open
    mode = 'rt' if fasta_file.endswith('.gz') else 'r'
    
    # Get file size for progress bar
    file_size = os.path.getsize(fasta_file)
    
    with open_func(fasta_file, mode) as f:
        if show_progress:
            pbar = tqdm(total=file_size, unit='B', unit_scale=True, 
                       desc='Reading FASTA', file=sys.stderr)
        
        for line in f:
            if show_progress:
                pbar.update(len(line.encode('utf-8')))
            
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('>'):
                # Save previous chromosome
                if current_chr is not None:
                    chromosomes[current_chr] = ''.join(current_seq)
                    if show_progress:
                        pbar.set_postfix({'chr': current_chr, 'len': f"{len(chromosomes[current_chr]):,}"})
                
                # Start new chromosome
                current_chr = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line.upper())
        
        # Save last chromosome
        if current_chr is not None:
            chromosomes[current_chr] = ''.join(current_seq)
        
        if show_progress:
            pbar.close()
    
    if not chromosomes:
        raise ValueError(f"No sequences found in {fasta_file}")
    
    print(f"\nTotal chromosomes loaded: {len(chromosomes)}", file=sys.stderr)
    for chr_name, seq in chromosomes.items():
        print(f"  {chr_name}: {len(seq):,} bp", file=sys.stderr)
    
    return chromosomes


def lorenz_curve(x, x0, y0):
    """Lorenz curve function"""
    if x <= x0:
        return (y0 / x0) * x
    else:
        return ((1 - y0) / (1 - x0)) * (x - x0) + y0


def fit_beta_to_lorenz(x0, y0, num_points=1000):
    """Fit Beta distribution parameters to Lorenz curve"""
    x_vals = np.linspace(0, 1, num_points)
    lorenz_vals = np.array([lorenz_curve(x, x0, y0) for x in x_vals])
    
    # Calculate derivatives to get PDF
    pdf_vals = np.diff(lorenz_vals) / np.diff(x_vals)
    pdf_vals = np.append(pdf_vals, pdf_vals[-1])
    
    # Normalize PDF
    pdf_vals = pdf_vals / np.sum(pdf_vals)
    
    # Estimate Beta parameters
    mean_val = np.sum(x_vals * pdf_vals)
    var_val = np.sum((x_vals - mean_val)**2 * pdf_vals)
    
    if var_val > 0:
        alpha = mean_val * ((mean_val * (1 - mean_val)) / var_val - 1)
        beta = (1 - mean_val) * ((mean_val * (1 - mean_val)) / var_val - 1)
        alpha = max(0.1, alpha)
        beta = max(0.1, beta)
    else:
        alpha, beta = 2.0, 2.0
    
    return alpha, beta


def gen_readcount(cov, l, window_size, num_windows, Alpha, Beta):
    """
    Generate read counts for each window using standard Metropolis-Hastings
    
    Note: Parameter 'u' is kept for API compatibility but not used in standard MH.
    Standard MH uses probabilistic acceptance based on the ratio.
    """
    
    # Calculate mean read count per window
    x0 = Alpha / (Alpha + Beta)
    mean_read = int(float(cov * window_size) / float(l))
    
    # Initialize
    readcounts = []
    
    # Starting point
    x_p = x0
    prob_x_p = beta_dist.pdf(x_p, Alpha, Beta)
    
    for i in range(num_windows):
        if i == 0:
            # First window uses initial value
            read_p = x_p / x0 * mean_read
            readcounts.append(int(read_p))
        else:
            # Proposal: sample from normal distribution centered at current value
            proposal_std = 0.1
            new_x = np.random.normal(x_p, proposal_std)
            
            # Ensure new_x is in valid range (0, 1)
            # Use reflection at boundaries
            while new_x <= 0 or new_x >= 1:
                if new_x <= 0:
                    new_x = -new_x
                if new_x >= 1:
                    new_x = 2 - new_x
            
            # Calculate probability of new value under Beta distribution
            new_x_p = beta_dist.pdf(new_x, Alpha, Beta)
            
            # Calculate acceptance probability
            prob_ratio = new_x_p / prob_x_p if prob_x_p > 0 else 1.0
            
            # Proposal is symmetric (normal distribution), so prop_ratio = 1
            acceptance_prob = min(1, prob_ratio)
            
            # Standard MH: accept with probability acceptance_prob
            if np.random.random() < acceptance_prob:
                x_p = new_x
                prob_x_p = new_x_p
            # else: keep current x_p (rejection)
            
            # Convert to read count
            read_p = x_p / x0 * mean_read
            readcounts.append(int(read_p))
    
    return readcounts

def write_fasta(chr_name, sequence, filename):
    """Write a single chromosome to FASTA file"""
    with open(filename, 'w') as f:
        f.write(f">{chr_name}\n")
        # Write sequence in 60bp lines
        for i in range(0, len(sequence), 60):
            f.write(sequence[i:i+60] + '\n')


def run_wgsim_for_window(args_tuple):
    """
    Run wgsim for a single window
    Returns the output FASTQ filenames
    """
    (chr_name, window_idx, window_seq, num_reads, read_length, 
     insert_size, insert_std, error_rate, wgsim_path, seed) = args_tuple
    
    if num_reads == 0:
        return None
    
    # Create temporary files
    temp_dir = tempfile.mkdtemp()
    temp_fasta = os.path.join(temp_dir, f"window_{chr_name}_{window_idx}.fa")
    temp_fq1 = os.path.join(temp_dir, f"reads_{chr_name}_{window_idx}_1.fq")
    temp_fq2 = os.path.join(temp_dir, f"reads_{chr_name}_{window_idx}_2.fq")
    
    # Write window sequence to temp FASTA
    write_fasta(f"{chr_name}_window_{window_idx}", window_seq, temp_fasta)
    
    # Build wgsim command
    wgsim_seed = seed if seed is not None else 0
    cmd = [
        wgsim_path,
        '-e', str(error_rate),
        '-d', str(insert_size),
        '-s', str(insert_std),
        '-N', str(num_reads),
        '-1', str(read_length),
        '-2', str(read_length),
        '-S', str(wgsim_seed),
        temp_fasta,
        temp_fq1,
        temp_fq2
    ]
    
    # Run wgsim
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return (temp_fq1, temp_fq2, temp_dir)
    except subprocess.CalledProcessError as e:
        print(f"Error running wgsim for {chr_name} window {window_idx}: {e}", file=sys.stderr)
        return None


def generate_all_window_tasks(chromosomes, args, alpha, beta, chr_seeds):
    """
    Generate all window tasks for all chromosomes
    Returns a list of tasks and metadata
    """
    all_tasks = []
    chr_stats = {}
    
    for chr_name, sequence in chromosomes.items():
        chr_seed = chr_seeds[chr_name] if chr_seeds else None
        
        # Set chromosome-specific seed
        if chr_seed is not None:
            np.random.seed(chr_seed)
        
        chr_len = len(sequence)
        num_windows = chr_len // args.window_size
        
        if num_windows == 0:
            print(f"Warning: Chromosome {chr_name} is too short ({chr_len} bp), skipping", file=sys.stderr)
            continue
        
        # Generate read counts for each window
        readcounts = gen_readcount(
            args.coverage/2, 
            args.read_length, 
            args.window_size, 
            num_windows, 
            alpha, 
            beta, 
        )
        
        # Create tasks for this chromosome
        chr_tasks = []
        total_reads = 0
        
        for i, count in enumerate(readcounts):
            window_start = i * args.window_size
            window_end = min(window_start + args.window_size, chr_len)
            window_seq = sequence[window_start:window_end]
            
            # Generate seed for this window
            window_seed = chr_seed + i if chr_seed is not None else None
            
            task = (
                chr_name, i, window_seq, count, args.read_length,
                args.insert_size, args.insert_std, args.error_rate,
                args.wgsim, window_seed
            )
            
            chr_tasks.append(task)
            total_reads += count
        
        all_tasks.extend(chr_tasks)
        chr_stats[chr_name] = {
            'length': chr_len,
            'windows': num_windows,
            'reads': total_reads
        }
    
    return all_tasks, chr_stats


def merge_fastq_files(temp_file_list, output_prefix, show_progress=True):
    """Merge all temporary FASTQ files into final output"""
    output_fq1 = f"{output_prefix}_1.fastq"
    output_fq2 = f"{output_prefix}_2.fastq"
    
    print(f"\nMerging FASTQ files...", file=sys.stderr)
    
    # Filter out None results
    valid_files = [f for f in temp_file_list if f is not None]
    
    # Merge R1 files
    if show_progress:
        pbar = tqdm(valid_files, desc='Merging R1', unit='file', file=sys.stderr)
    else:
        pbar = valid_files
    
    with open(output_fq1, 'w') as out1:
        for fq1, fq2, temp_dir in pbar:
            if os.path.exists(fq1):
                with open(fq1, 'r') as f:
                    out1.write(f.read())
    
    if show_progress:
        pbar.close()
    
    # Merge R2 files
    if show_progress:
        pbar = tqdm(valid_files, desc='Merging R2', unit='file', file=sys.stderr)
    else:
        pbar = valid_files
    
    with open(output_fq2, 'w') as out2:
        for fq1, fq2, temp_dir in pbar:
            if os.path.exists(fq2):
                with open(fq2, 'r') as f:
                    out2.write(f.read())
    
    if show_progress:
        pbar.close()
    
    # Clean up temporary files
    if show_progress:
        print("Cleaning up temporary files...", file=sys.stderr)
        pbar = tqdm(valid_files, desc='Cleanup', unit='file', file=sys.stderr)
    else:
        pbar = valid_files
    
    for fq1, fq2, temp_dir in pbar:
        try:
            if os.path.exists(fq1):
                os.remove(fq1)
            if os.path.exists(fq2):
                os.remove(fq2)
            # Remove temp fasta if exists
            temp_fasta = os.path.join(temp_dir, os.path.basename(fq1).replace('reads_', 'window_').replace('_1.fq', '.fa'))
            if os.path.exists(temp_fasta):
                os.remove(temp_fasta)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception as e:
            pass
    
    if show_progress:
        pbar.close()
    
    return output_fq1, output_fq2


def count_fastq_reads(fastq_file, show_progress=True):
    """Count number of reads in FASTQ file"""
    count = 0
    
    if show_progress:
        # Get file size for progress bar
        file_size = os.path.getsize(fastq_file)
        with open(fastq_file, 'r') as f:
            pbar = tqdm(total=file_size, unit='B', unit_scale=True,
                       desc='Counting reads', file=sys.stderr)
            for i, line in enumerate(f):
                pbar.update(len(line))
                if i % 4 == 0:  # Read name lines
                    count += 1
            pbar.close()
    else:
        with open(fastq_file, 'r') as f:
            for i, line in enumerate(f):
                if i % 4 == 0:
                    count += 1
    
    return count


def main():
    args = parse_args()
    
    show_progress = not args.no_progress
    
    # Check if wgsim is available
    print(f"Checking for wgsim at: {args.wgsim}", file=sys.stderr)
    if not check_wgsim(args.wgsim):
        print(f"Error: wgsim not found at '{args.wgsim}'", file=sys.stderr)
        print("Please install wgsim or specify the correct path with --wgsim", file=sys.stderr)
        sys.exit(1)
    print("wgsim found ✓", file=sys.stderr)
    
    # Validate threads parameter
    if args.threads < 1:
        print("Error: threads must be >= 1", file=sys.stderr)
        sys.exit(1)
    
    max_threads = cpu_count()
    if args.threads > max_threads:
        print(f"Warning: Requested {args.threads} threads but only {max_threads} CPUs available", file=sys.stderr)
        print(f"Using {max_threads} threads instead", file=sys.stderr)
        args.threads = max_threads
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Random seed set to: {args.seed}", file=sys.stderr)
    
    # Validate parameters
    if args.x0 < 0 or args.x0 > 1 or args.y0 < 0 or args.y0 > 1:
        print("Error: x0 and y0 must be between 0 and 1", file=sys.stderr)
        sys.exit(1)

    # Read all chromosomes from FASTA
    print(f"\nReading FASTA file: {args.fasta}", file=sys.stderr)
    chromosomes = read_fasta(args.fasta, show_progress)
    
    # Fit Beta distribution to Lorenz curve
    print(f"\nFitting Beta distribution to Lorenz curve (x0={args.x0}, y0={args.y0})...", file=sys.stderr)
    alpha, beta = fit_beta_to_lorenz(args.x0, args.y0)
    print(f"Beta distribution parameters: α={alpha:.4f}, β={beta:.4f}", file=sys.stderr)
    
    # Generate chromosome-specific seeds for reproducibility
    chr_seeds = None
    if args.seed is not None:
        chr_seeds = {chr_name: args.seed + i for i, chr_name in enumerate(chromosomes.keys())}
    
    # Generate all window tasks
    print(f"\nGenerating window tasks...", file=sys.stderr)
    all_tasks, chr_stats = generate_all_window_tasks(chromosomes, args, alpha, beta, chr_seeds)
    
    print(f"Total windows to process: {len(all_tasks):,}", file=sys.stderr)
    print(f"\nChromosome statistics:", file=sys.stderr)
    for chr_name, stats in chr_stats.items():
        print(f"  {chr_name}: {stats['windows']:,} windows, {stats['reads']:,} reads, {stats['length']:,} bp", file=sys.stderr)
    
    # Process windows in parallel
    print(f"\nUsing {args.threads} thread(s) for simulation...", file=sys.stderr)
    
    all_temp_files = []
    
    if args.threads == 1:
        # Single-threaded execution with progress bar
        if show_progress:
            pbar = tqdm(all_tasks, desc='Processing windows', unit='window', file=sys.stderr)
        else:
            pbar = all_tasks
        
        for task in pbar:
            result = run_wgsim_for_window(task)
            if result:
                all_temp_files.append(result)
        
        if show_progress:
            pbar.close()
    else:
        # Multi-threaded execution
        print("Starting parallel simulation...", file=sys.stderr)
        
        with Pool(processes=args.threads) as pool:
            if show_progress:
                # Use imap_unordered for better performance
                results = list(tqdm(
                    pool.imap_unordered(run_wgsim_for_window, all_tasks, chunksize=args.chunk_size),
                    total=len(all_tasks),
                    desc='Processing windows',
                    unit='window',
                    file=sys.stderr
                ))
            else:
                results = pool.map(run_wgsim_for_window, all_tasks, chunksize=args.chunk_size)
        
        # Collect all temporary files (filter out None results)
        all_temp_files = [r for r in results if r is not None]
    
    print(f"\nGenerated {len(all_temp_files):,} temporary file sets", file=sys.stderr)
    
    # Merge all FASTQ files
    output_fq1, output_fq2 = merge_fastq_files(all_temp_files, args.output, show_progress)
    
    # Count total reads
    print("\nCounting total reads...", file=sys.stderr)
    total_reads = count_fastq_reads(output_fq1, show_progress)
    
    # Print summary statistics
    print("\n" + "="*60, file=sys.stderr)
    print("SIMULATION SUMMARY", file=sys.stderr)
    print("="*60, file=sys.stderr)
    print(f"Threads used: {args.threads}", file=sys.stderr)
    print(f"Total chromosomes processed: {len(chromosomes)}", file=sys.stderr)
    print(f"Total windows processed: {len(all_tasks):,}", file=sys.stderr)
    print(f"Total read pairs generated: {total_reads:,}", file=sys.stderr)
    print(f"Total bases sequenced: {total_reads * args.read_length * 2:,}", file=sys.stderr)
    
    total_genome_size = sum(len(seq) for seq in chromosomes.values())
    actual_coverage = (total_reads * args.read_length * 2) / total_genome_size
    print(f"Actual average coverage: {actual_coverage:.2f}x", file=sys.stderr)
    print(f"Target coverage: {args.coverage:.2f}x", file=sys.stderr)
    print(f"\nOutput files:", file=sys.stderr)
    print(f"  R1: {output_fq1}", file=sys.stderr)
    print(f"  R2: {output_fq2}", file=sys.stderr)
    print("="*60, file=sys.stderr)
    
    print(f"\nDone!", file=sys.stderr)


if __name__ == '__main__':
    main()
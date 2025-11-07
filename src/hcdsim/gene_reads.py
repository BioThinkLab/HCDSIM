#!/usr/bin/env python3
"""
Single-cell DNA-seq Simulator with Coverage Bias using wgsim
Based on Mallory et al. 2020 PLOS Computational Biology

Author: Your Name
Date: 2024
"""

import os
import sys
import argparse
import tempfile
import shutil
import subprocess
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from scipy.special import betainc, betaincinv
from scipy.optimize import fsolve
from scipy.stats import beta as beta_dist, norm
from pyfaidx import Fasta
from tqdm import tqdm
import logging

# ============================================================================
# Logging Setup
# ============================================================================

def setup_logger(log_file=None):
    """Setup logging configuration"""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )
    return logging.getLogger(__name__)


# ============================================================================
# Lorenz Curve to Beta Distribution
# ============================================================================

# def lorenz_to_beta(x0, y0, logger=None):
#     """
#     Convert a point on Lorenz curve to Beta distribution parameters
    
#     Based on equations (1) and (2) in Mallory et al. 2020:
#     F(x) = I_x(α, β)
#     φ(x) = I_x(α+1, β)
    
#     Parameters:
#         x0: X-coordinate on Lorenz curve (cumulative proportion of bins)
#         y0: Y-coordinate on Lorenz curve (cumulative proportion of coverage)
        
#     Returns:
#         (alpha, beta): Parameters for Beta distribution
#     """
#     if logger:
#         logger.info(f"Converting Lorenz({x0}, {y0}) to Beta distribution...")
    
#     def equations(params):
#         alpha, beta = params
#         if alpha <= 0 or beta <= 0:
#             return [1e10, 1e10]
        
#         try:
#             # Find x value from inverse regularized incomplete beta function
#             x = betaincinv(alpha, beta, x0)
            
#             # Check constraints
#             eq1 = betainc(alpha, beta, x) - x0
#             eq2 = betainc(alpha + 1, beta, x) - y0
            
#             return [eq1, eq2]
#         except:
#             return [1e10, 1e10]
    
#     # Solve for alpha and beta
#     initial_guess = [2.0, 2.0]
#     solution = fsolve(equations, initial_guess)
#     alpha, beta = solution
    
#     # Validate
#     if alpha <= 0 or beta <= 0:
#         raise ValueError(f"Invalid Beta parameters: α={alpha}, β={beta}")
    
#     if logger:
#         logger.info(f"  Beta(α={alpha:.4f}, β={beta:.4f})")
    
#     return alpha, beta

def lorenz_to_beta(x0, y0, logger=None):
    """
    Convert a point on Lorenz curve to Beta distribution parameters
    
    Based on equations (1) and (2) in Mallory et al. 2020:
    F(x) = I_x(α, β)
    φ(x) = I_x(α+1, β)
    
    Parameters:
        x0: X-coordinate on Lorenz curve (cumulative proportion of bins)
        y0: Y-coordinate on Lorenz curve (cumulative proportion of coverage)
        
    Returns:
        (alpha, beta): Parameters for Beta distribution
    """
    if logger:
        logger.info(f"Converting Lorenz({x0}, {y0}) to Beta distribution...")
    
    def equations(params):
        alpha, beta = params
        if alpha <= 0 or beta <= 0:
            return [1e10, 1e10]
        
        # Add constraint to avoid extreme parameters
        if alpha + beta > 20:
            return [1e10, 1e10]
        
        try:
            # Find x value from inverse regularized incomplete beta function
            x = betaincinv(alpha, beta, x0)
            
            # Check constraints
            eq1 = betainc(alpha, beta, x) - x0
            eq2 = betainc(alpha + 1, beta, x) - y0
            
            return [eq1, eq2]
        except:
            return [1e10, 1e10]
    
    # Use adaptive initial guess based on y0 value
    # Smaller y0 needs smaller alpha/beta (flatter distribution)
    if y0 < 0.23:
        initial_alpha = 0.5 + (y0 - 0.15) * 6
    elif y0 < 0.28:
        initial_alpha = 1.0 + (y0 - 0.23) * 8
    elif y0 < 0.38:
        initial_alpha = 1.5 + (y0 - 0.28) * 15
    else:
        initial_alpha = 3.0 + (y0 - 0.38) * 10
    
    # Clamp initial guess to reasonable range
    initial_alpha = max(0.3, min(initial_alpha, 8.0))
    
    # Try multiple initial guesses to find the best solution
    best_solution = None
    best_error = float('inf')
    
    for init_alpha in [initial_alpha * 0.5, initial_alpha, initial_alpha * 1.5]:
        try:
            solution = fsolve(equations, [init_alpha, init_alpha], full_output=True)
            params, info, ier, msg = solution
            
            if ier == 1:  # Solution converged
                alpha, beta = params
                
                # Calculate error
                error = sum([e**2 for e in info['fvec']])
                
                # Validate solution
                if alpha > 0 and beta > 0 and alpha + beta <= 20:
                    # Additional check: verify the solution produces reasonable quantiles
                    try:
                        x = betaincinv(alpha, beta, x0)
                        actual_y = betainc(alpha + 1, beta, x)
                        
                        # Accept if error is small
                        if abs(actual_y - y0) < 0.01 and error < best_error:
                            best_error = error
                            best_solution = (alpha, beta)
                    except:
                        continue
        except:
            continue
    
    # If no good solution found, use fallback with relaxed constraints
    if best_solution is None:
        if logger:
            logger.warning(f"  Primary solver failed, using fallback method...")
        
        # Fallback: use bounded optimization
        from scipy.optimize import minimize
        
        def objective(params):
            alpha, beta = params
            if alpha <= 0 or beta <= 0:
                return 1e10
            
            try:
                x = betaincinv(alpha, beta, x0)
                err1 = (betainc(alpha, beta, x) - x0)**2
                err2 = (betainc(alpha + 1, beta, x) - y0)**2
                
                # Penalty for extreme parameters
                penalty = 0
                if alpha + beta > 15:
                    penalty = ((alpha + beta - 15) / 5)**2
                
                return err1 * 100 + err2 * 100 + penalty
            except:
                return 1e10
        
        result = minimize(
            objective,
            [initial_alpha, initial_alpha],
            method='L-BFGS-B',
            bounds=[(0.1, 10), (0.1, 10)]
        )
        
        if result.success and result.fun < 1.0:
            best_solution = tuple(result.x)
        else:
            # Last resort: use simple symmetric distribution
            if logger:
                logger.warning(f"  Fallback optimization failed, using approximate solution...")
            
            # Empirical approximation: α ≈ β ≈ f(y0)
            # Based on the pattern that smaller y needs smaller parameters
            approx_alpha = 0.5 + (y0 - 0.15) * 12
            approx_alpha = max(0.3, min(approx_alpha, 8.0))
            best_solution = (approx_alpha, approx_alpha)
    
    alpha, beta = best_solution
    
    # Final validation
    if alpha <= 0 or beta <= 0:
        raise ValueError(f"Invalid Beta parameters: α={alpha}, β={beta}")
    
    if logger:
        logger.info(f"  Beta(α={alpha:.4f}, β={beta:.4f}), α+β={alpha+beta:.4f}")
        
        # Log diagnostic information
        try:
            x = betaincinv(alpha, beta, x0)
            actual_y = betainc(alpha + 1, beta, x)
            variance = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
            logger.info(f"  Verification: target y={y0:.4f}, actual y={actual_y:.4f}, variance={variance:.6f}")
        except:
            pass
    
    return alpha, beta

# ============================================================================
# Coverage Sampling with Gaussian Copula
# ============================================================================

def sample_coverage_with_correlation(n_bins, alpha, beta, correlation_length=10, logger=None):
    """
    Sample coverage with spatial correlation using Gaussian copula
    
    This method ensures:
    1. Marginal distribution is exactly Beta(alpha, beta)
    2. Adjacent bins have spatial correlation
    3. Correlation decays exponentially with distance
    
    Parameters:
        n_bins: Number of bins
        alpha, beta: Beta distribution parameters
        correlation_length: Number of bins over which correlation decays
        
    Returns:
        coverage_array: Relative coverage (mean=1.0)
    """
    if logger:
        logger.info(f"Sampling {n_bins} bins using Gaussian copula...")
        logger.info(f"  Correlation length: {correlation_length} bins")
    
    # Step 1: Create correlation matrix (exponential decay)
    indices = np.arange(n_bins)
    distances = np.abs(indices[:, None] - indices[None, :])
    corr_matrix = np.exp(-distances / correlation_length)
    
    # Step 2: Sample from multivariate Gaussian
    mean = np.zeros(n_bins)
    
    # Use Cholesky decomposition for sampling
    try:
        L = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        # If matrix is not positive definite, add small diagonal
        corr_matrix += np.eye(n_bins) * 1e-6
        L = np.linalg.cholesky(corr_matrix)
    
    gaussian_samples = mean + L @ np.random.randn(n_bins)
    
    # Step 3: Transform to uniform via Gaussian CDF
    uniform_samples = norm.cdf(gaussian_samples)
    
    # Step 4: Transform to Beta via inverse CDF (quantile function)
    beta_samples = beta_dist.ppf(uniform_samples, alpha, beta)
    
    # Step 5: Normalize to mean=1.0
    coverage = beta_samples / np.mean(beta_samples)
    
    if logger:
        logger.info(f"  Relative coverage stats:")
        logger.info(f"    Mean: {np.mean(coverage):.6f}")
        logger.info(f"    Std:  {np.std(coverage):.6f}")
        logger.info(f"    Min:  {np.min(coverage):.6f}")
        logger.info(f"    Max:  {np.max(coverage):.6f}")
        logger.info(f"    CV:   {np.std(coverage)/np.mean(coverage):.4f}")
        
        # Compare with theoretical Beta distribution
        var_beta = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
        theoretical_std = np.sqrt(var_beta)
        logger.info(f"  Theoretical Beta std: {theoretical_std:.6f}")
    
    return coverage


# ============================================================================
# Genome Binning
# ============================================================================

def generate_bin_regions(fasta_file, bin_size, logger=None):
    """
    Divide genome into non-overlapping bins
    
    Parameters:
        fasta_file: Input genome FASTA file
        bin_size: Size of each bin in bp
        
    Returns:
        List of (chrom, start, end, bin_index) tuples
    """
    if logger:
        logger.info(f"Generating bins (bin_size={bin_size:,}bp)...")
    
    genome = Fasta(fasta_file)
    bins = []
    bin_idx = 0
    
    for chrom in genome.keys():
        chrom_len = len(genome[chrom])
        
        for start in range(0, chrom_len, bin_size):
            end = min(start + bin_size, chrom_len)
            bins.append((str(chrom), start, end, bin_idx))
            bin_idx += 1
    
    if logger:
        total_length = sum(len(genome[k]) for k in genome.keys())
        logger.info(f"  Total bins: {len(bins):,}")
        logger.info(f"  Genome length: {total_length:,}bp")
    
    return bins


# ============================================================================
# wgsim Worker Function
# ============================================================================

def wgsim_worker(args):
    """
    Worker function for generating reads for a single bin using wgsim
    
    Parameters:
        args: Tuple of (fasta_file, bin_info, coverage, params)
        
    Returns:
        (r1_fastq, r2_fastq, bin_idx, n_reads) or None if failed
    """
    fasta_file, bin_info, coverage, params = args
    chrom, start, end, bin_idx = bin_info
    bin_length = end - start
    
    # Calculate number of read pairs needed
    # coverage = (n_reads × 2 × read_length) / bin_length
    n_reads = int(coverage * bin_length / (2 * params['read_length']))
    
    # Skip bins with zero coverage
    if n_reads == 0:
        return None
    
    # Create temporary directory for this bin
    temp_dir = tempfile.mkdtemp(prefix=f'bin_{bin_idx}_')
    
    try:
        # Extract region using samtools faidx
        region_fasta = os.path.join(temp_dir, 'region.fa')
        region = f"{chrom}:{start+1}-{end}"  # 1-based coordinate
        
        cmd_extract = [
            'samtools', 'faidx',
            fasta_file,
            region
        ]
        
        with open(region_fasta, 'w') as f:
            result = subprocess.run(cmd_extract, stdout=f, stderr=subprocess.PIPE, 
                                   check=True, text=True)
        
        # wgsim output files
        r1_out = os.path.join(temp_dir, f'R1.fq')
        r2_out = os.path.join(temp_dir, f'R2.fq')
        
        # Run wgsim
        cmd_wgsim = [
            'wgsim',
            '-e', str(params['error_rate']),
            '-1', str(params['read_length']),
            '-2', str(params['read_length']),
            '-d', str(params['insert_size']),
            '-s', str(params['insert_std']),
            '-N', str(n_reads),
            '-r', '0',  # No mutations
            '-R', '0',  # No indel fraction
            '-X', '0',  # No indel extension
            '-S', str(params['seed'] + bin_idx),
            region_fasta,
            r1_out,
            r2_out
        ]
        
        result = subprocess.run(cmd_wgsim, capture_output=True, text=True, check=True)
        
        return (r1_out, r2_out, bin_idx, n_reads)
        
    except subprocess.CalledProcessError as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return None
    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return None


# ============================================================================
# FASTQ Merging
# ============================================================================

def merge_fastq_files(fastq_list, output_file, logger=None):
    """
    Merge multiple FASTQ files into one
    
    Parameters:
        fastq_list: List of FASTQ file paths
        output_file: Output merged FASTQ file path
    """
    if logger:
        logger.info(f"Merging {len(fastq_list)} FASTQ files into {output_file}...")
    
    total_reads = 0
    
    with open(output_file, 'w') as out_f:
        for fq in tqdm(fastq_list, desc="Merging", disable=(logger is None)):
            if fq and os.path.exists(fq):
                with open(fq, 'r') as in_f:
                    content = in_f.read()
                    out_f.write(content)
                    total_reads += content.count('\n') // 4
    
    if logger:
        logger.info(f"  Total reads in {output_file}: {total_reads:,}")
    
    return total_reads


# ============================================================================
# Lorenz Curve Verification
# ============================================================================

def calculate_lorenz_curve(coverage_array, logger=None):
    """
    Calculate Lorenz curve from coverage array
    
    Returns:
        (x_values, y_values): Arrays for Lorenz curve
    """
    # Sort coverage in ascending order
    sorted_cov = np.sort(coverage_array)
    n = len(sorted_cov)
    
    # Calculate cumulative proportions
    x = np.arange(1, n+1) / n  # Cumulative proportion of bins
    y = np.cumsum(sorted_cov) / np.sum(sorted_cov)  # Cumulative proportion of coverage
    
    # Add origin point
    x = np.concatenate([[0], x])
    y = np.concatenate([[0], y])
    
    if logger:
        # Find y value at x=0.5
        idx = np.argmin(np.abs(x - 0.5))
        logger.info(f"  Lorenz curve verification:")
        logger.info(f"    At x=0.5, y={y[idx]:.4f}")
        
        # Calculate Gini coefficient
        gini = 1 - 2 * np.trapz(y, x)
        logger.info(f"    Gini coefficient: {gini:.4f}")
    
    return x, y


# ============================================================================
# Main Pipeline
# ============================================================================

def simulate_biased_reads(args, logger):
    """
    Main pipeline for simulating biased single-cell DNA-seq reads
    """
    logger.info("="*70)
    logger.info("Single-cell DNA-seq Simulator with Coverage Bias")
    logger.info("Based on Mallory et al. 2020 (PLOS Comp Biol)")
    logger.info("="*70)
    
    # Validate inputs
    if not os.path.exists(args.fasta):
        logger.error(f"FASTA file not found: {args.fasta}")
        sys.exit(1)
    
    # Check required tools
    for tool in ['samtools', 'wgsim']:
        if shutil.which(tool) is None:
            logger.error(f"Required tool '{tool}' not found in PATH")
            sys.exit(1)
    
    # Step 1: Lorenz curve to Beta distribution
    logger.info("\n[Step 1/6] Converting Lorenz curve to Beta distribution")
    alpha, beta = lorenz_to_beta(args.lorenz_x, args.lorenz_y, logger)
    
    # Calculate approximate Gini coefficient
    gini_approx = 1 - 2 * args.lorenz_y / args.lorenz_x if args.lorenz_x > 0 else 0
    logger.info(f"  Target Gini coefficient: {gini_approx:.4f}")
    
    # Step 2: Generate bins
    logger.info(f"\n[Step 2/6] Dividing genome into bins")
    bins = generate_bin_regions(args.fasta, args.bin_size, logger)
    
    # Step 3: Sample RELATIVE coverage per bin
    logger.info(f"\n[Step 3/6] Sampling relative coverage per bin")
    relative_coverage = sample_coverage_with_correlation(
        len(bins), alpha, beta, 
        correlation_length=args.correlation_length, 
        logger=logger
    )
    
    # Verify Lorenz curve
    logger.info("\n  Verifying generated coverage distribution:")
    x_lorenz, y_lorenz = calculate_lorenz_curve(relative_coverage, logger)
    
    # Step 4: Scale to target coverage
    logger.info(f"\n[Step 4/6] Scaling coverage to target {args.coverage}X")
    coverage_per_bin = relative_coverage * args.coverage
    
    logger.info(f"  Final coverage stats:")
    logger.info(f"    Mean: {np.mean(coverage_per_bin):.6f}X")
    logger.info(f"    Std:  {np.std(coverage_per_bin):.6f}X")
    logger.info(f"    Min:  {np.min(coverage_per_bin):.6f}X")
    logger.info(f"    Max:  {np.max(coverage_per_bin):.6f}X")
    logger.info(f"    CV:   {np.std(coverage_per_bin)/np.mean(coverage_per_bin):.4f}")
    
    # Save coverage profile
    coverage_file = args.output_prefix + '_coverage.txt'
    logger.info(f"  Saving coverage profile to {coverage_file}")
    with open(coverage_file, 'w') as f:
        f.write("bin_index\tchrom\tstart\tend\trelative_coverage\tabsolute_coverage\n")
        for (chrom, start, end, idx), rel_cov, abs_cov in zip(bins, relative_coverage, coverage_per_bin):
            f.write(f"{idx}\t{chrom}\t{start}\t{end}\t{rel_cov:.6f}\t{abs_cov:.6f}\n")
    
    # Save Lorenz curve
    lorenz_file = args.output_prefix + '_lorenz.txt'
    logger.info(f"  Saving Lorenz curve to {lorenz_file}")
    with open(lorenz_file, 'w') as f:
        f.write("x\ty\n")
        for x, y in zip(x_lorenz, y_lorenz):
            f.write(f"{x:.6f}\t{y:.6f}\n")
    
    # Step 5: Generate reads with wgsim (parallel)
    logger.info(f"\n[Step 5/6] Generating reads with wgsim")
    logger.info(f"  Using {args.threads} threads")
    
    # Prepare parameters
    params = {
        'read_length': args.read_length,
        'insert_size': args.insert_size,
        'insert_std': args.insert_std,
        'error_rate': args.error_rate,
        'seed': args.seed
    }
    
    # Prepare tasks
    tasks = [
        (args.fasta, bin_info, cov, params)
        for bin_info, cov in zip(bins, coverage_per_bin)
    ]
    
    # Run parallel processing
    r1_files = []
    r2_files = []
    total_reads_expected = 0
    
    logger.info(f"  Processing {len(tasks)} bins...")
    
    with Pool(processes=args.threads) as pool:
        results = list(tqdm(
            pool.imap(wgsim_worker, tasks),
            total=len(tasks),
            desc="Generating reads"
        ))
    
    # Collect results
    temp_dirs = []
    for result in results:
        if result is not None:
            r1, r2, bin_idx, n_reads = result
            r1_files.append(r1)
            r2_files.append(r2)
            total_reads_expected += n_reads
            temp_dirs.append(os.path.dirname(r1))
    
    logger.info(f"  Successfully generated reads for {len(r1_files)}/{len(bins)} bins")
    logger.info(f"  Expected total read pairs: {total_reads_expected:,}")
    
    # Step 6: Merge FASTQ files
    logger.info(f"\n[Step 6/6] Merging FASTQ files")
    
    output_r1 = args.output_prefix + '_R1.fastq'
    output_r2 = args.output_prefix + '_R2.fastq'
    
    n_reads_r1 = merge_fastq_files(r1_files, output_r1, logger)
    n_reads_r2 = merge_fastq_files(r2_files, output_r2, logger)
    
    if n_reads_r1 != n_reads_r2:
        logger.warning(f"Read counts don't match: R1={n_reads_r1}, R2={n_reads_r2}")
    
    # Calculate actual coverage
    genome = Fasta(args.fasta)
    total_genome_length = sum(len(genome[k]) for k in genome.keys())
    actual_coverage = (n_reads_r1 * 2 * args.read_length) / total_genome_length
    
    logger.info(f"  Actual genome coverage: {actual_coverage:.6f}X")
    
    # Clean up
    logger.info(f"\nCleaning up temporary files")
    for temp_dir in tqdm(temp_dirs, desc="Cleaning up"):
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    logger.info(f"  Removed {len(temp_dirs)} temporary directories")
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("✓ Simulation completed successfully!")
    logger.info("="*70)
    logger.info(f"  Output files:")
    logger.info(f"    R1: {output_r1}")
    logger.info(f"    R2: {output_r2}")
    logger.info(f"    Coverage profile: {coverage_file}")
    logger.info(f"    Lorenz curve: {lorenz_file}")
    logger.info(f"  Total read pairs: {n_reads_r1:,}")
    logger.info(f"  Target mean coverage: {args.coverage}X")
    logger.info(f"  Actual mean coverage: {actual_coverage:.6f}X")
    logger.info(f"  Target Lorenz: ({args.lorenz_x}, {args.lorenz_y})")
    logger.info(f"  Target Gini: {gini_approx:.4f}")
    logger.info("="*70)


# ============================================================================
# Command Line Interface
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate single-cell DNA-seq with coverage bias using wgsim",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default Lorenz curve (Gini ≈ 0.2)
  python simulate_scDNA_wgsim.py -f genome.fa -o sim_output -cov 0.1 -t 8

  # Simulate MALBAC-like high variability (Gini ≈ 0.4)
  python simulate_scDNA_wgsim.py -f genome.fa -o malbac -cov 0.1 \\
      -x 0.5 -y 0.15 -t 16

  # Simulate low variability (Gini ≈ 0.05)
  python simulate_scDNA_wgsim.py -f genome.fa -o bulk -cov 0.1 \\
      -x 0.5 -y 0.45 -t 8

Technology presets (approximate Lorenz parameters):
  MALBAC:   -x 0.5 -y 0.15  (Gini ≈ 0.40)
  DOP-PCR:  -x 0.5 -y 0.25  (Gini ≈ 0.25)
  TnBC:     -x 0.5 -y 0.35  (Gini ≈ 0.15)
  Bulk:     -x 0.5 -y 0.45  (Gini ≈ 0.05)
"""
    )
    
    # Required arguments
    required = parser.add_argument_group('Required arguments')
    required.add_argument('-f', '--fasta', required=True,
                         help='Input genome FASTA file')
    required.add_argument('-o', '--output-prefix', required=True,
                         help='Output file prefix')
    
    # Lorenz curve parameters
    lorenz = parser.add_argument_group('Lorenz curve parameters')
    lorenz.add_argument('-x', '--lorenz-x', type=float, default=0.5,
                       help='Lorenz curve X coordinate (default: 0.5)')
    lorenz.add_argument('-y', '--lorenz-y', type=float, default=0.3,
                       help='Lorenz curve Y coordinate (default: 0.3, Gini≈0.2)')
    lorenz.add_argument('--correlation-length', type=int, default=10,
                       help='Spatial correlation length in bins (default: 10)')
    
    # Sequencing parameters
    seq = parser.add_argument_group('Sequencing parameters')
    seq.add_argument('-cov', '--coverage', type=float, default=0.1,
                    help='Target mean coverage for output FASTQ (default: 0.1)')
    seq.add_argument('-l', '--read-length', type=int, default=150,
                    help='Read length (default: 150)')
    seq.add_argument('-d', '--insert-size', type=int, default=300,
                    help='Insert size (default: 300)')
    seq.add_argument('-s', '--insert-std', type=int, default=50,
                    help='Insert size standard deviation (default: 50)')
    seq.add_argument('-e', '--error-rate', type=float, default=0.02,
                    help='Base error rate (default: 0.02)')
    
    # Binning parameters
    binning = parser.add_argument_group('Binning parameters')
    binning.add_argument('-b', '--bin-size', type=int, default=200000,
                        help='Bin size in bp (default: 200000)')
    
    # Runtime parameters
    runtime = parser.add_argument_group('Runtime parameters')
    runtime.add_argument('-t', '--threads', type=int, default=cpu_count(),
                        help=f'Number of threads (default: {cpu_count()})')
    runtime.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    runtime.add_argument('--log', type=str, default=None,
                        help='Log file path (default: stdout only)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.lorenz_x <= 0 or args.lorenz_x >= 1:
        parser.error("--lorenz-x must be in (0, 1)")
    if args.lorenz_y <= 0 or args.lorenz_y >= 1:
        parser.error("--lorenz-y must be in (0, 1)")
    if args.lorenz_y >= args.lorenz_x:
        parser.error("--lorenz-y must be less than --lorenz-x (inequality constraint)")
    if args.coverage <= 0:
        parser.error("--coverage must be positive")
    if args.bin_size <= 0:
        parser.error("--bin-size must be positive")
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    
    return args


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    args = parse_args()
    
    # Setup logger
    logger = setup_logger(args.log)
    
    # Set random seed
    np.random.seed(args.seed)
    
    try:
        simulate_biased_reads(args, logger)
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
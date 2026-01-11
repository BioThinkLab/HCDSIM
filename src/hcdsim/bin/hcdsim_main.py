import argparse
from hcdsim import HCDSIM
import inspect

def create_hcdsim_from_args(args):
    """
    Build HCDSIM instance from command line arguments.
    """
    hcdsim_params = inspect.signature(HCDSIM.__init__).parameters

    # Handle special list arguments
    args_dict = vars(args).copy()
    
    # Convert weights string to list of floats if provided
    if 'weights' in args_dict and args_dict['weights'] is not None:
        args_dict['weights'] = [float(x) for x in args_dict['weights'].split(',')]
    
    # Convert lambdas string to list of ints if provided
    if 'lambdas' in args_dict and args_dict['lambdas'] is not None:
        args_dict['lambdas'] = [int(x) for x in args_dict['lambdas'].split(',')]

    # Filter None values and keep only params that exist in HCDSIM.__init__
    hcdsim_args = {
        key: value
        for key, value in args_dict.items()
        if key in hcdsim_params and value is not None
    }

    # Create and return HCDSIM instance
    return HCDSIM(**hcdsim_args)

def sim(args):
    """Run the complete HCDSIM pipeline."""
    hcdsim = create_hcdsim_from_args(args)
    hcdsim.sim()

def generate_profile(args):
    """Generate CNA profile."""
    hcdsim = create_hcdsim_from_args(args)
    hcdsim.gprofile()

def generate_fasta(args):
    """Generate clone FASTA files."""
    hcdsim = create_hcdsim_from_args(args)
    hcdsim.gfasta()

def generate_fastq(args):
    """Generate clone FASTQ files."""
    hcdsim = create_hcdsim_from_args(args)
    hcdsim.gfastq()

def alignment(args):
    """Align FASTQ files to reference genome."""
    hcdsim = create_hcdsim_from_args(args)
    hcdsim.align()

def downsampling(args):
    """Downsample clone BAM to cell-level coverage."""
    hcdsim = create_hcdsim_from_args(args)
    hcdsim.downsam()

def process_cell_bam(args):
    """Process cell BAM files."""
    hcdsim = create_hcdsim_from_args(args)
    hcdsim.pbam()

def generate_barcode_bam(args):
    """Generate barcode-tagged BAM file."""
    hcdsim = create_hcdsim_from_args(args)
    hcdsim.bcbam()

def cal_rdr(args):
    """Calculate read depth ratios (RDR)."""
    hcdsim = create_hcdsim_from_args(args)
    hcdsim.rdr()

def cal_baf(args):
    """Calculate B-allele frequencies (BAF)."""
    hcdsim = create_hcdsim_from_args(args)
    hcdsim.baf()

def add_shared_arguments(parser):
    """Add basic arguments shared by all subcommands."""
    # HCDSIM param: ref_genome
    parser.add_argument('-r', '--ref_genome', type=str, metavar="", help='Path to reference genome [required]')
    # HCDSIM param: outdir
    parser.add_argument('-o', '--outdir', type=str, required=False, default='./', metavar="", help='Output directory (default: current directory)')
    # HCDSIM param: ignore
    parser.add_argument('-g', '--ignore', type=str, required=False, default=None, metavar="", help='Path to the exclusion list of contigs file (default: none)')
    # HCDSIM param: bin_size
    parser.add_argument('-b', '--bin_size', type=str, required=False, default='100kb', metavar="", help='The fixed bin size, with or without "kb" or "Mb" (default: 100kb)')
    # HCDSIM param: genome_version
    parser.add_argument('-gv', '--genome_version', type=str, required=False, default='hg38', metavar="", help='Genome version: hg19 or hg38 (default: hg38)')
    # HCDSIM param: clone_no
    parser.add_argument('-cno', '--clone_no', type=int, required=False, default=2, metavar="", help='The clone number contained in evolution tree, including normal clone (default: 2)')
    # HCDSIM param: cell_no
    parser.add_argument('-eno', '--cell_no', type=int, required=False, default=2, metavar="", help='The total cell number for this simulation dataset (default: 2)')
    # HCDSIM param: thread
    parser.add_argument('-t', '--thread', type=int, required=False, default=None, metavar="", help='Number of parallel jobs to use (default: equal to number of available processors)')
    # HCDSIM param: random_seed
    parser.add_argument('--random_seed', type=int, required=False, default=None, metavar="", help='Random seed for reproducibility (default: none)')

def add_executable_arguments(parser):
    """Add executable path arguments."""
    # HCDSIM param: wgsim
    parser.add_argument('--wgsim', type=str, required=False, default='wgsim', metavar="", help='Path to the executable "wgsim" file (default: in $PATH)')
    # HCDSIM param: bwa
    parser.add_argument('--bwa', type=str, required=False, default='bwa', metavar="", help='Path to the executable "bwa" file (default: in $PATH)')
    # HCDSIM param: samtools
    parser.add_argument('--samtools', type=str, required=False, default='samtools', metavar="", help='Path to the executable "samtools" file (default: in $PATH)')
    # HCDSIM param: bedtools
    parser.add_argument('--bedtools', type=str, required=False, default='bedtools', metavar="", help='Path to the executable "bedtools" file (default: in $PATH)')
    # HCDSIM param: bcftools
    parser.add_argument('--bcftools', type=str, required=False, default='bcftools', metavar="", help='Path to the executable "bcftools" file (default: in $PATH)')

def add_gprofile_arguments(parser):
    """Add CNA profile generation arguments."""
    # Tree structure arguments
    # HCDSIM param: tree_alpha
    parser.add_argument('--tree_alpha', type=float, required=False, default=10.0, metavar="", help='Alpha parameter for beta-splitting tree model (default: 10.0)')
    # HCDSIM param: tree_beta
    parser.add_argument('--tree_beta', type=float, required=False, default=10.0, metavar="", help='Beta parameter for beta-splitting tree model (default: 10.0)')
    # HCDSIM param: max_tree_depth
    parser.add_argument('-d', '--max_tree_depth', type=int, required=False, default=4, metavar="", help='The maximum depth of random evolution tree (default: 4)')
    # HCDSIM param: tree_depth_sigma
    parser.add_argument('--tree_depth_sigma', type=float, required=False, default=0.5, metavar="", help='Sigma for tree depth variation (default: 0.5)')
    # HCDSIM param: max_node_children
    parser.add_argument('--max_node_children', type=int, required=False, default=4, metavar="", help='Maximum number of children per node (default: 4)')
    # HCDSIM param: tree_balance_factor
    parser.add_argument('--tree_balance_factor', type=float, required=False, default=0.8, metavar="", help='Balance factor for tree generation (default: 0.8)')
    # HCDSIM param: tree_newwick
    parser.add_argument('--tree_newwick', type=str, required=False, default=None, metavar="", help='Path to a newick format tree file (default: none, generate random tree)')
    # HCDSIM param: tree_mode
    parser.add_argument('--tree_mode', type=int, required=False, default=0, metavar="", help='Tree generation mode (default: 0)')
    
    # SNP arguments
    # HCDSIM param: snp_list
    parser.add_argument('-l', '--snp_list', type=str, required=False, default=None, metavar="", help='Path to the known germline SNPs file (default: none, SNPs are placed randomly)')
    # HCDSIM param: snp_ratio
    parser.add_argument('-p', '--snp_ratio', type=float, required=False, default=0.001, metavar="", help='Ratio of SNPs to place randomly when a snp file is not given (default: 0.001)')
    # HCDSIM param: heho_ratio
    parser.add_argument('-hr', '--heho_ratio', type=float, required=False, default=0.67, metavar="", help='Ratio of heterozygous SNPs compared to homozygous ones (default: 0.67)')
    
    # CNA arguments
    # HCDSIM param: cna_prob
    parser.add_argument('-cp', '--cna_prob', type=float, required=False, default=0.02, metavar="", help='The probability of a bin undergoing CNA (default: 0.02)')
    # HCDSIM param: del_prob
    parser.add_argument('--del_prob', type=float, required=False, default=0.2, metavar="", help='Probability of deletion vs duplication (default: 0.2)')
    # HCDSIM param: cna_copy_param
    parser.add_argument('--cna_copy_param', type=float, required=False, default=0.5, metavar="", help='Parameter for geometric distribution of copy number (default: 0.5)')
    # HCDSIM param: max_cna_value
    parser.add_argument('--max_cna_value', type=int, required=False, default=10, metavar="", help='Maximum CNA value allowed (default: 10)')
    # HCDSIM param: max_ploidy
    parser.add_argument('--max_ploidy', type=int, required=False, default=None, metavar="", help='Maximum ploidy for WGD events (default: none)')
    
    # Mixture Poisson parameters for CNA length
    # HCDSIM param: weights (List[float])
    parser.add_argument('--weights', type=str, required=False, default=None, metavar="", help='Comma-separated weights for mixture Poisson distribution of CNA length, e.g., "0.3,0.4,0.2,0.1"')
    # HCDSIM param: lambdas (List[int])
    parser.add_argument('--lambdas', type=str, required=False, default=None, metavar="", help='Comma-separated lambda values for mixture Poisson distribution of CNA length, e.g., "5,20,100,300"')
    
    # CNA event counts
    # HCDSIM param: wgd
    parser.add_argument('-wgd', '--wgd', action='store_true', help='Enable whole-genome duplication (WGD) in tumor evolution (default: False)')
    # HCDSIM param: chrom_cna_no
    parser.add_argument('--chrom_cna_no', type=int, required=False, default=2, metavar="", help='Number of chromosomes to have chromosome-level CNAs (default: 2)')
    # HCDSIM param: chrom_arm_rate
    parser.add_argument('--chrom_arm_rate', type=float, required=False, default=0.75, metavar="", help='Rate of arm-level vs whole-chromosome CNAs (default: 0.75)')
    # HCDSIM param: loh_cna_no
    parser.add_argument('-loh', '--loh_cna_no', type=int, required=False, default=15, metavar="", help='Number of LOH CNAs per clone (default: 15)')
    # HCDSIM param: goh_cna_no
    parser.add_argument('-goh', '--goh_cna_no', type=int, required=False, default=5, metavar="", help='Number of GOH CNAs per clone (default: 5)')
    # HCDSIM param: unique_ratio
    parser.add_argument('--unique_ratio', type=float, required=False, default=0.5, metavar="", help='Ratio of cells with unique mutations per clone (default: 0.5)')

def add_gfasta_arguments(parser):
    """Add clone FASTA generation arguments."""
    pass
    
def add_gfastq_arguments(parser):
    """Add clone FASTQ generation arguments."""
    # HCDSIM param: clone_coverage
    parser.add_argument('-c', '--clone_coverage', type=float, required=False, default=30, metavar="", help='The reads coverage for clone (default: 30)')
    # HCDSIM param: reads_len
    parser.add_argument('-rl', '--reads_len', type=int, required=False, default=150, metavar="", help='The length of the reads in FASTQ (default: 150)')
    # HCDSIM param: insertion_size
    parser.add_argument('-i', '--insertion_size', type=int, required=False, default=350, metavar="", help='The outer distance between the two ends (default: 350)')
    # HCDSIM param: error_rate
    parser.add_argument('-e', '--error_rate', type=float, required=False, default=0.02, metavar="", help='The base error rate (default: 0.02)')
    # HCDSIM param: lorenz_x
    parser.add_argument('--lorenz_x', type=float, required=False, default=0.5, metavar="", help='Lorenz curve x parameter for coverage bias (default: 0.5)')
    # HCDSIM param: lorenz_y
    parser.add_argument('--lorenz_y', type=float, required=False, default=0.35, metavar="", help='Lorenz curve y parameter for coverage bias (default: 0.35)')
    # HCDSIM param: window_size
    parser.add_argument('--window_size', type=int, required=False, default=200000, metavar="", help='Window size for read generation (default: 200000)')
    # HCDSIM param: correlation_len
    parser.add_argument('--correlation_len', type=int, required=False, default=10, metavar="", help='Correlation length for coverage simulation (default: 10)')

def add_align_arguments(parser):
    """Add alignment arguments."""
    pass

def add_downsample_arguments(parser):
    """Add downsampling arguments."""
    # HCDSIM param: cell_coverage
    parser.add_argument('-cc', '--cell_coverage', type=float, required=False, default=0.01, metavar="", help='The reads coverage for cell (default: 0.01)')

def add_pbam_arguments(parser):
    """Add process cell BAM arguments."""
    pass

def add_bcbam_arguments(parser):
    """Add barcode BAM arguments."""
    # HCDSIM param: barcode_len
    parser.add_argument('-bcl', '--barcode_len', type=int, required=False, default=12, metavar="", help='Length of barcodes (default: 12)')

def main():
    parser = argparse.ArgumentParser(
        prog="hcdsim",
        description="HCDSIM: A Single-Cell Genomics Simulator with Haplotype-Specific Copy Number Annotation",
    )

    # Add subcommands
    subparsers = parser.add_subparsers(
        title="Subcommands",
        description="Available subcommands",
        help="Use `hcdsim <subcommand> --help` for more information.",
        dest="subcommand"
    )
    subparsers.required = True

    # sim subcommand - full pipeline
    sim_parser = subparsers.add_parser("sim", help="Run the complete HCDSIM pipeline.")
    add_shared_arguments(sim_parser)
    add_gprofile_arguments(sim_parser)
    add_gfasta_arguments(sim_parser)
    add_gfastq_arguments(sim_parser)
    add_align_arguments(sim_parser)
    add_downsample_arguments(sim_parser)
    add_pbam_arguments(sim_parser)
    add_bcbam_arguments(sim_parser)
    add_executable_arguments(sim_parser)
    sim_parser.set_defaults(func=sim)

    # gprofile subcommand
    gprofile_parser = subparsers.add_parser("gprofile", help="Generate CNA profile.")
    add_shared_arguments(gprofile_parser)
    add_gprofile_arguments(gprofile_parser)
    gprofile_parser.add_argument('--samtools', type=str, required=False, default='samtools', metavar="", help='Path to the executable "samtools" file (default: in $PATH)')
    gprofile_parser.set_defaults(func=generate_profile)

    # gfasta subcommand
    gfasta_parser = subparsers.add_parser("gfasta", help="Generate clone FASTA files.")
    add_shared_arguments(gfasta_parser)
    add_gfasta_arguments(gfasta_parser)
    gfasta_parser.set_defaults(func=generate_fasta)

    # gfastq subcommand
    gfastq_parser = subparsers.add_parser("gfastq", help="Generate clone FASTQ files.")
    add_shared_arguments(gfastq_parser)
    add_gfastq_arguments(gfastq_parser)
    gfastq_parser.add_argument('--wgsim', type=str, required=False, default='wgsim', metavar="", help='Path to the executable "wgsim" file (default: in $PATH)')
    gfastq_parser.add_argument('--samtools', type=str, required=False, default='samtools', metavar="", help='Path to the executable "samtools" file (default: in $PATH)')
    gfastq_parser.set_defaults(func=generate_fastq)

    # align subcommand
    align_parser = subparsers.add_parser("align", help="Align FASTQ files to reference genome.")
    add_shared_arguments(align_parser)
    add_align_arguments(align_parser)
    align_parser.add_argument('--bwa', type=str, required=False, default='bwa', metavar="", help='Path to the executable "bwa" file (default: in $PATH)')
    align_parser.add_argument('--samtools', type=str, required=False, default='samtools', metavar="", help='Path to the executable "samtools" file (default: in $PATH)')
    align_parser.set_defaults(func=alignment)

    # downsample subcommand
    downsample_parser = subparsers.add_parser("downsam", help="Downsample clone BAM to cell-level coverage.")
    add_shared_arguments(downsample_parser)
    add_downsample_arguments(downsample_parser)
    add_gfastq_arguments(downsample_parser)  # Need clone_coverage for ratio calculation
    downsample_parser.add_argument('--samtools', type=str, required=False, default='samtools', metavar="", help='Path to the executable "samtools" file (default: in $PATH)')
    downsample_parser.set_defaults(func=downsampling)

    # pbam subcommand
    pbam_parser = subparsers.add_parser("pbam", help="Process cell BAM files.")
    add_shared_arguments(pbam_parser)
    add_pbam_arguments(pbam_parser)
    pbam_parser.add_argument('--samtools', type=str, required=False, default='samtools', metavar="", help='Path to the executable "samtools" file (default: in $PATH)')
    pbam_parser.set_defaults(func=process_cell_bam)

    # bcbam subcommand
    bcbam_parser = subparsers.add_parser("bcbam", help="Generate barcode-tagged BAM file.")
    add_shared_arguments(bcbam_parser)
    add_bcbam_arguments(bcbam_parser)
    bcbam_parser.add_argument('--bwa', type=str, required=False, default='bwa', metavar="", help='Path to the executable "bwa" file (default: in $PATH)')
    bcbam_parser.add_argument('--samtools', type=str, required=False, default='samtools', metavar="", help='Path to the executable "samtools" file (default: in $PATH)')
    bcbam_parser.add_argument('--bcftools', type=str, required=False, default='bcftools', metavar="", help='Path to the executable "bcftools" file (default: in $PATH)')
    bcbam_parser.set_defaults(func=generate_barcode_bam)

    # rdr subcommand
    rdr_parser = subparsers.add_parser("rdr", help="Calculate read depth ratios (RDR).")
    add_shared_arguments(rdr_parser)
    rdr_parser.add_argument('--bedtools', type=str, required=False, default='bedtools', metavar="", help='Path to the executable "bedtools" file (default: in $PATH)')
    rdr_parser.add_argument('--samtools', type=str, required=False, default='samtools', metavar="", help='Path to the executable "samtools" file (default: in $PATH)')
    rdr_parser.set_defaults(func=cal_rdr)

    # baf subcommand
    baf_parser = subparsers.add_parser("baf", help="Calculate B-allele frequencies (BAF).")
    add_shared_arguments(baf_parser)
    baf_parser.add_argument('--bcftools', type=str, required=False, default='bcftools', metavar="", help='Path to the executable "bcftools" file (default: in $PATH)')
    baf_parser.add_argument('--bedtools', type=str, required=False, default='bedtools', metavar="", help='Path to the executable "bedtools" file (default: in $PATH)')
    baf_parser.add_argument('--samtools', type=str, required=False, default='samtools', metavar="", help='Path to the executable "samtools" file (default: in $PATH)')
    baf_parser.set_defaults(func=cal_baf)

    # Parse command line arguments
    args = parser.parse_args()
    if args.ref_genome is None or args.ref_genome.strip() == '':
        raise ValueError("The reference genome file (-r/--ref_genome) is required!")
    args.func(args)

if __name__ == "__main__":
    main()
import argparse
from hcdsim import HCDSIM
import inspect

def create_hcdsim_from_args(args):
    """
    from args buid HCDSIM
    """
    hcdsim_params = inspect.signature(HCDSIM.__init__).parameters

    # filter None value and same params
    hcdsim_args = {
        key: value
        for key, value in vars(args).items()
        if key in hcdsim_params and value is not None
    }

    # create HCDSim class
    return HCDSIM(**hcdsim_args)

def sim(args):
    # run hcdsim full pipeline
    hcdsim = create_hcdsim_from_args(args)
    hcdsim.sim()

def generate_profile(args):
    hcdsim = create_hcdsim_from_args(args)
    hcdsim.gprofile()

def generate_fasta(args):
    hcdsim = create_hcdsim_from_args(args)
    hcdsim.gfasta()

def generate_fastq(args):
    hcdsim = create_hcdsim_from_args(args)
    hcdsim.gfastq()

def alignment(args):
    hcdsim = create_hcdsim_from_args(args)
    hcdsim.align()

def downsampling(args):
    hcdsim = create_hcdsim_from_args(args)
    hcdsim.downsam()

def process_cell_bam(args):
    hcdsim = create_hcdsim_from_args(args)
    hcdsim.pbam()

def generate_barcode_bam(args):
    hcdsim = create_hcdsim_from_args(args)
    hcdsim.bcbam()

def cal_rdr(args):
    hcdsim = create_hcdsim_from_args(args)
    hcdsim.rdr()

def cal_baf(args):
    hcdsim = create_hcdsim_from_args(args)
    hcdsim.baf()

def add_shared_arguments(parser):
    # Basics, Inputs and Executables
    parser.add_argument('-r', '--ref-genome', type=str, metavar="", help='Path to reference genome [required]')
    parser.add_argument('-o', '--outdir', type=str, required=False, default='./', metavar="", help='Output directory (default: current directory)')
    parser.add_argument('-g', '--ignore', type=str, required=False, default=None, metavar="", help='Path to the exclusion list of contigs file containing a line-per-line list with chromosome names to ignore in the reference (default: none)')
    parser.add_argument('-b', '--bin-size', type=str, required=False, default='5Mb', metavar="", help='The fixed bin size, with or without \"kb\" or \"Mb\" (default: 5Mb)')
    parser.add_argument('-cno', '--clone-no', type=int, required=False, default=3, metavar="", help='The random clone number contained in evolution tree, including normal clone (default: 3)')
    parser.add_argument('-eno', '--cell-no', type=int, required=False, default=5, metavar="", help='The total cell number for this simultion dataset (deafult: 5)')
    parser.add_argument('-t', '--thread', type=int, required=False, default=None, metavar="", help='Number of parallele jobs to use (default: equal to number of available processors)')
    
def add_executable_arguments(parser):
    parser.add_argument('--wgsim', type=str, required=False, default='wgsim', metavar="", help='Path to the executable \"wgsim\" fileexecutable (default: in $PATH)')
    parser.add_argument('--bwa', type=str, required=False, default='bwa', metavar="", help='Path to the executable \"bwa\" file (default: in $PATH)')
    parser.add_argument('--samtools', type=str, required=False, default='samtools', metavar="", help='Path to the executable \"samtools\" file (default: in $PATH)')
    # parser.add_argument('--picard', type=str, required=False, default='picard.jar', metavar="", help='Path to the \"picard.jar\" file (default: ./picard.jar)')

def add_gprofile_arguments(parser):
    # CNA Profile
    parser.add_argument('-d', '--max-tree-depth', type=int, required=False, default=4, metavar="", help='The maximum depth of random evolution tree (default: 4)')
    parser.add_argument('-cp', '--cna-prob-cutoff', type=float, required=False, default=0.8, metavar="", help='The cutoff probability of a bin undergoing CNA, if random probability is larger than cutoff, CNA happens (default: 0.8)')
    parser.add_argument('-wgd', '--wgd-cna-no', type=int, required=False, default=0, metavar="", help='Number of clonal whole-genome duplications (WGDs) to introduce in the ancestor of tumor evolution (default: 0)')
    parser.add_argument('-wcl', '--wcl-cna-no', type=int, required=False, default=0, metavar="", help='Number of clonal whole-chromosome losses (WCLs) to introduce in the ancestor of tumor evolution (default: 0)')
    parser.add_argument('-loh', '--loh-cna-no', type=int, required=False, default=30, metavar="", help='Number of loss of heterozygosity (LOH) CNAs to introduce in the each clone of tumor evolution, including CNL_LOH, CNN_LOH, CNG_LOH (default: 30)')
    parser.add_argument('-goh', '--goh-cna-no', type=int, required=False, default=10, metavar="", help='Number of gain of heterozygosity (GOH) CNAs to introduce in the each clone of tumor evolution (default: 10)')
    parser.add_argument('-m', '--mirror-cna-no', type=int, required=False, default=10, metavar="", help='Number of mirror CNAs to introduce in the each clone of tumor evolution (default: 10)')
    parser.add_argument('-l', '--snp-list', type=str, required=False, default=None, metavar="", help='Path to the known germline SNPs file containing a SNP positions to add in the simulate human genome in the\nformat "#CHR POSITION REF_ALLELES ALT_ALLELES" with first line as headline (default: none, SNPs are placed randomly)')
    parser.add_argument('-p', '--snp-ratio', type=float, required=False, default=0.001, metavar="", help='Ratio of SNPs to place randomly when a snp file is not given (snp-ratio is requried only whether snp-list is not provided. default: 0.001).')
    parser.add_argument('-hr', '--heho-ratio', type=float, required=False, default=0.67, metavar="", help='Ratio of heterozygous SNPs compared to homozygous ones (default: 0.67)')

def add_gfasta_arguments(parser):
    # Clone FASTA
    pass
    
def add_gfastq_arguments(parser):
    # Clone FASTQ
    parser.add_argument('-c', '--clone-coverage', type=float, required=False, default=30, metavar="", help='The reads coverage for clone (default: 30)')
    parser.add_argument('-rl', '--reads-len', type=int, required=False, default=150, metavar="", help='The length of the reads in FASTQ (default: 150)')
    parser.add_argument('-i', '--insertion-size', type=int, required=False, default=350, metavar="", help='The outer distance between the two ends (default: 350)')
    parser.add_argument('-e', '--error-rate', type=float, required=False, default=0.0, metavar="", help='The base error rate (default: 0.0)')

def add_align_arguments(parser):
    # Alignment
    pass
    

def add_downsample_arguments(parser):
    # Downsampling
    parser.add_argument('-cc', '--cell-coverage', type=float, required=False, default=0.5, metavar="", help='The reads coverage for clone (default: 0.5)')

def add_pbam_arguments(parser):
    # Process Cell Bam
    pass

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

    # sim subcommand
    sim_parser = subparsers.add_parser("sim", help="Running the complete HCDSIM pipeline.")
    add_shared_arguments(sim_parser)
    add_gprofile_arguments(sim_parser)
    add_gfasta_arguments(sim_parser)
    add_gfastq_arguments(sim_parser)
    add_align_arguments(sim_parser)
    add_downsample_arguments(sim_parser)
    add_pbam_arguments(sim_parser)
    add_executable_arguments(sim_parser)
    sim_parser.set_defaults(func=sim)

    # gprofile subcommand
    gprofile_parser = subparsers.add_parser("gprofile", help="Generating CNA profile.")
    add_shared_arguments(gprofile_parser)
    add_gprofile_arguments(gprofile_parser)
    gprofile_parser.set_defaults(func=generate_profile)

    # gfasta subcommand
    gfasta_parser = subparsers.add_parser("gfasta", help="Generating clone FASTA file.")
    add_shared_arguments(gfasta_parser)
    add_gfasta_arguments(gfasta_parser)
    gfasta_parser.set_defaults(func=generate_fasta)

    # gfastq subcommand
    gfastq_parser = subparsers.add_parser("gfastq", help="Generating clone FASTQ file.")
    add_shared_arguments(gfastq_parser)
    add_gfastq_arguments(gfastq_parser)
    gfastq_parser.add_argument('--wgsim', type=str, required=False, default='wgsim', metavar="", help='Path to the executable \"wgsim\" fileexecutable (default: in $PATH)')
    gfastq_parser.set_defaults(func=generate_fastq)

    # align subcommand
    align_parser = subparsers.add_parser("align", help="Aligning clone FASTQ file.")
    add_shared_arguments(align_parser)
    add_align_arguments(align_parser)
    align_parser.add_argument('--bwa', type=str, required=False, default='bwa', metavar="", help='Path to the executable \"bwa\" file (default: in $PATH)')
    align_parser.add_argument('--samtools', type=str, required=False, default='samtools', metavar="", help='Path to the executable \"samtools\" file (default: in $PATH)')
    align_parser.set_defaults(func=alignment)

    # downsample subcommand
    downsample_parser = subparsers.add_parser("downsam", help="Downsampling clone BAM.")
    add_shared_arguments(downsample_parser)
    add_downsample_arguments(downsample_parser)
    downsample_parser.add_argument('--samtools', type=str, required=False, default='samtools', metavar="", help='Path to the executable \"samtools\" file (default: in $PATH)')
    downsample_parser.set_defaults(func=downsampling)

    # pbam subcommand
    pbam_parser = subparsers.add_parser("pbam", help="Processing cell BAMs.")
    add_shared_arguments(pbam_parser)
    add_pbam_arguments(pbam_parser)
    pbam_parser.add_argument('--samtools', type=str, required=False, default='samtools', metavar="", help='Path to the executable \"samtools\" file (default: in $PATH)')
    # pbam_parser.add_argument('--picard', type=str, required=False, default='picard.jar', metavar="", help='Path to the \"picard.jar\" file (default: in ./)')
    pbam_parser.set_defaults(func=process_cell_bam)

    # bcbam subcommand
    bcbam_parser = subparsers.add_parser("bcbam", help="Generating barcode BAM file.")
    add_shared_arguments(bcbam_parser)
    bcbam_parser.add_argument('-bcl', '--barcode-len', type=int, required=False, default=12, metavar="", help='Length of barcodes (default: 12)')
    bcbam_parser.add_argument('--bwa', type=str, required=False, default='bwa', metavar="", help='Path to the executable \"bwa\" file (default: in $PATH)')
    bcbam_parser.add_argument('--samtools', type=str, required=False, default='samtools', metavar="", help='Path to the executable \"samtools\" file (default: in $PATH)')
    bcbam_parser.add_argument('--bcftools', type=str, required=False, default='bcftools', metavar="", help='Path to the executable \"bcftools\" file (default: in $PATH)')
    bcbam_parser.set_defaults(func=generate_barcode_bam)

    # rdr subcommand
    rdr_parser = subparsers.add_parser("rdr", help="Computing RDRs.")
    add_shared_arguments(rdr_parser)
    rdr_parser.add_argument('--bedtools', type=str, required=False, default='bedtools', metavar="", help='Path to the executable \"bedtools\" file (default: in $PATH)')
    rdr_parser.set_defaults(func=cal_rdr)

    # baf subcommand
    baf_parser = subparsers.add_parser("baf", help="Computing BAFs.")
    add_shared_arguments(baf_parser)
    baf_parser.add_argument('--bcftools', type=str, required=False, default='bcftools', metavar="", help='Path to the executable \"bcftools\" file (default: in $PATH)')
    baf_parser.add_argument('--bedtools', type=str, required=False, default='bedtools', metavar="", help='Path to the executable \"bedtools\" file (default: in $PATH)')
    baf_parser.set_defaults(func=cal_baf)


    # 解析命令行参数
    args = parser.parse_args()
    if args.ref_genome is None or args.ref_genome.strip() == '': #or not os.path.isfile(args.ref_genome):
        raise ValueError("The specified human reference-genome file is required and does not exist!")
    args.func(args)

if __name__ == "__main__":
    main()
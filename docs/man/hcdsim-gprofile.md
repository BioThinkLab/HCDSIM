# hcdsim gprofile

`hcdsim gprofile` command to generate the CNA profile base on the given paramaters.

```shell
usage: hcdsim gprofile [-h] [-r] [-o] [-g] [-b] [-gv] [-cno] [-eno] [-t] [--random_seed] [--tree_alpha] [--tree_beta] [-d] [--tree_depth_sigma]
                       [--max_node_children] [--tree_balance_factor] [--tree_newwick] [--tree_mode] [-l] [-p] [-hr] [-cp] [--del_prob]
                       [--cna_copy_param] [--max_cna_value] [--max_ploidy] [--weights] [--lambdas] [-wgd] [--chrom_cna_no] [--chrom_arm_rate] [-loh]
                       [-goh] [--unique_ratio] [--samtools]

options:
  -h, --help            show this help message and exit
  -r , --ref_genome     Path to reference genome [required]
  -o , --outdir         Output directory (default: current directory)
  -g , --ignore         Path to the exclusion list of contigs file (default: none)
  -b , --bin_size       The fixed bin size, with or without "kb" or "Mb" (default: 100kb)
  -gv , --genome_version 
                        Genome version: hg19 or hg38 (default: hg38)
  -cno , --clone_no     The clone number contained in evolution tree, including normal clone (default: 2)
  -eno , --cell_no      The total cell number for this simulation dataset (default: 2)
  -t , --thread         Number of parallel jobs to use (default: equal to number of available processors)
  --random_seed         Random seed for reproducibility (default: none)
  --tree_alpha          Alpha parameter for beta-splitting tree model (default: 10.0)
  --tree_beta           Beta parameter for beta-splitting tree model (default: 10.0)
  -d , --max_tree_depth 
                        The maximum depth of random evolution tree (default: 4)
  --tree_depth_sigma    Sigma for tree depth variation (default: 0.5)
  --max_node_children   Maximum number of children per node (default: 4)
  --tree_balance_factor 
                        Balance factor for tree generation (default: 0.8)
  --tree_newwick        Path to a newick format tree file (default: none, generate random tree)
  --tree_mode           Tree generation mode (default: 0)
  -l , --snp_list       Path to the known germline SNPs file (default: none, SNPs are placed randomly)
  -p , --snp_ratio      Ratio of SNPs to place randomly when a snp file is not given (default: 0.001)
  -hr , --heho_ratio    Ratio of heterozygous SNPs compared to homozygous ones (default: 0.67)
  -cp , --cna_prob      The probability of a bin undergoing CNA (default: 0.02)
  --del_prob            Probability of deletion vs duplication (default: 0.2)
  --cna_copy_param      Parameter for geometric distribution of copy number (default: 0.5)
  --max_cna_value       Maximum CNA value allowed (default: 10)
  --max_ploidy          Maximum ploidy for WGD events (default: none)
  --weights             Comma-separated weights for mixture Poisson distribution of CNA length, e.g., "0.3,0.4,0.2,0.1"
  --lambdas             Comma-separated lambda values for mixture Poisson distribution of CNA length, e.g., "5,20,100,300"
  -wgd, --wgd           Enable whole-genome duplication (WGD) in tumor evolution (default: False)
  --chrom_cna_no        Number of chromosomes to have chromosome-level CNAs (default: 2)
  --chrom_arm_rate      Rate of arm-level vs whole-chromosome CNAs (default: 0.75)
  -loh , --loh_cna_no   Number of LOH CNAs per clone (default: 15)
  -goh , --goh_cna_no   Number of GOH CNAs per clone (default: 5)
  --unique_ratio        Ratio of cells with unique mutations per clone (default: 0.5)
  --samtools            Path to the executable "samtools" file (default: in $PATH)
```
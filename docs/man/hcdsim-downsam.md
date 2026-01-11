# hcdsim downsam

`hcdsim downsam` command to run downsampling from clone bam file to generate cell bams.

```shell
usage: hcdsim downsam [-h] [-r] [-o] [-g] [-b] [-gv] [-cno] [-eno] [-t] [--random_seed] [-cc] [-c] [-rl] [-i] [-e] [--lorenz_x] [--lorenz_y]
                      [--window_size] [--correlation_len] [--samtools]

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
  -cc , --cell_coverage 
                        The reads coverage for cell (default: 0.01)
  -c , --clone_coverage 
                        The reads coverage for clone (default: 30)
  -rl , --reads_len     The length of the reads in FASTQ (default: 150)
  -i , --insertion_size 
                        The outer distance between the two ends (default: 350)
  -e , --error_rate     The base error rate (default: 0.02)
  --lorenz_x            Lorenz curve x parameter for coverage bias (default: 0.5)
  --lorenz_y            Lorenz curve y parameter for coverage bias (default: 0.35)
  --window_size         Window size for read generation (default: 200000)
  --correlation_len     Correlation length for coverage simulation (default: 10)
  --samtools            Path to the executable "samtools" file (default: in $PATH)
```
# hcdsim baf

`hcdsim baf` command to compute BAFs with `bcftools` and `bedtools`.

```shell
usage: hcdsim baf [-h] [-r] [-o] [-g] [-b] [-gv] [-cno] [-eno] [-t] [--random_seed] [--bcftools] [--bedtools] [--samtools]

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
  --bcftools            Path to the executable "bcftools" file (default: in $PATH)
  --bedtools            Path to the executable "bedtools" file (default: in $PATH)
  --samtools            Path to the executable "samtools" file (default: in $PATH)
```
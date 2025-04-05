# hcdsim bcbam

`hcdsim bcbam` command to generate a barcode bam from all cell bams. This command is achieved using [`chisel_prep`](https://github.com/raphael-group/chisel/blob/master/man/chisel-prep.md) scripts.

```shell
usage: hcdsim bcbam [-h] [-r] [-o] [-g] [-b] [-cno] [-eno] [-t] [-bcl] [--bwa] [--samtools] [--bcftools]

options:
  -h, --help            show this help message and exit
  -r , --ref-genome     Path to reference genome [required]
  -o , --outdir         Output directory (default: current directory)
  -g , --ignore         Path to the exclusion list of contigs file containing a line-per-line list with chromosome names to ignore in the
                        reference (default: none)
  -b , --bin-size       The fixed bin size, with or without "kb" or "Mb" (default: 5Mb)
  -cno , --clone-no     The random clone number contained in evolution tree, including normal clone (default: 3)
  -eno , --cell-no      The total cell number for this simultion dataset (deafult: 5)
  -t , --thread         Number of parallele jobs to use (default: equal to number of available processors)
  -bcl , --barcode-len 
                        Length of barcodes (default: 12)
  --bwa                 Path to the executable "bwa" file (default: in $PATH)
  --samtools            Path to the executable "samtools" file (default: in $PATH)
  --bcftools            Path to the executable "bcftools" file (default: in $PATH)
```
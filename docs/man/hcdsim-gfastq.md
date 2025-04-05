# hcdsim gfastq

`hcdsim gfastq` command to generate fastq file for each clone using `wgsim`.

```shell
usage: hcdsim gfastq [-h] [-r] [-o] [-g] [-b] [-cno] [-eno] [-t] [-c] [-rl] [-i] [-e] [--wgsim]

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
  -c , --clone-coverage 
                        The reads coverage for clone (default: 30)
  -rl , --reads-len     The length of the reads in FASTQ (default: 150)
  -i , --insertion-size 
                        The outer distance between the two ends (default: 350)
  -e , --error-rate     The base error rate (default: 0.0)
  --wgsim               Path to the executable "wgsim" fileexecutable (default: in $PATH)
```
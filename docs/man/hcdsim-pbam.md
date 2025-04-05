# hcdsim pbam

`hcdsim pbam` command to process cell bams, including sort, markduplications and add reads group, using `samtools` tools.

```shell
usage: hcdsim pbam [-h] [-r] [-o] [-g] [-b] [-cno] [-eno] [-t] [--samtools]

options:
  -h, --help          show this help message and exit
  -r , --ref-genome   Path to reference genome [required]
  -o , --outdir       Output directory (default: current directory)
  -g , --ignore       Path to the exclusion list of contigs file containing a line-per-line list with chromosome names to ignore in the
                      reference (default: none)
  -b , --bin-size     The fixed bin size, with or without "kb" or "Mb" (default: 5Mb)
  -cno , --clone-no   The random clone number contained in evolution tree, including normal clone (default: 3)
  -eno , --cell-no    The total cell number for this simultion dataset (deafult: 5)
  -t , --thread       Number of parallele jobs to use (default: equal to number of available processors)
  --samtools          Path to the executable "samtools" file (default: in $PATH)
```
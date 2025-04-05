# hcdsim baf

`hcdsim baf` command to compute BAFs with `bcftools` and `bedtools`.

```shell
usage: hcdsim baf [-h] [-r] [-o] [-g] [-b] [-cno] [-eno] [-t] [--bcftools] [--bedtools]

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
  --bcftools          Path to the executable "bcftools" file (default: in $PATH)
  --bedtools          Path to the executable "bedtools" file (default: in $PATH)
```
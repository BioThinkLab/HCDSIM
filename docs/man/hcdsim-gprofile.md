# hcdsim gprofile

`hcdsim gprofile` command to generate the CNA profile base on the given paramaters.

```shell
usage: hcdsim gprofile [-h] [-r] [-o] [-g] [-b] [-cno] [-eno] [-t] [-d] [-cp] [-wgd] [-wcl] [-loh] [-goh] [-m] [-l] [-p] [-hr]

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
  -d , --max-tree-depth 
                        The maximum depth of random evolution tree (default: 4)
  -cp , --cna-prob-cutoff 
                        The cutoff probability of a bin undergoing CNA, if random probability is larger than cutoff, CNA happens (default: 0.8)
  -wgd , --wgd-cna-no   Number of clonal whole-genome duplications (WGDs) to introduce in the ancestor of tumor evolution (default: 0)
  -wcl , --wcl-cna-no   Number of clonal whole-chromosome losses (WCLs) to introduce in the ancestor of tumor evolution (default: 0)
  -loh , --loh-cna-no   Number of loss of heterozygosity (LOH) CNAs to introduce in the each clone of tumor evolution, including CNL_LOH,
                        CNN_LOH, CNG_LOH (default: 30)
  -goh , --goh-cna-no   Number of gain of heterozygosity (GOH) CNAs to introduce in the each clone of tumor evolution (default: 10)
  -m , --mirror-cna-no 
                        Number of mirror CNAs to introduce in the each clone of tumor evolution (default: 10)
  -l , --snp-list       Path to the known germline SNPs file containing a SNP positions to add in the simulate human genome in the format "#CHR
                        POSITION REF_ALLELES ALT_ALLELES" with first line as headline (default: none, SNPs are placed randomly)
  -p , --snp-ratio      Ratio of SNPs to place randomly when a snp file is not given (snp-ratio is requried only whether snp-list is not
                        provided. default: 0.001).
  -hr , --heho-ratio    Ratio of heterozygous SNPs compared to homozygous ones (default: 0.67)
```
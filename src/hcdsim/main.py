import copy
import datetime
import multiprocessing as mp
import os
import random
import sys
from collections import deque
from multiprocessing import Pool, Value, Lock
from scipy.stats import poisson

import pandas as pd

from . import beta_splitting_model_tree as random_tree
from . import utils
from .utils import ProgressBar, bcolors


pd.options.mode.chained_assignment = None

def init_gfasta(lock, counter, l):
    global gfasta_bar
    gfasta_bar = ProgressBar(lock=lock, counter=counter, total=l, length=40, verbose=False)

def init_gfastq(lock, counter, l):
    global gfastq_bar
    gfastq_bar = ProgressBar(lock=lock, counter=counter, total=l, length=40, verbose=False)

def init_align(lock, counter, l):
    global align_bar
    align_bar = ProgressBar(lock=lock, counter=counter, total=l, length=40, verbose=False)

def init_downsam(lock, counter, l):
    global downsam_bar
    downsam_bar = ProgressBar(lock=lock, counter=counter, total=l, length=40, verbose=False)

def init_pbam(lock, counter, l):
    global pbam_bar
    pbam_bar = ProgressBar(lock=lock, counter=counter, total=l, length=40, verbose=False)

def init_bcftools(lock, counter, l):
    global bcftools_bar
    bcftools_bar = ProgressBar(lock=lock, counter=counter, total=l, length=40, verbose=False)

def init_cov(lock, counter, l):
    global cov_bar
    cov_bar = ProgressBar(lock=lock, counter=counter, total=l, length=40, verbose=False)

class HCDSIM:
    def __init__(self, 
                ref_genome: str, 
                snp_list: str = None, 
                ignore: str = None, 
                outdir: str = './', 
                clone_no: int = 2, 
                cell_no: int = 2, 
                tree_alpha: float = 10.0, 
                tree_beta: float = 10.0, 
                max_tree_depth: int = 4, 
                tree_depth_sigma: float = 0.5, 
                max_node_children: int = 4, 
                tree_balance_factor: float = 0.8, 
                tree_seed: int = None,
                bin_size: str = '5Mb', 
                snp_ratio: float = 0.001, 
                thread: int = None, 
                heho_ratio: float = 0.67, 
                cna_prob_cutoff: float = 0.8, 
                clone_coverage: float = 30, 
                cell_coverage: float = 0.5, 
                reads_len: int = 150, 
                insertion_size: int = 350, 
                error_rate: float = 0.0, 
                wgd_cna_no: int = 0, 
                wcl_cna_no: int = 0, 
                loh_cna_no: int = 30, 
                goh_cna_no: int = 10, 
                mirror_cna_no: int = 10, 
                barcode_len: int = 12,
                lorenz_x: float = 0.5,
                lorenz_y: float = 0.4,
                max_ploidy: int = None,
                wgsim: str = 'wgsim', 
                samtools: str = 'samtools', 
                bwa: str = 'bwa', 
                bedtools: str = 'bedtools',
                bcftools: str = 'bcftools'):
        # binding each param to self
        params = locals()
        params.pop('self')
        for key, value in params.items():
            setattr(self, key, value)

        # validate thread
        if not thread:
            self.thread = mp.cpu_count()
            params['thread'] = self.thread
        
        # validate bin size 
        self._validate_bin_size(bin_size)

        # check params
        self.log('Parsing and checking arguments', level='PROGRESS')
        self._check_params()
        self.log('\n'.join(['Arguments:'] + ['\t{} : {}'.format(a, params[a]) for a in params]) + '\n', level='INFO')

        # set attributes of self for downstep calculation
        self.chrom_sizes = {}
        self.ignore_list = []
        self.samples = dict.fromkeys(['cell' + str(i+1) for i in range(self.cell_no)])
        for sample in self.samples:
            self.samples[sample] = {}

    def _validate_bin_size(self, bin_size):
        try:
            if bin_size.endswith("kb"):
                self.bin_size = int(bin_size[:-2]) * 1000
            elif bin_size.endswith("Mb"):
                self.bin_size = int(bin_size[:-2]) * 1000000
            else:
                self.bin_size = int(bin_size)
        except ValueError:
            raise ValueError("Bin-size must be a number, optionally ending with 'kb' or 'Mb'!")
    

    def setup_dir(self):
        outdir = self.outdir
        if any(os.path.isdir(os.path.join(outdir, x)) for x in ['profile', 'fasta', 'fastq', 'clone_bams', 'cell_bams', 'barcode_bam', 'rdr', 'baf', 'tmp', 'log']):
            self.log('Some of the working folders already exist in the running directory and content will be overwritten, please interrupt the process if this was not intended.', level='WARN')

        dprofile = os.path.join(outdir, 'profile')
        if not os.path.isdir(dprofile):
            os.mkdir(dprofile)

        dfasta = os.path.join(outdir, 'fasta')
        if not os.path.isdir(dfasta):
            os.mkdir(dfasta)

        dfastq = os.path.join(outdir, 'fastq')
        if not os.path.isdir(dfastq):
            os.mkdir(dfastq)

        dclone = os.path.join(outdir, 'clone_bams')
        if not os.path.isdir(dclone):
            os.mkdir(dclone)

        dcell = os.path.join(outdir, 'cell_bams')
        if not os.path.isdir(dcell):
            os.mkdir(dcell)
        
        dbarcode = os.path.join(outdir, 'barcode_bam')
        if not os.path.isdir(dbarcode):
            os.mkdir(dbarcode)

        drdr = os.path.join(outdir, 'rdr')
        if not os.path.isdir(drdr):
            os.mkdir(drdr)
        
        dbaf = os.path.join(outdir, 'baf')
        if not os.path.isdir(dbaf):
            os.mkdir(dbaf)

        dtmp = os.path.join(outdir, 'tmp')
        if not os.path.isdir(dtmp):
            os.mkdir(dtmp)

        dlog = os.path.join(outdir, 'log')
        if not os.path.isdir(dlog):
            os.mkdir(dlog)
        
        # create log files
        hcdsim_log = os.path.join(dlog, 'hcdsim_log.txt')
        wgsim_log = os.path.join(dlog, 'wgsim_log.txt')
        bwa_log = os.path.join(dlog, 'bwa_log.txt')
        samtools_log = os.path.join(dlog, 'samtools_log.txt')
        barcode_bam_log = os.path.join(dlog, 'barcode_bam_log.txt')
        bedtools_log = os.path.join(dlog, 'bedtools_log.txt')
        bcftools_log = os.path.join(dlog, 'bcftools_log.txt')
        for log_file in [hcdsim_log, wgsim_log, bwa_log, samtools_log, barcode_bam_log, bedtools_log, bcftools_log]:
            if not os.path.isfile(log_file):
                os.system('touch {}'.format(log_file))

        return dprofile, dfasta, dfastq, dclone, dcell, dbarcode, drdr, dbaf, dtmp, dlog
    
    def log(self, msg, level='STEP', lock=None):
        """
        输出日志信息到标准错误输出。

        :param msg: 需要输出的日志消息。
        :param level: 日志级别，决定日志的输出样式和颜色。
        :param lock: 用于并发控制的日志锁，确保日志输出的顺序性。
        """
        log_dir = os.path.join(self.outdir, 'log')
        if not os.path.isdir(log_dir):
                os.mkdir(log_dir)

        log_file = os.path.join(log_dir, 'hcdsim_log.txt')
        if not os.path.isfile(log_file):
            os.system('touch {}'.format(log_file))

        # 获取当前时间戳
        timestamp = f'{datetime.datetime.now():%Y-%b-%d %H:%M:%S}'

        # 根据日志级别选择不同的颜色
        if level == "STEP":
            color = f"{bcolors.BOLD}{bcolors.HEADER}"
        elif level == "INFO":
            color = f"{bcolors.OKGREEN}"
        elif level == "WARN":
            color = f"{bcolors.WARNING}"
        elif level == "PROGRESS":
            color = f"{bcolors.UNDERLINE}{bcolors.BBLUE}"
        elif level == "ERROR":
            color = f"{bcolors.FAIL}"
        else:
            color = ""

        # 组合颜色代码和日志信息，并在日志信息后重置颜色
        log_msg = f"{color}[{timestamp}]{msg}{bcolors.ENDC}"

        if lock is None:
            with open(log_file, 'a') as output:
                output.write(f"[{timestamp}]{msg}\n")
            sys.stderr.write(f"{log_msg}\n")
        else:
            with lock:
                with open(log_file, 'a') as output:
                    output.write(f"[{timestamp}]{msg}\n")
                sys.stderr.write(f"{log_msg}\n")

    def _check_params(self):
        """Check SCSilicon parameters

        This allows us to fail early - otherwise certain unacceptable
        parameter choices, such as threads='10.5', would only fail after
        minutes of runtime.

        Raises
        ------
        ValueError : unacceptable choice of parameters
        """
        utils.check_exist(ref_genome=self.ref_genome)
        if self.snp_list:
            utils.check_exist(snp_list=self.snp_list)
        if self.ignore:
            utils.check_exist(ignore=self.ignore)
        utils.check_int(clone_no=self.clone_no)
        utils.check_positive(clone_no=self.clone_no)
        utils.check_int(cell_no=self.cell_no)
        utils.check_positive(cell_no=self.cell_no)
        if self.clone_no < 2:
            raise ValueError(
                "The number of clones must be at least 2.")
        if self.cell_no < self.clone_no:
            raise ValueError(
                "The number of cells should not be less than the number of clones.")
        utils.check_int(max_tree_depth=self.max_tree_depth)
        utils.check_positive(max_tree_depth=self.max_tree_depth)
        utils.check_int(bin_size=self.bin_size)
        utils.check_positive(bin_size=self.bin_size)
        utils.check_between(0,1,heho_ratio=self.snp_ratio)
        utils.check_between(0,1,heho_ratio=self.heho_ratio)
        utils.check_between(0,1,cna_prob_cutoff=self.cna_prob_cutoff)
        utils.check_positive(clone_coverage=self.clone_coverage)
        utils.check_positive(cell_coverage=self.cell_coverage)
        utils.check_int(reads_len=self.reads_len)
        utils.check_positive(reads_len=self.reads_len)
        utils.check_int(reads_len=self.thread)
        utils.check_positive(reads_len=self.thread)
        utils.check_int(insertion_size=self.insertion_size)
        utils.check_positive(insertion_size=self.insertion_size)
        utils.check_between(0,1,error_rate=self.error_rate)
        utils.check_int(wgd_cna_no=self.wgd_cna_no)
        utils.check_lt_zero(wgd_cna_no=self.wgd_cna_no)
        utils.check_int(wcl_cna_no=self.wcl_cna_no)
        utils.check_lt_zero(wcl_cna_no=self.wcl_cna_no)
        utils.check_int(loh_cna_no=self.loh_cna_no)
        utils.check_lt_zero(loh_cna_no=self.loh_cna_no)
        utils.check_int(goh_cna_no=self.goh_cna_no)
        utils.check_lt_zero(goh_cna_no=self.goh_cna_no)
        utils.check_int(mirror_cna_no=self.mirror_cna_no)
        utils.check_lt_zero(mirror_cna_no=self.mirror_cna_no)
        utils.check_int(barcode_len=self.barcode_len)
        utils.check_positive(barcode_len=self.barcode_len)

    def set_params(self, **params):
        """Set the parameters of HCDSIM.

        Any parameters not given as named arguments will be left at their
        current value.

        Parameters
        ----------

        ref_genome: str, required
            The reference genome file path
        
        snp_list: str, required
            The snp list file

        outdir: str, optional, default: './'
            The output directory
        
        clone_no: int, optional, default: 1
            The random clone number contained in evolution tree
        
        cell_no: int, optional, default: 2
            The total cell number for this simultion dataset
        
        max_tree_depth: int, optional, default: 4
            The maximum depth of random evolution tree
        
        bin_size: int, optional, default: 500000
            The fixed bin length
        
        heho_ratio: float, optional, default: 0.5
            Ratio of heterozygous SNPs

        wgd_cna_no: int, optional, default: 0

        
        cna_prob_cutoff: float, optional, default: 0.8
            The cutoff probability of a bin undergoing CNV, if random probability is larger than cutoff, CNV happens
        
        clone_coverage: float, optional, default: 30
            The coverage for clone fastq file

        cell_coverage: float, optional, default: 0.5
            The coverage for each cell in a clone
        
        reads_len: int, optional, default: 150
            The reads length in fastq file
        
        insertion_size: int, optional, default: 350
            The outer distance between the two ends
        
        error_rate: float, optional, default: 0.02
            The base error rate

        Returns
        -------
        self
        """

        # parameters
        if 'ref_genome' in params and params['ref_genome'] != self.ref_genome:
            self.ref_genome = params['ref_genome']
            del params['ref_genome']
        if 'snp_list' in params and params['snp_list'] != self.snp_list:
            self.snp_list = params['snp_list']
            del params['snp_list']
        if 'ignore' in params and params['ignore'] != self.ignore:
            self.ignore = params['ignore']
            del params['ignore']
        if 'outdir' in params and params['outdir'] != self.outdir:
            self.outdir = params['outdir']
            del params['outdir']
        if 'clone_no' in params and params['clone_no'] != self.clone_no:
            self.clone_no = params['clone_no']
            del params['clone_no']
        if 'cell_no' in params and params['cell_no'] != self.cell_no:
            self.cell_no = params['cell_no']
            del params['cell_no']
        if 'max_tree_depth' in params and params['max_tree_depth'] != self.max_tree_depth:
            self.max_tree_depth = params['max_tree_depth']
            del params['max_tree_depth']
        if 'bin_size' in params and params['bin_size'] != self.bin_size:
            self.bin_size = params['bin_size']
            del params['bin_size']
        if 'thread' in params and params['thread'] != self.thread:
            self.thread = params['thread']
            del params['thread']
        if 'heho_ratio' in params and params['heho_ratio'] != self.heho_ratio:
            self.heho_ratio = params['heho_ratio']
            del params['heho_ratio']
        if 'cna_prob_cutoff' in params and params['cna_prob_cutoff'] != self.cna_prob_cutoff:
            self.cna_prob_cutoff = params['cna_prob_cutoff']
            del params['cna_prob_cutoff']
        if 'clone_coverage' in params and params['clone_coverage'] != self.clone_coverage:
            self.clone_coverage = params['clone_coverage']
            del params['clone_coverage']
        if 'cell_coverage' in params and params['cell_coverage'] != self.cell_coverage:
            self.cell_coverage = params['cell_coverage']
            del params['cell_coverage']
        if 'reads_len' in params and params['reads_len'] != self.reads_len:
            self.reads_len = params['reads_len']
            del params['reads_len']
        if 'insertion_size' in params and params['insertion_size'] != self.insertion_size:
            self.insertion_size = params['insertion_size']
            del params['insertion_size']
        if 'error_rate' in params and params['error_rate'] != self.error_rate:
            self.error_rate = params['error_rate']
            del params['error_rate']
        if 'wgd_cna_no' in params and params['wgd_cna_no'] != self.wgd_cna_no:
            self.wgd_cna_no = params['wgd_cna_no']
            del params['wgd_cna_no']
        if 'wcl_cna_no' in params and params['wcl_cna_no'] != self.wcl_cna_no:
            self.wcl_cna_no = params['wcl_cna_no']
            del params['wcl_cna_no']
        if 'loh_cna_no' in params and params['loh_cna_no'] != self.loh_cna_no:
            self.loh_cna_no = params['loh_cna_no']
            del params['loh_cna_no']
        if 'goh_cna_no' in params and params['goh_cna_no'] != self.goh_cna_no:
            self.goh_cna_no = params['goh_cna_no']
            del params['goh_cna_no']
        if 'mirror_cna_no' in params and params['mirror_cna_no'] != self.mirror_cna_no:
            self.mirror_cna_no = params['mirror_cna_no']
            del params['mirror_cna_no']
        self._check_params()
        self.get_params()
        return self

    def get_params(self):
        print(vars(self))

    def _get_chrom_sizes(self):
        if self.ignore:
            self.ignore_list = utils.parseIgnoreList(self.ignore)

        # check fasta.fai file
        fai_file = self.ref_genome + '.fai'
        if not os.path.exists(fai_file):
            samtools_log = os.path.join(self.outdir, 'log/samtools_log.txt')
            cmd = '{0} faidx {1}'.format(self.samtools, self.ref_genome)
            utils.runcmd(cmd, samtools_log)

        with open(fai_file, "r") as fai:
            for line in fai:
                fields = line.strip().split("\t")
                chrom_name = fields[0]
                chrom_size = int(fields[1])
                if chrom_name not in self.ignore_list:
                    self.chrom_sizes[chrom_name] = chrom_size

    def _buildGenome(self, maternalFasta, paternalFasta, allele_phase_file):
        if self.snp_list == None:
            allsnps = utils.randomSNPList(self.chrom_sizes, self.snp_ratio)
        else:
            allsnps = utils.parseSNPList(self.snp_list)

        phases = {}
        # m_genome = {}
        # p_genome = {}
        with open(self.ref_genome, 'r') as refinput:
            with open(maternalFasta, 'w') as out1:
                with open(paternalFasta, 'w') as out2:
                    chrom = None
                    snps = None
                    
                    for line in refinput:
                        line = line.strip()
                        if line.startswith('>'):
                            if chrom and chrom not in self.ignore_list:
                                out1.write('\n')
                                out2.write('\n')
                            chrom = line.strip()[1:].split()[0]
                            if chrom in self.ignore_list:
                                continue
                            out1.write(line+'\n')
                            out2.write(line+'\n')
                            # m_genome[chrom] = ''
                            # p_genome[chrom] = ''
                            snps = allsnps[chrom]
                            snppos = sorted(snps.keys())
                            currentpos = 0 
                            currentsnppos = snppos.pop(0)
                            allele1 = snps[currentsnppos][0]
                            allele2 = snps[currentsnppos][1]
                        else:
                            if chrom in self.ignore_list:
                                continue
                            linelen = len(line.strip())

                            if int(currentsnppos) > currentpos and int(currentsnppos) <= currentpos + linelen:
                                mline = line
                                pline = line
                                while int(currentsnppos) > currentpos and int(currentsnppos) <= currentpos + linelen:
                                    sindex = int(currentsnppos)-currentpos-1
                                    a = line[sindex]
                                    if a.upper() != 'N' and random.random() < self.heho_ratio: #Heterozygous
                                        if random.random() < 0.5:
                                            a1 = allele1.lower() if a.islower() else allele1.upper()
                                            a2 = allele2.lower() if a.islower() else allele2.upper()
                                            if a1 != a:
                                                tempa = a1
                                            else:
                                                tempa = a2
                                            # phases[(chrom, currentsnppos)] = a1.upper() + ',' + a2.upper() + ',0|1'
                                            phases[(chrom, currentsnppos)] = a.upper() + ',' + tempa.upper() + ',0|1'
                                            mline = mline[:sindex]+a+mline[sindex+1:]
                                            pline = pline[:sindex]+tempa+pline[sindex+1:]
                                        else:
                                            a1 = allele2.lower() if a.islower() else allele2.upper()
                                            a2 = allele1.lower() if a.islower() else allele1.upper()
                                            if a1 != a:
                                                tempa = a1
                                            else:
                                                tempa = a2
                                            phases[(chrom, currentsnppos)] = tempa.upper() + ',' + a.upper() + ',1|0'
                                            mline = mline[:sindex]+tempa+mline[sindex+1:]
                                            pline = pline[:sindex]+a+pline[sindex+1:]
                                    # else: #Homozygous
                                        # a1 = allele1.lower() if a.islower() else allele1.upper()
                                        # mline = mline[:sindex]+a+mline[sindex+1:]
                                        # pline = pline[:sindex]+a+mline[sindex+1:]
                                        # mline = 
                                    if snppos:
                                        currentsnppos = snppos.pop(0)
                                        allele1 = snps[currentsnppos][0]
                                        allele2 = snps[currentsnppos][1]
                                    else:
                                        break
                                # m_genome[chrom] += mline.strip()
                                # p_genome[chrom] += pline.strip()
                                out1.write(mline)
                                out2.write(pline)
                            else:
                                # m_genome[chrom] += line.strip()
                                # p_genome[chrom] += line.strip()
                                out1.write(line)
                                out2.write(line)
                            currentpos += len(line)
                    out1.write('\n')
                    out2.write('\n')

        with open(allele_phase_file, 'w') as output:
            for g in sorted(phases.keys(), key=(lambda x : (int(''.join([l for l in x[0] if l.isdigit()])), x[1]))):
                output.write('{},{},{}\n'.format(g[0], str(g[1]), phases[g]))
            

    def _filter_repeat_region(self, ref):
        # df[df['name'].str.contains('#DNA')][['#chrom','chromStart','chromEnd']].to_csv('dna_repeat.bed', sep='\t', index=False, header=False)
        
        ref_bed = os.path.join(self.outdir, 'profile/ref.bed')
        ref.to_csv(ref_bed, sep='\t', header=False, index=False)
        

        # merge region in repeat masker bed
        # use awk 'BEGIN { OFS = "\t" }{print $1, $2, $3}' rp-4 > repeat.bed to process the repeat masker file
        rep_tmp_bed = os.path.join(self.outdir, 'profile/rep.tmp.bed')
        command = "{0} merge -i {1} > {2}".format(self.bedtools_path, self.repeat_file, rep_tmp_bed)
        code = os.system(command)

        # get overlap between ref bed with repeat bed
        overlap_bed = os.path.join(self.outdir, 'profile/overlap.bed')
        command = "{0} intersect -a {1} -b {2} -wao > {3}".format(self.bedtools_path, ref_bed, rep_tmp_bed, overlap_bed)
        code = os.system(command)
        
        # calculate overlap ratio
        df = pd.read_csv(overlap_bed, sep='\t', header=None)
        df.columns = ["chrom1", "start1", "end1", "chrom2", "start2", "end2", "overlap"]
        grouped_df = df.groupby(["chrom1", "start1", "end1"])["overlap"].sum().reset_index()
        grouped_df["length"] = grouped_df["end1"] - grouped_df["start1"] + 1
        grouped_df["overlap_ratio"] = grouped_df["overlap"] / grouped_df["length"]

        return grouped_df

    def _split_chr_to_bins(self, chrom):
        """Split chromosomes to fixed-lenght bins

        Parameters
        ----------
        bin_size : int
            fixed-bin-length

        Returns
        -------
        ref: Dataframe of pandas
        """
        ref = pd.DataFrame(
            columns=['Chromosome', 'Start', 'End'])
        bin_size = self.bin_size
        if chrom != 'all':
            chrom_size = self.chrom_sizes[chrom]
            start = 1
            end = bin_size
            count = 1
            while(start < chrom_size):
                ref = pd.concat([ref, pd.DataFrame([{
                    'Chromosome': chrom,
                    'Start': start,
                    'End': min(end, chrom_size),
                }])], ignore_index=True)
                count += 1
                start = end + 1
                end = bin_size * count     
        else:
            for chrom, chrom_size in self.chrom_sizes.items():
                start = 1
                end = bin_size
                count = 1
                while(start < chrom_size):
                    ref = pd.concat([ref, pd.DataFrame([{
                        'Chromosome': chrom,
                        'Start': start,
                        'End': min(end, chrom_size),
                    }])], ignore_index=True)
                    count += 1
                    start = end + 1
                    end = bin_size * count            
        return ref

    def _generate_cnv_profile_for_each_clone(self, root, ref, m_fasta, p_fasta):
        cutoff = self.cna_prob_cutoff
        changes = []
        all_chroms = ref['Chromosome'].unique().tolist()
        if self.wgd_cna_no + self.wcl_cna_no > len(all_chroms):
            raise Exception("The sum of wgd_cna_no and wcl_cna_no should be less or equal to the total number of chromosomes!")

        # store maternal and paternal genome to dict
        maternal_genome = {}
        paternal_genome = {}
        with open(m_fasta, 'r') as input:
            chrom = None
            for line in input:
                line = line.strip()
                if line.startswith('>'):
                    chrom = line.strip()[1:].split()[0]
                    maternal_genome[chrom] = ''
                else:
                    maternal_genome[chrom] += line
        with open(p_fasta, 'r') as input:
            chrom = None
            for line in input:
                line = line.strip()
                if line.startswith('>'):
                    chrom = line.strip()[1:].split()[0]
                    paternal_genome[chrom] = ''
                else:
                    paternal_genome[chrom] += line
        
        # add nromal clone to cnv matrix
        root.maternal_cnvs = []
        root.paternal_cnvs = []
        root.changes = []
        for i in range(ref.shape[0]):
            root.maternal_cnvs.append(1)
            root.paternal_cnvs.append(1)
        ref[root.name+'_maternal_cnas'] = root.maternal_cnvs
        ref[root.name+'_paternal_cnas'] = root.paternal_cnvs
        
        # add the children of normal clone to queue
        queue = deque(root.children)

        max_ploidy_flag = False
        while  queue:
            clone = queue.popleft()
            clone.maternal_cnvs = []
            clone.paternal_cnvs = []
            clone.changes = []

            if clone.depth == 1: # children of normal clone
                mirrored_cnv_flag = False

                wgd_chroms = []
                wcl_chroms = []
                # select WGD and WCL chromosomes
                random_chroms = random.sample(all_chroms, self.wgd_cna_no+self.wcl_cna_no)
                wgd_chroms = random_chroms[:self.wgd_cna_no]
                wcl_chroms = random_chroms[self.wcl_cna_no:]
                wgd_cnvs = dict.fromkeys(wgd_chroms) # store the cnv number for each wgd chrom
                wcl_cnvs = dict.fromkeys(wcl_chroms) # store the cnv number for each wgd chrom

                # select the position for CNL_LOH, CNN_LOH, GOH and Mirror CNA
                cnl_loh_no = int(self.loh_cna_no/3)
                random_bins = random.sample(range(0, ref.shape[0]), self.loh_cna_no + self.goh_cna_no + self.mirror_cna_no)
                cnl_loh_bins = random_bins[:cnl_loh_no]
                cnn_loh_bins = random_bins[cnl_loh_no:self.loh_cna_no]
                goh_bins = random_bins[self.loh_cna_no:self.loh_cna_no + self.goh_cna_no]
                mirrored_cnv_bins = random_bins[self.loh_cna_no + self.goh_cna_no:]
                
                for i in range(ref.shape[0]):
                    # if flag is ture, the previous bin has been process as Mirror CNA bin and skip it.
                    if mirrored_cnv_flag:
                        mirrored_cnv_flag = False
                        continue
                    
                    current_chrom = ref['Chromosome'][i]
                    start = ref['Start'][i]
                    end = ref['End'][i]
                    m_sequence = maternal_genome[current_chrom][start-1:end]
                    p_sequence = paternal_genome[current_chrom][start-1:end]

                    # handle WGD
                    if current_chrom in wgd_chroms:
                        if not wgd_cnvs[current_chrom]:
                            if self.max_ploidy and not max_ploidy_flag:
                                m_cnv = self.max_ploidy
                                p_cnv = self.max_ploidy
                                max_ploidy_flag = True
                            else:
                                m_cnv = utils.random_WGD()
                                p_cnv = utils.random_WGD()
                            wgd_cnvs[current_chrom] = [m_cnv, p_cnv]
                            # 1. WGD in maternal and paternal 2. WGD in maternal 3. WGD in paternal
                            random_prob = random.random()
                            if random_prob < 1/3:
                                case = 0
                                wgd_cnvs[current_chrom].append(0)
                            elif random_prob > 2/3:
                                case = 1
                                wgd_cnvs[current_chrom].append(1)
                            else:
                                case = 2
                                wgd_cnvs[current_chrom].append(2)
                            # clone.changes.append('WGD')
                        else:
                            m_cnv = wgd_cnvs[current_chrom][0]
                            p_cnv = wgd_cnvs[current_chrom][1]
                            case = wgd_cnvs[current_chrom][2]

                        # 1. WGD in maternal and paternal 2. WGD in maternal 3. WGD in paternal
                        if case == 0:
                            changes.append(['normal',clone.name,'maternal','WGD',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(m_cnv)])
                            changes.append(['normal',clone.name,'paternal','WGD',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(p_cnv)])
                            clone.maternal_cnvs.append(m_cnv)
                            clone.paternal_cnvs.append(p_cnv)
                        elif case == 1:
                            changes.append(['normal',clone.name,'maternal','WGD',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(m_cnv)])
                            clone.maternal_cnvs.append(m_cnv)
                            clone.paternal_cnvs.append(1)
                        else:
                            changes.append(['normal',clone.name,'paternal','WGD',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(p_cnv)])
                            clone.maternal_cnvs.append(1)
                            clone.paternal_cnvs.append(p_cnv)
                        clone.changes.append('WGD')
                        continue
                    
                    # handle WCL
                    if current_chrom in wcl_chroms:
                        if not wcl_cnvs[current_chrom]:
                            # 1. WCL in maternal and paternal 2. WCL in maternal 3. WCL in paternal
                            random_prob = random.random()
                            if random_prob < 1/3:
                                wcl_cnvs[current_chrom] = (True, True)
                            elif random_prob > 2/3:
                                wcl_cnvs[current_chrom] = (True, False)
                            else:
                                wcl_cnvs[current_chrom] = (False, True)

                        if wcl_cnvs[current_chrom][0]:
                            changes.append(['normal',clone.name,'maternal','WCL',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(0)])
                            clone.maternal_cnvs.append(0)
                        else:
                            clone.maternal_cnvs.append(1)
                        if wcl_cnvs[current_chrom][1]:
                            changes.append(['normal',clone.name,'paternal','WCL',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(0)])
                            clone.paternal_cnvs.append(0)
                        else:
                            clone.paternal_cnvs.append(1)
                        clone.changes.append('WCL')
                        continue
                    
                    # handle CNL_LOH: 1:0 or 0:1
                    if i in cnl_loh_bins:
                        # check heterozygosity
                        if m_sequence != p_sequence:
                            # cnl_cnv = utils.random_CNL()
                            cnl_cnv = 1

                            if random.random() < 0.5: # m:p = 1:0
                                # if cnl_cnv != 1:
                                #     changes.append(['normal',clone.name,'maternal','CNL_LOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(cnl_cnv)])
                                changes.append(['normal',clone.name,'paternal','CNL_LOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(0)])
                                clone.maternal_cnvs.append(cnl_cnv)
                                clone.paternal_cnvs.append(0)
                            else: # m:p = 0:1
                                # if cnl_cnv != 1:
                                #     changes.append(['normal',clone.name,'paternal','CNL_LOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(cnl_cnv)])
                                changes.append(['normal',clone.name,'maternal','CNL_LOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(0)])
                                clone.maternal_cnvs.append(0)
                                clone.paternal_cnvs.append(cnl_cnv)
                            clone.changes.append('CNL_LOH')
                            continue
                        else:
                            clone.maternal_cnvs.append(1)
                            clone.paternal_cnvs.append(1)
                            clone.changes.append('NONE')
                            continue
                    
                    # handle CNN_LOH: 2:0 or 0:2
                    if i in cnn_loh_bins:
                        # check heterozygosity
                        if m_sequence != p_sequence:
                            cnn_cnv = utils.random_mirrored_cnv()

                            if random.random() < 0.5: # m:p = 2:0
                                if cnn_cnv == 2:
                                    changes.append(['normal',clone.name,'maternal','CNN_LOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(cnn_cnv)])
                                    changes.append(['normal',clone.name,'paternal','CNN_LOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(0)])
                                else:
                                    changes.append(['normal',clone.name,'maternal','CNG_LOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(cnn_cnv)])
                                    changes.append(['normal',clone.name,'paternal','CNG_LOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(0)])
                                clone.maternal_cnvs.append(cnn_cnv)
                                clone.paternal_cnvs.append(0)
                            else: # m:p = 0:1
                                if cnn_cnv == 2:
                                    changes.append(['normal',clone.name,'maternal','CNN_LOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(0)])
                                    changes.append(['normal',clone.name,'paternal','CNN_LOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(cnn_cnv)])
                                else:
                                    changes.append(['normal',clone.name,'maternal','CNG_LOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(0)])
                                    changes.append(['normal',clone.name,'paternal','CNG_LOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(cnn_cnv)])
                                clone.maternal_cnvs.append(0)
                                clone.paternal_cnvs.append(cnn_cnv)
                            if cnn_cnv == 2:
                                clone.changes.append('CNN_LOH')
                            else:
                                clone.changes.append('CNG_LOH')
                            continue
                        else:
                            clone.maternal_cnvs.append(1)
                            clone.paternal_cnvs.append(1)
                            clone.changes.append('NONE')
                            continue
                    
                    # handle GOH
                    if i in goh_bins:
                        # check heterozygosity
                        if m_sequence == p_sequence:
                            m_cnv = utils.random_WGD()
                            p_cnv = utils.random_WGD()
                            changes.append(['normal',clone.name,'maternal','GOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(m_cnv)])
                            changes.append(['normal',clone.name,'paternal','GOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(p_cnv)])
                            clone.maternal_cnvs.append(m_cnv)
                            clone.paternal_cnvs.append(p_cnv)
                            clone.changes.append('GOH')
                            continue
                        else:
                            clone.maternal_cnvs.append(1)
                            clone.paternal_cnvs.append(1)
                            clone.changes.append('NONE')
                            continue
                    
                    # handle Mirror CNA
                    if i in mirrored_cnv_bins:
                        # make sure the next bin located in same chromosome
                        if i+1 >= ref.shape[0] or ref['Chromosome'][i] != ref['Chromosome'][i+1]:
                            clone.maternal_cnvs.append(1)
                            clone.paternal_cnvs.append(1)
                            clone.changes.append('NONE')
                            continue

                        # generate Mirror CNA number
                        total_cnv = utils.random_mirrored_cnv()
                        cnv1 = random.randint(0,total_cnv)
                        while cnv1 == total_cnv/2:
                            cnv1 = random.randint(0, total_cnv)
                        cnv2 = total_cnv - cnv1
                        if random.random() < 0.5: # m:p = cnv1:cnv2
                            changes.append(['normal',clone.name,'maternal','Mirror CNA',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(cnv1)])
                            changes.append(['normal',clone.name,'maternal','Mirror CNA',ref['Chromosome'][i+1]+':'+str(ref['Start'][i+1])+'-'+str(ref['End'][i+1]),'1->'+str(cnv2)])
                            changes.append(['normal',clone.name,'paternal','Mirror CNA',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(cnv2)])
                            changes.append(['normal',clone.name,'paternal','Mirror CNA',ref['Chromosome'][i+1]+':'+str(ref['Start'][i+1])+'-'+str(ref['End'][i+1]),'1->'+str(cnv1)])
                            clone.maternal_cnvs.append(cnv1)
                            clone.paternal_cnvs.append(cnv2)
                            clone.maternal_cnvs.append(cnv2)
                            clone.paternal_cnvs.append(cnv1)
                        else: # m:p = cnv2:cnv1
                            changes.append(['normal',clone.name,'maternal','Mirror CNA',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(cnv2)])
                            changes.append(['normal',clone.name,'maternal','Mirror CNA',ref['Chromosome'][i+1]+':'+str(ref['Start'][i+1])+'-'+str(ref['End'][i+1]),'1->'+str(cnv1)])
                            changes.append(['normal',clone.name,'paternal','Mirror CNA',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(cnv1)])
                            changes.append(['normal',clone.name,'paternal','Mirror CNA',ref['Chromosome'][i+1]+':'+str(ref['Start'][i+1])+'-'+str(ref['End'][i+1]),'1->'+str(cnv2)])
                            clone.maternal_cnvs.append(cnv2)
                            clone.paternal_cnvs.append(cnv1)
                            clone.maternal_cnvs.append(cnv1)
                            clone.paternal_cnvs.append(cnv2)
                        mirrored_cnv_flag = True
                        clone.changes.append('Mirror CNA')
                        clone.changes.append('Mirror CNA')
                        continue
                    
                    # generate random cnv
                    if random.random() > cutoff: # 20% cnv
                        m_cnv = utils.random_cnv()
                        p_cnv = utils.random_cnv()
                        

                        # check whether is CNL_LOH
                        if (m_sequence != p_sequence) and ((m_cnv == 0 and p_cnv !=0) or (m_cnv != 0 and p_cnv ==0)):
                            if m_cnv == 1 or p_cnv == 1:
                                changes.append(['normal',clone.name,'maternal','CNL_LOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(m_cnv)])
                                changes.append(['normal',clone.name,'paternal','CNL_LOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(p_cnv)])
                                clone.changes.append('CNL_LOH')
                            elif m_cnv == 2 or p_cnv == 2:
                                changes.append(['normal',clone.name,'maternal','CNN_LOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(m_cnv)])
                                changes.append(['normal',clone.name,'paternal','CNN_LOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(p_cnv)])
                                clone.changes.append('CNN_LOH')
                            else:
                                changes.append(['normal',clone.name,'maternal','CNG_LOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(m_cnv)])
                                changes.append(['normal',clone.name,'paternal','CNG_LOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(p_cnv)])
                                clone.changes.append('CNG_LOH')
                            clone.maternal_cnvs.append(m_cnv)
                            clone.paternal_cnvs.append(p_cnv)
                            continue
                        
                        # check mirrored CNV
                        if clone.changes and clone.changes[-1] in ['REGULAR', 'NONE']:
                            if m_cnv != p_cnv and m_cnv == clone.paternal_cnvs[-1] and clone.maternal_cnvs[-1] == p_cnv:
                                if clone.changes[-1] == 'REGULAR':
                                    clone.changes.pop()
                                    changes.pop()
                                    changes.pop()
                                changes.append(['normal',clone.name,'maternal','Mirror CNA',ref['Chromosome'][i-1]+':'+str(ref['Start'][i-1])+'-'+str(ref['End'][i-1]),'1->'+str(clone.maternal_cnvs[-1])])
                                changes.append(['normal',clone.name,'maternal','Mirror CNA',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(m_cnv)])
                                changes.append(['normal',clone.name,'paternal','Mirror CNA',ref['Chromosome'][i-1]+':'+str(ref['Start'][i-1])+'-'+str(ref['End'][i-1]),'1->'+str(clone.paternal_cnvs[-1])])
                                changes.append(['normal',clone.name,'paternal','Mirror CNA',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(p_cnv)])
                                clone.maternal_cnvs.append(m_cnv)
                                clone.paternal_cnvs.append(p_cnv)
                                clone.changes.append('Mirror CNA')
                                clone.changes.append('Mirror CNA')
                                continue
                        
                        # normal case
                        if m_cnv == 0:
                            mtype = 'DEL'
                        else:
                            mtype = 'DUP'
                        changes.append(['normal',clone.name,'maternal',mtype,ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(m_cnv)])
                        if p_cnv == 0:
                            ptype = 'DEL'
                        else:
                            ptype = 'DUP'
                        changes.append(['normal',clone.name,'paternal',ptype,ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(p_cnv)])

                        clone.maternal_cnvs.append(m_cnv)
                        clone.paternal_cnvs.append(p_cnv)
                        clone.changes.append('REGULAR')
                    else: # normal
                        clone.maternal_cnvs.append(1)
                        clone.paternal_cnvs.append(1)
                        clone.changes.append('NONE')
            else:
                # first inherit from parent
                clone.maternal_cnvs = copy.deepcopy(clone.parent.maternal_cnvs)
                clone.paternal_cnvs = copy.deepcopy(clone.parent.paternal_cnvs)
                clone.changes = copy.deepcopy(clone.parent.changes)

                 # select the position for CNL_LOH, CNN_LOH, GOH and Mirror CNA
                cnl_loh_no = int(self.loh_cna_no/3)
                random_bins = random.sample(range(0, ref.shape[0]), self.loh_cna_no + self.goh_cna_no + self.mirror_cna_no)
                cnl_loh_bins = random_bins[:cnl_loh_no]
                cnn_loh_bins = random_bins[cnl_loh_no:self.loh_cna_no]
                goh_bins = random_bins[self.loh_cna_no:self.loh_cna_no + self.goh_cna_no]
                mirrored_cnv_bins = random_bins[self.loh_cna_no + self.goh_cna_no:]
                mirrored_cnv_flag = False

                for i, item in enumerate(clone.paternal_cnvs):
                    # if flag is ture, the previous bin has been process as Mirror CNA bin and skip it.
                    if mirrored_cnv_flag:
                        mirrored_cnv_flag = False
                        continue
                    
                    if clone.parent.changes[i] not in ['REGULAR', 'NONE']:
                        continue

                    current_chrom = ref['Chromosome'][i]
                    start = ref['Start'][i]
                    end = ref['End'][i]
                    m_sequence = maternal_genome[current_chrom][start-1:end]
                    p_sequence = paternal_genome[current_chrom][start-1:end]

                    m_parent_cnv = clone.maternal_cnvs[i]
                    p_parent_cnv = clone.paternal_cnvs[i]

                    if clone.parent.changes[i] == 'NONE':
                        # handle CNL_LOH: 1:0 or 0:1
                        if i in cnl_loh_bins:
                            # check heterozygosity
                            if m_sequence != p_sequence:
                                # cnl_cnv = utils.random_CNL()
                                cnl_cnv = 1

                                if random.random() < 0.5: # m:p = 1:0
                                    # if cnl_cnv != 1:
                                    #     changes.append([clone.parent.name,clone.name,'maternal','CNL_LOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(cnl_cnv)])
                                    changes.append([clone.parent.name,clone.name,'paternal','CNL_LOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(0)])
                                    clone.maternal_cnvs[i] = cnl_cnv
                                    clone.paternal_cnvs[i] = 0
                                else: # m:p = 0:1
                                    # if cnl_cnv != 1:
                                    #     changes.append([clone.parent.name,clone.name,'paternal','CNL_LOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(cnl_cnv)])
                                    changes.append([clone.parent.name,clone.name,'maternal','CNL_LOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(0)])
                                    clone.maternal_cnvs[i] = 0
                                    clone.paternal_cnvs[i] = cnl_cnv
                                clone.changes[i] = 'CNL_LOH'
                                continue
                            else:
                                continue
                        
                        # handle CNN_LOH: 2:0 or 0:2
                        if i in cnn_loh_bins:
                            # check heterozygosity
                            if m_sequence != p_sequence:
                                cnn_cnv = utils.random_mirrored_cnv()

                                if random.random() < 0.5: # m:p = 2:0
                                    if cnn_cnv == 2:
                                        changes.append([clone.parent.name,clone.name,'maternal','CNN_LOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(cnn_cnv)])
                                        changes.append([clone.parent.name,clone.name,'paternal','CNN_LOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(0)])
                                    else:
                                        changes.append([clone.parent.name,clone.name,'maternal','CNG_LOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(cnn_cnv)])
                                        changes.append([clone.parent.name,clone.name,'paternal','CNG_LOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(0)])
                                    clone.maternal_cnvs[i] = cnn_cnv
                                    clone.paternal_cnvs[i] = 0
                                else: # m:p = 0:1
                                    if cnn_cnv == 2:
                                        changes.append([clone.parent.name,clone.name,'maternal','CNN_LOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(0)])
                                        changes.append([clone.parent.name,clone.name,'paternal','CNN_LOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(cnn_cnv)])
                                    else:
                                        changes.append([clone.parent.name,clone.name,'maternal','CNG_LOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(cnn_cnv)])
                                        changes.append([clone.parent.name,clone.name,'paternal','CNG_LOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(0)])
                                    clone.maternal_cnvs[i] = 0
                                    clone.paternal_cnvs[i] = cnn_cnv
                                if cnn_cnv == 2:
                                    clone.changes[i] = 'CNN_LOH'
                                else:
                                    clone.changes[i] = 'CNG_LOH'
                                continue
                            else:
                                continue
                    
                        # handle GOH
                        if i in goh_bins:
                            # check heterozygosity
                            if m_sequence == p_sequence:
                                m_cnv = utils.random_WGD()
                                p_cnv = utils.random_WGD()
                                changes.append([clone.parent.name,clone.name,'maternal','GOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(m_cnv)])
                                changes.append([clone.parent.name,clone.name,'paternal','GOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(p_cnv)])
                                clone.maternal_cnvs[i] = m_cnv
                                clone.paternal_cnvs[i] = p_cnv
                                clone.changes[i] = 'GOH'
                                continue
                            else:
                                continue
                    
                        # handle Mirror CNA
                        if i in mirrored_cnv_bins:
                            # make sure the next bin located in same chromosome
                            if i+1 >= len(clone.paternal_cnvs) or ref['Chromosome'][i] != ref['Chromosome'][i+1] or clone.parent.changes[i+1] != 'NONE':
                                continue

                            # generate Mirror CNA number
                            total_cnv = utils.random_mirrored_cnv()
                            cnv1 = random.randint(0,total_cnv)
                            while cnv1 == total_cnv/2:
                                cnv1 = random.randint(0, total_cnv)
                            cnv2 = total_cnv - cnv1
                            if random.random() < 0.5: # m:p = cnv1:cnv2
                                changes.append([clone.parent.name,clone.name,'maternal','Mirror CNA',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(cnv1)])
                                changes.append([clone.parent.name,clone.name,'maternal','Mirror CNA',ref['Chromosome'][i+1]+':'+str(ref['Start'][i+1])+'-'+str(ref['End'][i+1]),'1->'+str(cnv2)])
                                changes.append([clone.parent.name,clone.name,'paternal','Mirror CNA',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(cnv2)])
                                changes.append([clone.parent.name,clone.name,'paternal','Mirror CNA',ref['Chromosome'][i+1]+':'+str(ref['Start'][i+1])+'-'+str(ref['End'][i+1]),'1->'+str(cnv1)])
                                clone.maternal_cnvs[i] = cnv1
                                clone.paternal_cnvs[i] = cnv2
                                clone.maternal_cnvs[i+1] = cnv2
                                clone.paternal_cnvs[i+1] = cnv1
                            else: # m:p = cnv2:cnv1
                                changes.append([clone.parent.name,clone.name,'maternal','Mirror CNA',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(cnv2)])
                                changes.append([clone.parent.name,clone.name,'maternal','Mirror CNA',ref['Chromosome'][i+1]+':'+str(ref['Start'][i+1])+'-'+str(ref['End'][i+1]),'1->'+str(cnv1)])
                                changes.append([clone.parent.name,clone.name,'paternal','Mirror CNA',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),'1->'+str(cnv1)])
                                changes.append([clone.parent.name,clone.name,'paternal','Mirror CNA',ref['Chromosome'][i+1]+':'+str(ref['Start'][i+1])+'-'+str(ref['End'][i+1]),'1->'+str(cnv2)])
                                clone.maternal_cnvs[i] = cnv2
                                clone.paternal_cnvs[i] = cnv1
                                clone.maternal_cnvs[i+1] = cnv1
                                clone.paternal_cnvs[i+1] = cnv2
                            mirrored_cnv_flag = True
                            clone.changes[i] = 'Mirror CNA'
                            clone.changes[i+1] = 'Mirror CNA'
                            continue
                    
                    # regular situation
                    if random.random() > cutoff: # 20% cnv
                        m_cnv = utils.random_cnv()
                        p_cnv = utils.random_cnv()
                        if m_parent_cnv == 0:
                            m_cnv = 0
                        else:
                            m_cnv = random.randint(m_parent_cnv, max(5, m_parent_cnv+1))
                        
                        if p_parent_cnv == 0:
                            p_cnv = 0
                        else:
                            p_cnv = random.randint(p_parent_cnv, max(5, p_parent_cnv+1))
                        
                        # check whether is CNL_LOH
                        if (m_sequence != p_sequence) and ((m_cnv == 0 and p_cnv !=0) or (m_cnv != 0 and p_cnv ==0)):
                            if m_cnv == 1 or p_cnv == 1:
                                changes.append([clone.parent.name,clone.name,'maternal','CNL_LOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),str(m_parent_cnv)+'->'+str(m_cnv)])
                                changes.append([clone.parent.name,clone.name,'paternal','CNL_LOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),str(p_parent_cnv)+'->'+str(p_cnv)])
                                clone.changes[i] = 'CNL_LOH'
                            elif m_cnv == 2 or p_cnv == 2:
                                changes.append([clone.parent.name,clone.name,'maternal','CNN_LOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),str(m_parent_cnv)+'->'+str(m_cnv)])
                                changes.append([clone.parent.name,clone.name,'paternal','CNN_LOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),str(p_parent_cnv)+'->'+str(p_cnv)])
                                clone.changes[i] = 'CNN_LOH'
                            else:
                                changes.append([clone.parent.name,clone.name,'maternal','CNG_LOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),str(m_parent_cnv)+'->'+str(m_cnv)])
                                changes.append([clone.parent.name,clone.name,'paternal','CNG_LOH',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),str(p_parent_cnv)+'->'+str(p_cnv)])
                                clone.changes[i] = 'CNG_LOH'

                            clone.maternal_cnvs[i] = m_cnv
                            clone.paternal_cnvs[i] = p_cnv
                            continue
                        
                        # check mirrored CNV
                        if clone.changes and clone.changes[i-1] in ['REGULAR', 'NONE']:
                            if m_cnv != p_cnv and m_cnv == clone.paternal_cnvs[-1] and clone.maternal_cnvs[-1] == p_cnv:
                                clone.maternal_cnvs[i] = m_cnv
                                clone.paternal_cnvs[i] = p_cnv
                                if clone.changes[i-1] == 'REGULAR':
                                    changes.pop()
                                    changes.pop()
                                changes.append([clone.parent.name,clone.name,'maternal','Mirror CNA',ref['Chromosome'][i-1]+':'+str(ref['Start'][i-1])+'-'+str(ref['End'][i-1]),str(clone.parent.maternal_cnvs[i-1])+'->'+str(clone.maternal_cnvs[i-1])])
                                changes.append([clone.parent.name,clone.name,'maternal','Mirror CNA',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),str(m_parent_cnv)+'->'+str(m_cnv)])
                                changes.append([clone.parent.name,clone.name,'paternal','Mirror CNA',ref['Chromosome'][i-1]+':'+str(ref['Start'][i-1])+'-'+str(ref['End'][i-1]),str(clone.parent.paternal_cnvs[i-1])+'->'+str(clone.paternal_cnvs[i-1])])
                                changes.append([clone.parent.name,clone.name,'paternal','Mirror CNA',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),str(p_parent_cnv)+'->'+str(p_cnv)])
                                clone.changes[i-1] = 'Mirror CNA'
                                clone.changes[i] = 'Mirror CNA'
                                continue

                        if m_cnv != m_parent_cnv:
                            if m_cnv == 0:
                                changes.append([clone.parent.name,clone.name,'maternal','DEL',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),str(m_parent_cnv)+'->'+str(m_cnv)])
                            else:
                                changes.append([clone.parent.name,clone.name,'maternal','DUP',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),str(m_parent_cnv)+'->'+str(m_cnv)])
                        
                        if p_cnv != p_parent_cnv:
                            if p_cnv == 0:
                                changes.append([clone.parent.name,clone.name,'paternal','DEL',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),str(p_parent_cnv)+'->'+str(p_cnv)])
                            else:
                                changes.append([clone.parent.name,clone.name,'paternal','DUP',ref['Chromosome'][i]+':'+str(ref['Start'][i])+'-'+str(ref['End'][i]),str(p_parent_cnv)+'->'+str(p_cnv)])
                        clone.maternal_cnvs[i] = m_cnv
                        clone.paternal_cnvs[i] = p_cnv
                        clone.changes[i] = 'REGULAR'
            
            ref[clone.name+'_maternal_cnas'] = clone.maternal_cnvs
            ref[clone.name+'_paternal_cnas'] = clone.paternal_cnvs
            queue.extend(clone.children)

        return ref, changes, maternal_genome, paternal_genome

    def _generate_fasta_for_each_clone(self, job):
        (clone, ref, changes, maternal_genome, paternal_genome, outdir) = job

        gfasta_bar.progress(advance=False, msg="Start generating fasta file for {}".format(clone.name))

        clone.maternal_fasta = os.path.join(outdir, clone.name+'_maternal.fasta')
        clone.paternal_fasta = os.path.join(outdir, clone.name+'_paternal.fasta')
        
        with open(clone.maternal_fasta, 'w') as m_output:
            with open(clone.paternal_fasta, 'w') as p_output:
                for chrom in maternal_genome.keys():
                    m_output.write('>'+chrom+'\n')
                    p_output.write('>'+chrom+'\n')
                    chrom_ref = ref[ref['Chromosome'] == chrom]
                    for index, row in chrom_ref.iterrows():
                        m_cnv = int(row[clone.name+'_maternal_cnas'])
                        p_cnv = int(row[clone.name+'_paternal_cnas'])
                        start = int(row['Start'])
                        end = int(row['End'])

                        # handle CNN_LOH
                        segment = chrom + ':' + str(start) + '-' + str(end)
                        # logging.info(clone.name, segment)
                        all_types = changes[(changes['Child']==clone.name) & (changes['Segment']==segment)]['Type'].tolist()
                        if len(all_types):
                            cnv_type = all_types[0]
                        else:
                            cnv_type = 'REGULAR'
                        m_sequence = maternal_genome[chrom][start-1:end]
                        p_sequence = paternal_genome[chrom][start-1:end]
                        # if cnv_type == 'CNN_LOH':
                        #     if m_cnv != 0:
                        #         new_m_cnv = random.randint(1, m_cnv -1)
                        #         new_p_cnv = m_cnv - new_m_cnv
                        #         cnv_m_sequence = m_sequence * new_m_cnv
                        #         cnv_p_sequence = m_sequence * new_p_cnv
                        #     else:
                        #         new_m_cnv = random.randint(1, p_cnv -1)
                        #         new_p_cnv = p_cnv - new_m_cnv
                        #         cnv_m_sequence = p_sequence * new_m_cnv
                        #         cnv_p_sequence = p_sequence * new_p_cnv
                        if cnv_type == 'GOH':
                            seq_len = len(m_sequence)
                            random_snp_no = min(1, int(self.bin_size * self.snp_ratio))
                            random_snps = {snp : random.sample(['A','T','C','G'], 2) for snp in random.sample(range(seq_len), random_snp_no)}
                            new_m_sequence = ''
                            new_p_sequence = ''
                            snp_pos = random_snps.keys()
                            for pos in range(len(m_sequence)):
                                if pos in snp_pos:
                                    new_m_sequence += random_snps[pos][0]
                                    new_p_sequence += random_snps[pos][1]
                                else:
                                    new_m_sequence += m_sequence[pos]
                                    new_p_sequence += p_sequence[pos]
                            # cnv_m_sequence = new_m_sequence * m_cnv
                            # cnv_p_sequence = new_p_sequence * p_cnv
                            cnv_m_sequence = new_m_sequence
                            cnv_p_sequence = new_p_sequence
                        else:   
                            # cnv_m_sequence = m_sequence * m_cnv
                            # cnv_p_sequence = p_sequence * p_cnv
                            cnv_m_sequence = m_sequence
                            cnv_p_sequence = p_sequence
                        m_output.write(cnv_m_sequence)
                        p_output.write(cnv_p_sequence)
                        clone.maternal_fasta_length += len(cnv_m_sequence)
                        clone.paternal_fasta_length += len(cnv_p_sequence)
                    m_output.write('\n')
                    p_output.write('\n')

        # merge maternal and paternal fasta
        # clone.fasta = os.path.join(outdir, clone.name+'.fasta')
        # command = """sed '/^>chr/ s/$/-A/' {0} > {1} && sed '/^>chr/ s/$/-B/' {2} >> {1}""".format(clone.maternal_fasta, clone.fasta, clone.paternal_fasta)
        # utils.runcmd(command, self.outdir)
        gfasta_bar.progress(advance=True, msg="Finish generating fasta file for {}".format(clone.name))
        return (clone)
    
    def _find_mirrored_clones(self, cnv_profile):
        """
        Identify rows with mirrored-clone CNAs in a given CNV CSV file, excluding cases where the 
        allele-specific CNVs are equal (e.g., 1|1, 2|2).

        Parameters:
            cnv profile cnv

        Returns:
            pd.DataFrame: A DataFrame containing rows with mirrored-clone CNAs.
        """
        # Read the CSV file into a pandas DataFrame
        df = cnv_profile
        
        # Ensure the first three columns are Chromosome, Start, and End
        required_columns = ['Chromosome', 'Start', 'End']
        if not all(col in df.columns[:3] for col in required_columns):
            raise ValueError("The first three columns must be 'Chromosome', 'Start', and 'End'")
        
        # Extract clone columns (columns after the first three)
        clone_columns = df.columns[3:]
        
        # Prepare a list to store rows with mirrored-clone CNAs
        mirrored_rows = []
        
        # Iterate through each row in the DataFrame
        for _, row in df.iterrows():
            # Iterate through all pairs of clone columns to check for mirrored CNAs
            for i in range(len(clone_columns)):
                for j in range(i + 1, len(clone_columns)):
                    clone1 = row[clone_columns[i]]
                    clone2 = row[clone_columns[j]]
                    
                    # Split the allele-specific CNV (e.g., "1|2") into two haplotypes
                    try:
                        haplotype1 = tuple(map(int, clone1.split('|')))
                        haplotype2 = tuple(map(int, clone2.split('|')))
                    except ValueError:
                        raise ValueError(f"Invalid allele-specific CNV format in row: {row}")

                    # Check if they are mirrored (e.g., (1, 2) and (2, 1)),
                    # and exclude cases where both haplotypes are equal (e.g., (1, 1) or (2, 2))
                    if haplotype1 == haplotype2[::-1] and haplotype1[0] != haplotype1[1]:
                        mirrored_rows.append({
                            'Chromosome': row['Chromosome'],
                            'Start': row['Start'],
                            'End': row['End'],
                            'Clone1': clone_columns[i],
                            'Clone2': clone_columns[j],
                            'Clone1_CNA': clone1,
                            'Clone2_CNA': clone2
                        })
        
        # Convert the results into a DataFrame
        mirrored_df = pd.DataFrame(mirrored_rows)
        return mirrored_df

    def _out_cnv_profile(self, root, ref, changes, outdir):
        # out cnv profile csv
        df = ref[['Chromosome', 'Start', 'End']]
        queue = deque([root])
        while queue:
            clone = queue.popleft()
            df[clone.name] = ref[clone.name+'_maternal_cnas'].astype(str) + '|' + ref[clone.name+'_paternal_cnas'].astype(str)
            queue.extend(clone.children)
        df.to_csv(os.path.join(outdir, 'cna_profile.csv'), index=False)



        # out maternal cnv matrix
        indexes = ref['Chromosome'] + ':' + ref['Start'].astype(str) + '-' + ref['End'].astype(str)
        m_cnv = ref.filter(like='maternal_cnas')
        m_cnv.index = indexes
        m_cnv.to_csv(os.path.join(outdir, 'maternal_cna_matrix.csv'))

        # out paternal cnv matrix
        p_cnv = ref.filter(like='paternal_cnas')
        p_cnv.index = indexes
        p_cnv.to_csv(os.path.join(outdir, 'paternal_cna_matrix.csv'))

        # out changes profile
        columns = ['Parent', 'Child', 'Haplotype', 'Type', 'Segment', 'Change']
        change_df = pd.DataFrame(data=changes, columns=columns)
        change_df.to_csv(os.path.join(outdir, 'changes.csv'), index=False)
        ref.to_csv(os.path.join(outdir, 'reference.csv'), index=False)
        return change_df, df

    def _merge_fasta_for_each_clone(self, root, outdir):
        # merge fasta for each clone
        queue = deque([root])
        while queue:
            clone = queue.popleft()
            clone.fasta = os.path.join(outdir, clone.name+'.fasta')
            command = """sed '/^>chr/ s/$/-A/' {0} > {1} && sed '/^>chr/ s/$/-B/' {2} >> {1}""".format(clone.maternal_fasta, clone.fasta, clone.paternal_fasta)
            code = os.system(command)
            queue.extend(clone.children)

    def _wgsim_process(self, pe_reads, fasta, fq1, fq2):
        #where yyy is the read length, zzz is the error rate and $xxx * $yyy = 10000000.
        command = self.wgsim + " -e {0} -d {1} -s 35 -N {2} -1 {3} -2 {3} -r0 -R0 -X0 {4} {5} {6}".format(self.error_rate,self.insertion_size,pe_reads,self.reads_len,fasta,fq1,fq2)
        wgsim_log = os.path.join(self.outdir, 'log/wgsim_log.txt')
        utils.runcmd(command, wgsim_log)

    def _generate_fastq_for_each_clone_bk(self, job):
        (clone, outdir) = job
        gfastq_bar.progress(advance=False, msg="Start generating fastq file for {}".format(clone.name))

        # generate maternal fastq
        pe_reads = round(clone.maternal_fasta_length*self.clone_coverage/(self.reads_len*4), 6)
        fq1 = os.path.join(outdir, clone.name+'_maternal_r1.fq')
        fq2 = os.path.join(outdir, clone.name+'_maternal_r2.fq')
        clone.fq1 = fq1
        clone.fq2 = fq2

        self._wgsim_process(pe_reads,clone.maternal_fasta,fq1,fq2)

        # generae paternal fastq
        pe_reads = round(clone.paternal_fasta_length*self.clone_coverage/(self.reads_len*4), 6)
        fq1 = os.path.join(outdir, clone.name+'_paternal_r1.fq')
        fq2 = os.path.join(outdir, clone.name+'_paternal_r2.fq')
        clone.fq1 = fq1
        clone.fq2 = fq2

        self._wgsim_process(pe_reads,clone.paternal_fasta,fq1,fq2)
        gfastq_bar.progress(advance=True, msg="Finish generating fastq file for {}".format(clone.name))

    def _run_wgsim_for_region(self, pe_reads, fasta_region_file, temp_fq1, temp_fq2):
        """A helper function to run wgsim for a single small region."""
        if pe_reads <= 0:
            return
        command = self.wgsim + " -e {0} -d {1} -s 35 -N {2} -1 {3} -2 {3} -r0 -R0 -X0 {4} {5} {6}".format(self.error_rate,self.insertion_size,pe_reads,self.reads_len,fasta_region_file,temp_fq1,temp_fq2)
        wgsim_log = os.path.join(self.outdir, 'log/wgsim_log.txt')
        utils.runcmd(command, wgsim_log)

    def _generate_fastq_for_each_clone(self, job):
        """
        Generates biased FASTQ files for a single clone, keeping maternal and paternal files separate.
        """
        (clone, chrom_ref_df, outdir) = job
        gfastq_bar.progress(advance=False, msg="Start generating fastq file for {}".format(clone.name))
        
        temp_dir = os.path.join(self.outdir, 'tmp', clone.name)
        os.makedirs(temp_dir, exist_ok=True)

        # 定义最终的母本和父本FASTQ文件名
        final_maternal_fq1 = os.path.join(outdir, f'{clone.name}_maternal_r1.fq')
        final_maternal_fq2 = os.path.join(outdir, f'{clone.name}_maternal_r2.fq')
        final_paternal_fq1 = os.path.join(outdir, f'{clone.name}_paternal_r1.fq')
        final_paternal_fq2 = os.path.join(outdir, f'{clone.name}_paternal_r2.fq')

        final_files = [final_maternal_fq1, final_maternal_fq2, final_paternal_fq1, final_paternal_fq2]
        for f in final_files:
            open(f, 'w').close()

        Aa, Bb = utils.get_alpha_beta(self.lorenz_x, self.lorenz_y)
        # self.log(f"  > Lorenz curve parameters for {clone.name}: Alpha={Aa:.2f}, Beta={Bb:.2f}")

        for chrom in sorted(chrom_ref_df['Chromosome'].unique()):
            # self.log(f"  > Processing chromosome {chrom}...")
            chrom_bins_df = chrom_ref_df[chrom_ref_df['Chromosome'] == chrom].reset_index(drop=True)
            num_windows = len(chrom_bins_df)
            
            interval = max(10, num_windows // 100)
            cov_scalers = utils.gen_coverage(num_windows=num_windows, interval=interval, Aa=Aa, Bb=Bb)

            if len(cov_scalers) != num_windows:
                self.log(f"    ! Warning: Scaler length mismatch on {chrom}. Got {len(cov_scalers)}, expected {num_windows}. Skipping chrom.", level='WARN')
                continue

            for index, row in chrom_bins_df.iterrows():
                start, end = int(row['Start']), int(row['End'])
                bin_size = end - start
                
                m_cnv = int(row[f'{clone.name}_maternal_cnas'])
                p_cnv = int(row[f'{clone.name}_paternal_cnas'])
                total_cn = m_cnv + p_cnv
                
                if total_cn == 0:
                    continue
                
                cov_scaler = cov_scalers[index]
                
                base_diploid_pe_reads = (self.clone_coverage * bin_size) / (self.reads_len * 2)
                adjusted_total_pe_reads = base_diploid_pe_reads * (total_cn / 2) * cov_scaler
                total_reads_sampled = poisson.rvs(adjusted_total_pe_reads)
                m_pe_reads = round(total_reads_sampled * (m_cnv / total_cn))
                p_pe_reads = round(total_reads_sampled * (p_cnv / total_cn))

                region_str = f"{chrom}:{start}-{end}"

                if m_pe_reads > 0:
                    region_fasta = os.path.join(temp_dir, "m_region.fa")
                    temp_fq1 = os.path.join(temp_dir, "m_r1.fq")
                    temp_fq2 = os.path.join(temp_dir, "m_r2.fq")
                    
                    samtools_log = os.path.join(self.outdir, 'log/samtools_log.txt')
                    command = f"{self.samtools} faidx {clone.maternal_fasta} {region_str} > {region_fasta}"
                    utils.runcmd(command, samtools_log)
                    
                    self._run_wgsim_for_region(m_pe_reads, region_fasta, temp_fq1, temp_fq2)
                    
                    if os.path.exists(temp_fq1):
                        os.system(f'cat {temp_fq1} >> {final_maternal_fq1}')
                        os.system(f'cat {temp_fq2} >> {final_maternal_fq2}')

                if p_pe_reads > 0:
                    region_fasta = os.path.join(temp_dir, "p_region.fa")
                    temp_fq1 = os.path.join(temp_dir, "p_r1.fq")
                    temp_fq2 = os.path.join(temp_dir, "p_r2.fq")

                    samtools_log = os.path.join(self.outdir, 'log/samtools_log.txt')
                    command = f"{self.samtools} faidx {clone.paternal_fasta} {region_str} > {region_fasta}"
                    utils.runcmd(command, samtools_log)
                    
                    self._run_wgsim_for_region(p_pe_reads, region_fasta, temp_fq1, temp_fq2)
                    
                    if os.path.exists(temp_fq1):
                        os.system(f'cat {temp_fq1} >> {final_paternal_fq1}')
                        os.system(f'cat {temp_fq2} >> {final_paternal_fq2}')
        gfastq_bar.progress(advance=True, msg="Finish generating fastq file for {}".format(clone.name))

    def run_readcount_generation(self, clones, chrom_ref_df, lorenz_x, lorenz_y):
        """
        Generates a read count matrix directly without creating FASTQ files.
        """
        print("\n--- Starting Read Count Matrix Generation Workflow ---")
        
        Aa, Bb = utils.get_alpha_beta(lorenz_x, lorenz_y)
        print(f"Global Lorenz curve parameters: Alpha={Aa:.2f}, Beta={Bb:.2f}")

        output_file = os.path.join(self.outdir, 'readcounts.tsv')
        all_counts = []

        for chrom in sorted(chrom_ref_df['Chromosome'].unique()):
            print(f"  > Processing chromosome {chrom} for read counts...")
            chrom_bins_df = chrom_ref_df[chrom_ref_df['Chromosome'] == chrom].reset_index(drop=True)
            num_windows = len(chrom_bins_df)
            
            interval = max(10, num_windows // 100)
            cov_scalers = utils.gen_coverage(num_windows=num_windows, interval=interval, Aa=Aa, Bb=Bb)
            
            if len(cov_scalers) != num_windows:
                print(f"    ! Warning: Scaler length mismatch on {chrom}. Skipping.")
                continue

            for index, row in chrom_bins_df.iterrows():
                start, end = int(row['Start']), int(row['End'])
                bin_size = end - start
                cov_scaler = cov_scalers[index]
                
                base_diploid_pe_reads = (self.clone_coverage * bin_size) / (self.reads_len * 2)

                for clone in clones:
                    m_cnv = int(row[f'{clone.name}_maternal_cnas'])
                    p_cnv = int(row[f'{clone.name}_paternal_cnas'])
                    total_cn = m_cnv + p_cnv
                    
                    if total_cn == 0:
                        m_count, p_count = 0, 0
                    else:
                        adjusted_total_pe_reads = base_diploid_pe_reads * (total_cn / 2) * cov_scaler
                        total_reads_sampled = poisson.rvs(adjusted_total_pe_reads)
                        m_count = round(total_reads_sampled * (m_cnv / total_cn))
                        p_count = round(total_reads_sampled * (p_cnv / total_cn))
                    
                    all_counts.append({
                        'CELL': clone.name,
                        'chrom': chrom,
                        'start': start,
                        'end': end,
                        'Acount': m_count,
                        'Bcount': p_count
                    })

        count_df = pd.DataFrame(all_counts)
        count_df.to_csv(output_file, sep='\t', index=False)
        print(f"--- Read Count Matrix Generation Complete. Output: {output_file} ---")

    def _alignment_for_each_clone(self, job):
        (clone, fastq_dir, bam_dir, log_dir) = job

        bam_file = os.path.join(bam_dir, clone+".bam")
        sorted_bam_file = os.path.join(bam_dir, clone+".sorted.bam")
        samtools_log = os.path.join(log_dir, 'samtools_log.txt')
        bwa_log = os.path.join(log_dir, 'bwa_log.txt')

        # check bwa reference index files
        def check_and_index_bwa(reference):
            """
            check bwa reference index files
            """
            index_extensions = [".amb", ".ann", ".bwt", ".pac", ".sa"]
            
            all_index_exist = all(os.path.exists(reference + ext) for ext in index_extensions)
            
            if not all_index_exist:
                cmd = f"bwa index {reference}"
                utils.runcmd(cmd, bwa_log)

        check_and_index_bwa(self.ref_genome)

        # store tmp files
        tmp_files = []
        
        #run bwa for maternal
        align_bar.progress(advance=False, msg="BWA alignment for {}".format(clone))
        fq1 = os.path.join(fastq_dir, clone + "_maternal_r1.fq")
        fq2 = os.path.join(fastq_dir, clone + "_maternal_r2.fq")
        sam_file = os.path.join(bam_dir, clone+"_maternal.sam")
        tmp_files.append(sam_file)
        command = "{0} mem -M -t {1} {2} {3} {4} > {5}".format(self.bwa, self.thread, self.ref_genome, fq1, fq2, sam_file)
        utils.runcmd(command, bwa_log)

        fq1 = os.path.join(fastq_dir, clone + "_paternal_r1.fq")
        fq2 = os.path.join(fastq_dir, clone + "_paternal_r2.fq")
        sam_file = os.path.join(bam_dir, clone+"_paternal.sam")
        tmp_files.append(sam_file)
        command = "{0} mem -M -t {1} {2} {3} {4} > {5}".format(self.bwa, self.thread, self.ref_genome, fq1, fq2, sam_file)
        utils.runcmd(command, bwa_log)

        # samtools sam to bam
        align_bar.progress(advance=False, msg="Samtools sam to bam for {}".format(clone))
        sam_file = os.path.join(bam_dir, clone+"_maternal.sam")
        bam_file = os.path.join(bam_dir, clone+"_maternal.bam")
        tmp_files.append(bam_file)
        command = "{0} view -@ {1} -bS {2} > {3}".format(self.samtools, self.thread, sam_file, bam_file)
        utils.runcmd(command, samtools_log)

        sam_file = os.path.join(bam_dir, clone+"_paternal.sam")
        bam_file = os.path.join(bam_dir, clone+"_paternal.bam")
        tmp_files.append(bam_file)
        command = "{0} view -@ {1} -bS {2} > {3}".format(self.samtools, self.thread, sam_file, bam_file)
        utils.runcmd(command, samtools_log)

        align_bar.progress(advance=False, msg="Samtools sort bam for {}".format(clone))
        bam_file = os.path.join(bam_dir, clone+"_maternal.bam")
        sorted_bam_file = os.path.join(bam_dir, clone+"_maternal.sorted.bam")
        tmp_files.append(sorted_bam_file)
        command = "{0} sort -@ {1} {2} -o {3}".format(self.samtools, self.thread, bam_file, sorted_bam_file)
        utils.runcmd(command, samtools_log)

        bam_file = os.path.join(bam_dir, clone+"_paternal.bam")
        sorted_bam_file = os.path.join(bam_dir, clone+"_paternal.sorted.bam")
        tmp_files.append(sorted_bam_file)
        command = "{0} sort -@ {1} {2} -o {3}".format(self.samtools, self.thread, bam_file, sorted_bam_file)
        utils.runcmd(command, samtools_log)

        align_bar.progress(advance=False, msg="Samtools merge maternal and paternal bam for {}".format(clone))
        clone_bam = os.path.join(bam_dir, clone+".bam")
        m_sorted_bam_file = os.path.join(bam_dir, clone+"_maternal.sorted.bam")
        p_sorted_bam_file = os.path.join(bam_dir, clone+"_paternal.sorted.bam")
        command = "{0} merge -@ {1} -f {2} {3} {4}".format(self.samtools, self.thread, clone_bam, m_sorted_bam_file, p_sorted_bam_file)
        utils.runcmd(command, samtools_log)

        # clean sam and unsorted bam
        for tmp_file in tmp_files:
            if os.path.exists(tmp_file) and os.path.exists(clone_bam):
                os.remove(tmp_file)
        align_bar.progress(advance=True, msg="Finish alignment process for {}".format(clone))

    def _downsampling_cell_bam(self, job):
        (ratio, clone_bam_file, cell_bam_file, log_dir) = job

        samtools_log = os.path.join(log_dir, 'samtools_log.txt')
        cell_name = os.path.splitext(os.path.basename(cell_bam_file))[0]

        downsam_bar.progress(advance=False, msg="Downsampling cell bam for {}".format(cell_name))
        command = "{0} view -@ {1} -b -s {2} {3} > {4}".format(self.samtools, self.thread, ratio, clone_bam_file, cell_bam_file)
        utils.runcmd(command, samtools_log)
        downsam_bar.progress(advance=True, msg="Finish downsampling cell bam for {}".format(cell_name))

    def _process_cell_bam(self, job):
        (cell, dcell, dtmp, dlog) = job

        bam_file = os.path.join(dcell, cell + ".bam")
        query_sorted_bam_file = os.path.join(dcell, cell + ".query.bam")
        fixed_sorted_bam_file = os.path.join(dcell, cell + ".fixed.bam")
        sorted_bam_file = os.path.join(dcell, cell + ".sorted.bam")
        dedup_bam_file = os.path.join(dcell, cell + ".dedup.bam")
        rg_dedup_bam_file = os.path.join(dcell, cell + ".rg.bam")
        samtools_log = os.path.join(dlog, 'samtools_log.txt')
        # picard_log = os.path.join(dlog, 'picard_log.txt')
        tmp_files = [bam_file, query_sorted_bam_file, fixed_sorted_bam_file, sorted_bam_file, rg_dedup_bam_file]

        # run samtools sort bam
        pbam_bar.progress(advance=False, msg="Samtools sort by query name for {}".format(cell))
        command = '{0} sort -@ {1} -n {2} -o {3}'.format(self.samtools, self.thread, bam_file, query_sorted_bam_file)
        utils.runcmd(command, samtools_log)

        pbam_bar.progress(advance=False, msg="Samtools fixmate for {}".format(cell))
        command = '{0} fixmate -@ {1} -O bam -m {2} {3}'.format(self.samtools, self.thread, query_sorted_bam_file, fixed_sorted_bam_file)
        utils.runcmd(command, samtools_log)
        
        pbam_bar.progress(advance=False, msg="Samtools addreplacerg for {}".format(cell))
        command = """{0} addreplacerg -@ {1} \
                    -r ID:{2} \
                    -r LB:genome \
                    -r PL:ILLUMINA \
                    -r PU:HCDSIM \
                    -r SM:{2} \
                    -o {3} \
                    {4}""".format(self.samtools, self.thread, cell, rg_dedup_bam_file, fixed_sorted_bam_file)
        utils.runcmd(command, samtools_log)

        pbam_bar.progress(advance=False, msg="Samtools sort by coordinates for {}".format(cell))
        command = '{0} sort -@ {1} {2} -o {3}'.format(self.samtools, self.thread, rg_dedup_bam_file, sorted_bam_file)
        utils.runcmd(command, samtools_log)

        # run samtools remove dupliations
        pbam_bar.progress(advance=False, msg="Samtools markdup for {}".format(cell))
        command = '{0} markdup -@ {1} -r {2} {3}'.format(self.samtools, self.thread, sorted_bam_file, dedup_bam_file)
        utils.runcmd(command, samtools_log)

        pbam_bar.progress(advance=False, msg="Samtools index for {}".format(cell))
        command = "{0} index -@ {1} {2}".format(self.samtools, self.thread, dedup_bam_file)
        utils.runcmd(command, samtools_log)

        # clean tmp bam file
        for tmp_file in tmp_files:
            if os.path.exists(tmp_file) and os.path.exists(dedup_bam_file):
                os.remove(tmp_file)
        
        # rename cell bam
        os.rename(dedup_bam_file, bam_file)
        os.rename(dedup_bam_file + '.bai', bam_file + '.bai')
        pbam_bar.progress(advance=True, msg="Finish cell bam processing for {}".format(cell))

    def _call_bedtools(self, cell_bam, ref_bed, coverage_file):
        bedtools_log = os.path.join(self.outdir, 'log/bedtools_log.txt')
        samtools_log = os.path.join(self.outdir, 'log/samtools_log.txt')
        # check bai file
        if not os.path.exists(cell_bam + '.bai'):
            cmd = f"{self.samtools} index -@ {self.thread} {cell_bam}"
            utils.runcmd(cmd, samtools_log)

        # use bedtools to get coverage
        cmd = f"{self.bedtools} multicov -bams {cell_bam} -bed {ref_bed} -q 60 > {coverage_file}"
        utils.runcmd(cmd, bedtools_log)

    def _get_coverage_for_each_cell(self, job):
        (cell, dcell, dtmp, dlog) = job

        cell_bam = os.path.join(dcell, cell + '.bam')
        coverage_file = os.path.join(dtmp, cell + '.coverage.bed')
        ref_bed = os.path.join(dtmp, 'rdr_reference.bed')

        # use bedtools to get coverage for each cell bam
        cov_bar.progress(advance=False, msg="Counting germinal SNPs on {}".format(cell))
        self._call_bedtools(cell_bam, ref_bed, coverage_file)
        cov_bar.progress(advance=True, msg="Finish germinal SNPs on {}".format(cell))

    def _call_bcftools(self, snps_bed_file, bam_file, vcf_file, count_file):
        bcftools_log = os.path.join(self.outdir, 'log/bcftools_log.txt')

        # bcftools mpileup vcf file
        cmd = f"{self.bcftools} mpileup -f {self.ref_genome} -R {snps_bed_file} --skip-indels -a INFO/AD -Ou {bam_file} | bcftools view -Oz -o {vcf_file}"        
        utils.runcmd(cmd, bcftools_log)
        # bcftools index vcf file
        cmd = f"{self.bcftools} index {vcf_file}"
        utils.runcmd(cmd, bcftools_log)
        #bcftools query
        cmd = f"{self.bcftools} query -f '%CHROM\t%POS\t%REF\t%ALT\t[%AD]\n' {vcf_file} > {count_file}"
        utils.runcmd(cmd, bcftools_log)
    
    def _call_bcftools_for_each_cell(self, job):
        (cell, dcell, dtmp, dlog) = job

        cell_bam = os.path.join(dcell, cell + '.bam')
        snp_bed = os.path.join(dtmp, 'snps.bed')
        cell_vcf_file = os.path.join(dtmp, cell + '.vcf.gz')
        cell_count_file = os.path.join(dtmp, cell + '.count.bed')

        bcftools_bar.progress(advance=False, msg="Counting germinal SNPs on {}".format(cell))
        self._call_bcftools(snp_bed, cell_bam, cell_vcf_file, cell_count_file)
        bcftools_bar.progress(advance=True, msg="Finish germinal SNPs on {}".format(cell))

    @utils.log_runtime
    def gprofile(self):
        self.log('Setting directories', level='PROGRESS')
        dprofile, dfasta, dfastq, dclone, dcell, dbarcode, drdr, dbaf, dtmp, dlog = self.setup_dir()

        # set related files
        m_fasta = os.path.join(dfasta, 'normal_maternal.fasta')
        p_fasta = os.path.join(dfasta, 'normal_paternal.fasta')
        # phase_file = os.path.join(dprofile, 'phases.tsv')
        allele_phase_file = os.path.join(dprofile, 'snp_phases.csv')
        tree_newick = os.path.join(dprofile, 'tree.newick')
        tree_pdf = os.path.join(dprofile, 'tree.pdf')
        tree_json = os.path.join(dprofile, 'tree.json')

        # generate random clone tree and set root as normal clone
        self.log('Generating random cell-lineage tree...', level='PROGRESS')
        root = random_tree.generate_tree_beta(cell_num=self.cell_no, num_clones=self.clone_no, alpha=self.tree_alpha, beta=self.tree_beta, treedepth=self.max_tree_depth, treedepthsigma=self.tree_depth_sigma, max_children=self.max_node_children, balance_factor=self.tree_balance_factor, seed=self.tree_seed)

        self.log('Writing tree to file with newick format...', level='PROGRESS')
        result = random_tree.tree_to_newick(root)
        with open(tree_newick, 'w') as output:
            output.write(result)
        
        self.log('Drawing tree graph with pdf format...', level='PROGRESS')
        random_tree.draw_tree_to_pdf(root, tree_pdf)

        self.log('Getting chrommosome sizes...', level='PROGRESS')
        self._get_chrom_sizes()

        # set normal fasta file path
        root.maternal_fasta = m_fasta
        root.paternal_fasta = p_fasta
        normal_fasta_length = sum(self.chrom_sizes.values())
        root.maternal_fasta_length = normal_fasta_length
        root.paternal_fasta_length = normal_fasta_length

        # generate normal fasta with snps 
        self.log("Building normal fasta file with SNPs data...", level='PROGRESS')
        self._buildGenome(m_fasta, p_fasta, allele_phase_file)

        # generate cnv for each clone
        self.log('Generating CNV profile for each clone...', level='PROGRESS')
        loop_no = 1
        unique_mirrored_subclonal_cnas_no = 0
        while unique_mirrored_subclonal_cnas_no < 3 and loop_no < 5:
            ref = self._split_chr_to_bins('all')
            new_ref, changes, maternal_genome, paternal_genome = self._generate_cnv_profile_for_each_clone(root, ref, m_fasta, p_fasta)
            new_changes, cnv_profile = self._out_cnv_profile(root, new_ref, changes, dprofile)

            mirrored_subclonal_cnas = self._find_mirrored_clones(cnv_profile)
            if not mirrored_subclonal_cnas.empty:
                unique_mirrored_subclonal_cnas_no = len(mirrored_subclonal_cnas[['Chromosome', 'Start', 'End']].drop_duplicates())
            loop_no = loop_no + 1
        mirrored_subclonal_cnas.to_csv(os.path.join(dprofile, 'mirrored_subclonal_cnas.csv'), index=False)

        # store the tree to json file
        self.log('Storing the tree to json file...', level='PROGRESS')
        random_tree.save_tree_to_file(root, tree_json)
        self.log('gprofile BYEBYE')
    
    @utils.log_runtime
    def gfasta(self):
        self.log('Setting directories', level='PROGRESS')
        dprofile, dfasta, dfastq, dclone, dcell, dbarcode, drdr, dbaf, dtmp, dlog = self.setup_dir()

        tree_json = os.path.join(dprofile, 'tree.json')
        ref_file = os.path.join(dprofile, 'reference.csv')
        changes_file = os.path.join(dprofile, 'changes.csv')
        maternal_fasta = os.path.join(dfasta, 'normal_maternal.fasta')
        paternal_fasta = os.path.join(dfasta, 'normal_paternal.fasta')

        utils.check_exist(tree_json=tree_json)
        utils.check_exist(reference_csv=ref_file)
        utils.check_exist(changes_csv=changes_file)
        utils.check_exist(normal_maternal_fasta=maternal_fasta)
        utils.check_exist(normal_paternal_fasta=paternal_fasta)
        
        # load object from file
        root = random_tree.load_tree_from_file(tree_json)
        ref = pd.read_csv(ref_file)
        changes = pd.read_csv(changes_file)

        # store maternal and paternal genome to dict
        maternal_genome = {}
        paternal_genome = {}
        with open(maternal_fasta, 'r') as input:
            chrom = None
            for line in input:
                line = line.strip()
                if line.startswith('>'):
                    chrom = line.strip()[1:].split()[0]
                    maternal_genome[chrom] = ''
                else:
                    maternal_genome[chrom] += line
        with open(paternal_fasta, 'r') as input:
            chrom = None
            for line in input:
                line = line.strip()
                if line.startswith('>'):
                    chrom = line.strip()[1:].split()[0]
                    paternal_genome[chrom] = ''
                else:
                    paternal_genome[chrom] += line

        # set parallel jobs for each clone
        jobs = [(clone, ref, changes, maternal_genome, paternal_genome, dfasta) for clone in random_tree.collect_all_nodes(root)]
        lock = Lock()
        counter = Value('i', 0)
        init_args = (lock, counter, len(jobs))
        pool = Pool(processes=min(self.thread, len(jobs)), initializer=init_gfasta, initargs=init_args)

        self.log('Generating fasta file for each clone...', level='PROGRESS')
        for clone in pool.imap_unordered(self._generate_fasta_for_each_clone, jobs):
            root = random_tree.update_node_in_tree(root, clone)
        pool.close()
        pool.join()

        self.log('Storing the tree to json file...', level='PROGRESS')
        random_tree.save_tree_to_file(root, tree_json)
        self.log('gfasta BYEBYE')

    @utils.log_runtime
    def gfastq(self):
        self.log('Setting directories', level='PROGRESS')
        dprofile, dfasta, dfastq, dclone, dcell, dbarcode, drdr, dbaf, dtmp, dlog = self.setup_dir()

        tree_json = os.path.join(dprofile, 'tree.json')
        ref_file = os.path.join(dprofile, 'reference.csv')
    
        utils.check_exist(tree_json=tree_json)
        utils.check_exist(reference_csv=ref_file)

        # load object from file
        root = random_tree.load_tree_from_file(tree_json)
        all_clones = random_tree.collect_all_nodes(root, 1)
        ref = pd.read_csv(ref_file)

        jobs = []
        # check fasta file for each clone
        for clone in all_clones:
            utils.check_exist(maternal_fasta=clone.maternal_fasta)
            utils.check_exist(paternal_fasta=clone.paternal_fasta)
            jobs.append((clone, ref, dfastq))

        # set parallel jobs for each clone
        lock = Lock()
        counter = Value('i', 0)
        init_args = (lock, counter, len(jobs))
        pool = Pool(processes=min(self.thread, len(jobs)), initializer=init_gfastq, initargs=init_args)
        
        self.log('Generating fastq file for each clone...', level='PROGRESS')
        for _ in pool.imap_unordered(self._generate_fastq_for_each_clone, jobs):
            pass
        pool.close()
        pool.join()

        self.log('Storing the tree to json file...', level='PROGRESS')
        random_tree.save_tree_to_file(root, tree_json)
        self.log('gfastq BYEBYE')

    @utils.log_runtime
    def align(self):
        self.log('Setting directories', level='PROGRESS')
        dprofile, dfasta, dfastq, dclone, dcell, dbarcode, drdr, dbaf, dtmp, dlog = self.setup_dir()

        tree_json = os.path.join(dprofile, 'tree.json')
    
        utils.check_exist(tree_json=tree_json)
        
        # load object from file
        root = random_tree.load_tree_from_file(tree_json)
        all_clones = random_tree.collect_all_nodes(root, 1)

        jobs = []
        # check fasta file for each clone
        for clone in all_clones:
            m_fq1 = os.path.join(dfastq, clone.name + "_maternal_r1.fq")
            m_fq2 = os.path.join(dfastq, clone.name + "_maternal_r2.fq")
            p_fq1 = os.path.join(dfastq, clone.name + "_paternal_r1.fq")
            p_fq2 = os.path.join(dfastq, clone.name + "_paternal_r2.fq")
            utils.check_exist(maternal_fastq_r1=m_fq1)
            utils.check_exist(maternal_fastq_r2=m_fq2)
            utils.check_exist(paternal_fastq_r1=p_fq1)
            utils.check_exist(paternal_fastq_r2=p_fq2)
            jobs.append((clone.name, dfastq, dclone, dlog))

        # set parallel jobs for each clone
        lock = Lock()
        counter = Value('i', 0)
        init_args = (lock, counter, len(jobs))
        pool = Pool(processes=1, initializer=init_align, initargs=init_args)
        
        self.log('Aligning fastq file for each clone...', level='PROGRESS')
        for _ in pool.imap_unordered(self._alignment_for_each_clone, jobs):
            pass
        pool.close()
        pool.join()

        self.log('Storing the tree to json file...', level='PROGRESS')
        random_tree.save_tree_to_file(root, tree_json)
        self.log('align BYEBYE')
    
    @utils.log_runtime
    def downsam(self):
        self.log('Setting directories', level='PROGRESS')
        dprofile, dfasta, dfastq, dclone, dcell, dbarcode, drdr, dbaf, dtmp, dlog = self.setup_dir()

        tree_json = os.path.join(dprofile, 'tree.json')
    
        utils.check_exist(tree_json=tree_json)
        
        # load object from file
        root = random_tree.load_tree_from_file(tree_json)
        all_clones = random_tree.collect_all_nodes(root, 1)
        
        # check bam file for each clone
        for clone in all_clones:
            clone_bam_file = os.path.join(dclone, clone.name + ".bam")
            utils.check_exist(clone_bam_file=clone_bam_file)

        # assign cells for each clone and generating job list
        barcodes = []
        jobs = []
        clones = ['clone' + str(i) for i in range(1, self.clone_no)]
        clones.append('normal')
        assign_cells = utils.assign_cells_to_clones(self.cell_no, self.clone_no)
        cell_ratio = round(self.cell_coverage/self.clone_coverage, 2)
        for index, clone in enumerate(all_clones):
            clone_cell_no = assign_cells[index]
            clone_bam_file = os.path.join(dclone, clone.name + '.bam')
            for i in range(clone_cell_no):
                cell_name = clone.name + '_cell' + str(i+1)
                barcodes.append(cell_name)
                cell_bam_file = os.path.join(dcell, cell_name+'.bam')
                ratio = cell_ratio + i
                jobs.append((ratio, clone_bam_file, cell_bam_file, dlog))
        
        # set parallel jobs for each cell
        lock = Lock()
        counter = Value('i', 0)
        init_args = (lock, counter, len(jobs))
        pool = Pool(processes=1, initializer=init_downsam, initargs=init_args)
        
        self.log('Downsampling cell bam...', level='PROGRESS')
        for _ in pool.imap_unordered(self._downsampling_cell_bam, jobs):
            pass
        pool.close()
        pool.join()

        self.log('Writing cell list to barcode.txt...', level='PROGRESS')
        barcodes_file = os.path.join(dprofile, 'barcodes.txt')
        with open(barcodes_file, 'w') as output:
            for barcode in barcodes:
                output.write(barcode+'\n')

        self.log('Storing the tree to json file...', level='PROGRESS')
        random_tree.save_tree_to_file(root, tree_json)
        self.log('downsam BYEBYE')

    @utils.log_runtime
    def pbam(self):
        self.log('Setting directories', level='PROGRESS')
        dprofile, dfasta, dfastq, dclone, dcell, dbarcode, drdr, dbaf, dtmp, dlog = self.setup_dir()

        # load cell list from barcode file
        cell_list = []
        barcode_file = os.path.join(dprofile, 'barcodes.txt')
        with open(barcode_file, 'r') as output:
            for line in output.readlines():
                if line.strip() != '':
                    cell_list.append(line.strip())

        # set jobs
        jobs = []
        for cell in cell_list:
            cell_bam = os.path.join(dcell, cell + '.bam')
            utils.check_exist(cell_bam=cell_bam)
            jobs.append((cell, dcell, dtmp, dlog))
        
        # set parallel jobs for each cell
        lock = Lock()
        counter = Value('i', 0)
        init_args = (lock, counter, len(jobs))
        pool = Pool(processes=1, initializer=init_pbam, initargs=init_args)
        
        self.log('Processing cell bam...', level='PROGRESS')
        for _ in pool.imap_unordered(self._process_cell_bam, jobs):
            pass
        pool.close()
        pool.join()

        self.log('pbam BYEBYE')
    
    @utils.log_runtime
    def bcbam(self):
        self.log('Setting directories', level='PROGRESS')
        dprofile, dfasta, dfastq, dclone, dcell, dbarcode, drdr, dbaf, dtmp, dlog = self.setup_dir()

        self.log('Staring generating barcode bam file...', level='PROGRESS')
        barcode_bam_log = os.path.join(dlog, 'barcode_bam_log.txt')
        barcode_py = os.path.join(utils.root_path(), 'generate_barcode_bam.py')
        barcode_bam = os.path.join(dbarcode, 'barcode.bam')
        command = 'python {0} -x {1} -o {2} --noduplicates {3}/*.bam --barcodelength {4} --bcftools {5} --samtools {6} --bwa {7} -j {8}'.format(barcode_py, dbarcode, barcode_bam, dcell, self.barcode_len, self.bcftools, self.samtools, self.bwa, self.thread)
        utils.runcmd(command, barcode_bam_log)

        self.log('bcbam BYEBYE')

    @utils.log_runtime
    def rdr(self):
        self.log('Setting directories', level='PROGRESS')
        dprofile, dfasta, dfastq, dclone, dcell, dbarcode, drdr, dbaf, dtmp, dlog = self.setup_dir()

        self.log('Calculating RDRs', level='PROGRESS')

        # load ref file
        self._get_chrom_sizes()
        ref = self._split_chr_to_bins('all')

        # write ref to bed file to drdr without header and index
        ref['Start'] = ref['Start'] - 1
        ref.to_csv(os.path.join(dtmp, 'rdr_reference.bed'), sep='\t', index=False, header=False)

        # load cell list from barcode file
        cell_list = []
        barcode_file = os.path.join(dprofile, 'barcodes.txt')
        with open(barcode_file, 'r') as output:
            for line in output.readlines():
                if line.strip() != '':
                    cell_list.append(line.strip())
        # set jobs
        jobs = []
        for cell in cell_list:
            cell_bam = os.path.join(dcell, cell + '.bam')
            utils.check_exist(cell_bam=cell_bam)
            jobs.append((cell, dcell, dtmp, dlog))

        # set parallel jobs for each cell
        lock = Lock()
        counter = Value('i', 0)
        init_args = (lock, counter, len(jobs))
        pool = Pool(processes=min(self.thread, len(jobs)), initializer=init_cov, initargs=init_args)

        self.log('Computing reads for each cell', level='PROGRESS')
        for _ in pool.imap_unordered(self._get_coverage_for_each_cell, jobs):
            pass
        pool.close()
        pool.join()

        self.log('Calculating RDRs', level='PROGRESS')
        count_files = [os.path.join(dtmp, cell + '.coverage.bed') for cell in cell_list]
        coverage_df = utils.merge_coverage_files(count_files)
        coverage_df.to_csv(os.path.join(drdr, 'coverage.tsv'), sep='\t')

        # calculate RDRs
        # rdr = ratio X scale
        # scale = total reads in normal / total reads in cell
        # ratio = reads count in each bin in cell / reads count in each bin in normal
        # finally, rdr is a matrix same as coverage_df, but value is rar value
        normal_cov = coverage_df['normal_cell1']
        normal_sum = coverage_df['normal_cell1'].sum()
        coverage_sums = coverage_df.sum(axis=0)
        scale = normal_sum / coverage_sums
        ratio = coverage_df / normal_cov.values.reshape(-1, 1)
        rdr = ratio * scale.values
        rdr_df = pd.DataFrame(rdr, index=coverage_df.index, columns=coverage_df.columns).round(4)
        rdr_df.to_csv(os.path.join(drdr, 'rdr.tsv'), sep='\t')

        self.log('RDR BYEBYE')
    
    @utils.log_runtime
    def baf(self):
        self.log('Setting directories', level='PROGRESS')
        dprofile, dfasta, dfastq, dclone, dcell, dbarcode, drdr, dbaf, dtmp, dlog = self.setup_dir()
        
        self.log('Calculating BAFs', level='PROGRESS')
        
        # load ref file
        self._get_chrom_sizes()
        ref = self._split_chr_to_bins('all')

        # handle phased snps
        allele_phase_file = os.path.join(dprofile, 'snp_phases.csv')
        phased_snps = utils.read_phase(allele_phase_file)

        # generate bed file for bcftools with filtered phased snps
        filtered_snp_bed = os.path.join(dtmp, 'snps.bed')
        utils.write_snp_bed_file(filtered_snp_bed, phased_snps)

        # load cell list from barcode file
        cell_list = []
        barcode_file = os.path.join(dprofile, 'barcodes.txt')
        with open(barcode_file, 'r') as output:
            for line in output.readlines():
                if line.strip() != '':
                    cell_list.append(line.strip())
        # set jobs
        jobs = []
        for cell in cell_list:
            cell_bam = os.path.join(dcell, cell + '.bam')
            utils.check_exist(cell_bam=cell_bam)
            jobs.append((cell, dcell, dtmp, dlog))

        # set parallel jobs for each cell
        lock = Lock()
        counter = Value('i', 0)
        init_args = (lock, counter, len(jobs))
        pool = Pool(processes=min(self.thread, len(jobs)), initializer=init_bcftools, initargs=init_args)

        self.log('Counting germinal SNPs for each cell', level='PROGRESS')
        for _ in pool.imap_unordered(self._call_bcftools_for_each_cell, jobs):
            pass
        pool.close()
        pool.join()

        # merge all count files to A-allele and B-allele matrix file
        count_files = [os.path.join(dtmp, cell + '.count.bed') for cell in cell_list]

        # Save DataFrames to TSV files
        a_allele_tsv = os.path.join(dbaf, 'a-allele.snp.tsv')
        b_allele_tsv = os.path.join(dbaf, 'b-allele.snp.tsv')
        a_allele_bed = os.path.join(dtmp, 'a-allele.bed')
        b_allele_bed = os.path.join(dtmp, 'b-allele.bed')
        a_allele_df, b_allele_df = utils.create_allele_count_matrices(count_files, phased_snps, a_allele_bed, b_allele_bed)
        snp_baf = b_allele_df / (a_allele_df + b_allele_df)
        snp_baf = snp_baf.round(4)
        a_allele_df.to_csv(a_allele_tsv, sep='\t', index_label='snp_id')
        b_allele_df.to_csv(b_allele_tsv, sep='\t', index_label='snp_id')
        snp_baf.to_csv(os.path.join(dbaf, 'baf.snp.tsv'), sep='\t', index_label='snp_id')
        
        ref_bed = os.path.join(dtmp, 'baf_reference.bed')
        ref.to_csv(ref_bed, sep='\t', index=False, header=False)

        # use bedtools to get intersect of a_allele, b_allele and reference bed files
        a_allele_intersect = os.path.join(dtmp, 'a-allele_intersect.bed')
        b_allele_intersect = os.path.join(dtmp, 'b-allele_intersect.bed')
        bedtools_log = os.path.join(dlog, 'bedtools_log.txt')
        cmd = f"{self.bedtools} intersect -a {ref_bed} -b {a_allele_bed} -wa -wb > {a_allele_intersect}"
        utils.runcmd(cmd, bedtools_log)
        cmd = f"{self.bedtools} intersect -a {ref_bed} -b {b_allele_bed} -wa -wb > {b_allele_intersect}"
        utils.runcmd(cmd, bedtools_log)

        # read intersected.bed file
        cols = ['bin_chrom', 'bin_start', 'bin_end', 'snp_chrom', 'snp_start', 'snp_end', 'cell', 'count']
        a_allele_intersect_data = pd.read_csv(a_allele_intersect, sep='\t', names=cols)
        b_allele_intersect_data = pd.read_csv(b_allele_intersect, sep='\t', names=cols)

        # create bin column
        a_allele_intersect_data['bin'] = a_allele_intersect_data['bin_chrom'] + ':' + a_allele_intersect_data['bin_start'].astype(str) + '-' + a_allele_intersect_data['bin_end'].astype(str)
        b_allele_intersect_data['bin'] = b_allele_intersect_data['bin_chrom'] + ':' + b_allele_intersect_data['bin_start'].astype(str) + '-' + b_allele_intersect_data['bin_end'].astype(str)

        # calculate total count for each bin
        a_allele_count = a_allele_intersect_data.groupby(['bin', 'cell'])['count'].sum().unstack(fill_value=0)
        b_allele_count = b_allele_intersect_data.groupby(['bin', 'cell'])['count'].sum().unstack(fill_value=0)
        baf = b_allele_count / (a_allele_count + b_allele_count)

        # store result
        a_allele_file = os.path.join(dbaf, 'a-allele.bin.tsv')
        b_allele_file = os.path.join(dbaf, 'b-allele.bin.tsv')
        baf_matrix_file = os.path.join(dbaf, 'baf.bin.tsv')

        # a_allele_martrix, b_allele_martrix, baf_matrix = merge_count_files(count_files, phased_snps_filtered, ref)
        a_allele_count = a_allele_count.sort_values(by="bin", key=lambda col: col.map(utils.extract_chr_and_start)).round(4)
        b_allele_count = b_allele_count.sort_values(by="bin", key=lambda col: col.map(utils.extract_chr_and_start)).round(4)
        baf = baf.sort_values(by="bin", key=lambda col: col.map(utils.extract_chr_and_start)).round(4)
        a_allele_count.to_csv(a_allele_file, sep='\t')
        b_allele_count.to_csv(b_allele_file, sep='\t')
        baf.to_csv(baf_matrix_file, sep='\t')

        self.log('BAF BYEBYE')

    def sim(self):
        self.gprofile()
        self.gfasta()
        self.gfastq()
        self.align()
        self.downsam()
        self.pbam()
        self.bcbam()
        self.rdr()
        self.baf()
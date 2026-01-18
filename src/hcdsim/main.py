import copy
import datetime
import multiprocessing as mp
import os
import random
import sys
from collections import deque
from multiprocessing import Pool, Value, Lock
from scipy.stats import poisson
import numpy as np
import pandas as pd
from typing import List, Optional

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
                genome_version: str = 'hg38',
                clone_no: int = 2, 
                cell_no: int = 2, 
                tree_alpha: float = 10.0, 
                tree_beta: float = 10.0, 
                max_tree_depth: int = 4, 
                tree_depth_sigma: float = 0.5, 
                max_node_children: int = 4, 
                tree_balance_factor: float = 0.8,
                tree_newwick: str = None,
                tree_mode: int = 0,
                random_seed: int = None,
                bin_size: str = '100kb', 
                snp_ratio: float = 0.001, 
                thread: int = None, 
                heho_ratio: float = 0.67, 
                cna_prob: float = 0.02,
                del_prob: float = 0.2,
                cna_copy_param: float = 0.5,
                weights: Optional[List[float]] = None,
                lambdas: Optional[List[int]] = None,
                clone_coverage: float = 30, 
                cell_coverage: float = 0.01, 
                reads_len: int = 150, 
                insertion_size: int = 350, 
                error_rate: float = 0.02, 
                wgd: bool = False,
                chrom_cna_no: int = 2,
                chrom_arm_rate: float = 0.75,
                loh_cna_no: int = 15, 
                goh_cna_no: int = 5, 
                unique_ratio: float = 0.5,
                barcode_len: int = 12,
                lorenz_x: float = 0.5,
                lorenz_y: float = 0.35,
                window_size: int = 200000,
                correlation_len: int = 10,
                max_ploidy: int = None,
                max_cna_value: int = 10,
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
        
        # set default weights and lambdas for mixture poisson
        if weights is None:
            self.weights = [0.3, 0.4, 0.2, 0.1]
        else:
            self.weights = weights

        if lambdas is None:
            self.lambdas = [5, 20, 100, 300]
        else:
            self.lambdas = lambdas

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
        
        utils.set_random_seed(self.random_seed)

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
        utils.check_between(0,1,cna_prob=self.cna_prob)
        utils.check_positive(clone_coverage=self.clone_coverage)
        utils.check_positive(cell_coverage=self.cell_coverage)
        utils.check_int(reads_len=self.reads_len)
        utils.check_positive(reads_len=self.reads_len)
        utils.check_int(reads_len=self.thread)
        utils.check_positive(reads_len=self.thread)
        utils.check_int(insertion_size=self.insertion_size)
        utils.check_positive(insertion_size=self.insertion_size)
        utils.check_between(0,1,error_rate=self.error_rate)
        utils.check_int(loh_cna_no=self.loh_cna_no)
        utils.check_lt_zero(loh_cna_no=self.loh_cna_no)
        utils.check_int(goh_cna_no=self.goh_cna_no)
        utils.check_lt_zero(goh_cna_no=self.goh_cna_no)
        utils.check_int(barcode_len=self.barcode_len)
        utils.check_positive(barcode_len=self.barcode_len)

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
        """Split chromosomes to fixed-length bins with chromosome arm annotation

        Parameters
        ----------
        chrom : str
            Chromosome name or 'all' for all chromosomes
        genome_version : str
            'hg19' or 'hg38' (default: 'hg38')

        Returns
        -------
        ref: DataFrame of pandas with columns ['Chromosome', 'Start', 'End', 'Arm']
        """
        genome_version = self.genome_version
        bin_size = self.bin_size
        
        centromere_hg19 = {
            'chr1': 125200000, 'chr2': 93650000, 'chr3': 90900000,
            'chr4': 50450000, 'chr5': 48400000, 'chr6': 61000000,
            'chr7': 59850000, 'chr8': 45600000, 'chr9': 49000000,
            'chr10': 40150000, 'chr11': 53650000, 'chr12': 35750000,
            'chr13': 17900000, 'chr14': 17600000, 'chr15': 18250000,
            'chr16': 36600000, 'chr17': 24000000, 'chr18': 17200000,
            'chr19': 26500000, 'chr20': 27500000, 'chr21': 12600000,
            'chr22': 15050000, 'chrX': 60550000, 'chrY': 12500000
        }
        
        centromere_hg38 = {
            'chr1': 123400000, 'chr2': 93900000, 'chr3': 90900000,
            'chr4': 50000000, 'chr5': 48750000, 'chr6': 60550000,
            'chr7': 60100000, 'chr8': 45200000, 'chr9': 43850000,
            'chr10': 39800000, 'chr11': 53400000, 'chr12': 35500000,
            'chr13': 17700000, 'chr14': 17150000, 'chr15': 19000000,
            'chr16': 36850000, 'chr17': 25050000, 'chr18': 18450000,
            'chr19': 26150000, 'chr20': 28050000, 'chr21': 11950000,
            'chr22': 15550000, 'chrX': 60950000, 'chrY': 10450000
        }
        
        if genome_version.lower() == 'hg19':
            centromere_positions = centromere_hg19
        elif genome_version.lower() == 'hg38':
            centromere_positions = centromere_hg38
        else:
            raise ValueError(f"Unsupported genome version: {genome_version}. Use 'hg19' or 'hg38'.")
        
        bins = []
        
        def process_chromosome(chrom, chrom_size):
            centro_pos = centromere_positions.get(chrom, chrom_size / 2)
            start = 1
            end = bin_size
            count = 1
            while start < chrom_size:
                bin_end = min(end, chrom_size)
                mid_point = (start + bin_end) / 2
                arm = 'p' if mid_point < centro_pos else 'q'
                bins.append({
                    'Chromosome': chrom,
                    'Start': start,
                    'End': bin_end,
                    'Arm': arm
                })
                count += 1
                start = end + 1
                end = bin_size * count
        
        if chrom != 'all':
            process_chromosome(chrom, self.chrom_sizes[chrom])
        else:
            for chrom, chrom_size in self.chrom_sizes.items():
                process_chromosome(chrom, chrom_size)
        
        return pd.DataFrame(bins)

    def _generate_cna_profile_for_each_clone(self, root, ref, m_fasta, p_fasta):
        
        all_chroms = ref['Chromosome'].unique().tolist()

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
        
        # add nromal clone to cna matrix
        root.maternal_cnas = []
        root.paternal_cnas = []
        root.changes = []
        for i in range(ref.shape[0]):
            root.maternal_cnas.append(1)
            root.paternal_cnas.append(1)
        ref[root.name+'_maternal_cnas'] = root.maternal_cnas
        ref[root.name+'_paternal_cnas'] = root.paternal_cnas
        
        # add the children of normal clone to queue
        queue = deque(root.children)
        total_bin_lens = ref.shape[0]
        wgd_flag = False
        while  queue:
            clone = queue.popleft()
            clone.maternal_cnas = []
            clone.paternal_cnas = []
            clone.changes = []
            clone.cna_status = [None] * total_bin_lens  # None, 'cnl', 'cnn', 'goh', 'mirror'

            if clone.depth == 1: # children of normal clone
                # if include wgd, set one clone to wgd
                if self.wgd and not wgd_flag:
                    if self.max_ploidy:
                        cna_copies = int(self.max_ploidy/2)
                    else:
                        cna_copies = np.clip(np.random.geometric(self.cna_copy_param), 2, int(self.max_cna_value/2))
                    
                    for i in range(total_bin_lens):
                        clone.cna_status[i] = 'wgd'
                        clone.maternal_cnas.append(cna_copies)
                        clone.paternal_cnas.append(cna_copies)
                    
                    # output wgd per chroms
                    for chrom in ref['Chromosome'].unique():
                        chrom_bins = ref[ref['Chromosome'] == chrom]
                        chrom_start = chrom_bins['Start'].min()
                        chrom_end = chrom_bins['End'].max()
                        
                        clone.changes.append([
                            'normal',
                            clone.name,
                            'maternal',
                            'WGD',
                            f"{chrom}:{chrom_start}-{chrom_end}", str(chrom_end - chrom_start + 1),
                            f'1->{cna_copies}'
                        ])
                        clone.changes.append([
                            'normal',
                            clone.name,
                            'paternal',
                            'WGD',
                            f"{chrom}:{chrom_start}-{chrom_end}",str(chrom_end - chrom_start + 1),
                            f'1->{cna_copies}'
                        ])
                    
                    wgd_flag = True
                    ref[clone.name+'_maternal_cnas'] = clone.maternal_cnas
                    ref[clone.name+'_paternal_cnas'] = clone.paternal_cnas
                    queue.extend(clone.children)
                    continue

                cna_event_id = 1

                # select chrom cna chromosomes
                chrom_cna_chroms = random.sample(all_chroms, self.chrom_cna_no)
                for chrom in chrom_cna_chroms:
                    chrom_indices = ref[ref['Chromosome'] == chrom].index.tolist()
                    # arm level event
                    if np.random.binomial(1, self.chrom_arm_rate):
                        if np.random.binomial(1, 0.5):  # p arm
                            arm = 'p'
                        else:  # q arm
                            arm = 'q'
                        arm_indices = ref[(ref['Chromosome'] == chrom) & (ref['Arm'] == arm)].index.tolist()
                        for i in arm_indices:
                            clone.cna_status[i] = ('chrom-arm-cna', len(arm_indices), cna_event_id)
                        cna_event_id += 1 
                            
                    else:  # whole chrom level event
                        for i in chrom_indices:
                            clone.cna_status[i] = ('whole-chrom-cna', len(chrom_indices), cna_event_id)
                        cna_event_id += 1
                                    
                # select the position for CNL_LOH, CNN_LOH, GOH and Mirror CNA
                available_indices = [i for i in range(total_bin_lens) if clone.cna_status[i] is None]

                if len(available_indices) == 0:
                    raise Exception("No available bins for LOH and GOH CNAs after assigning WGD and WCL CNAs. Please decrease the number of WGD and WCL CNAs.")

                cnl_loh_no = int(self.loh_cna_no/3)

                cna_types_to_generate = (
                    ["cnl"] * cnl_loh_no + 
                    ["cnn"] * (self.loh_cna_no - cnl_loh_no) + 
                    ["goh"] * self.goh_cna_no
                )

                random.shuffle(available_indices)
                random.shuffle(cna_types_to_generate)

                cna_generated = 0
                total_cnas = len(cna_types_to_generate)

                for start_idx in available_indices:
                    if cna_generated >= total_cnas:
                        break
                    
                    if clone.cna_status[start_idx] is not None:
                        continue
                    
                    num_windows_span = utils.generate_mixture_poisson(self.weights, self.lambdas)
                    
                    # Get chromosome of the start position
                    start_chrom = ref['Chromosome'][start_idx]
                    
                    # Check if CNA can be placed without crossing chromosome boundary
                    can_place = True
                    span_indices = []
                    
                    for j in range(num_windows_span):
                        idx = start_idx + j
                        
                        # Check if we've exceeded total bins
                        if idx >= total_bin_lens:
                            break
                        
                        # Check if we've crossed chromosome boundary
                        if ref['Chromosome'][idx] != start_chrom:
                           break
                        
                        # Check if position is already occupied
                        if clone.cna_status[idx] is not None:
                            can_place = False
                            break
                        
                        span_indices.append(idx)
                    
                    if can_place and len(span_indices) > 0:
                        cna_type = cna_types_to_generate[cna_generated]
                        
                        # Mark all positions in the span
                        for idx in span_indices:
                            clone.cna_status[idx] = (cna_type, len(span_indices), cna_event_id)
                        
                        cna_event_id += 1 
                        cna_generated += 1

                # generate usual cna according to probability cutoff
                i = 0
                while i < total_bin_lens:
                    if clone.cna_status[i] is None:
                        if np.random.binomial(1, self.cna_prob):
                            num_windows_span = utils.generate_mixture_poisson(self.weights, self.lambdas)
                            
                            # Get chromosome of the start position
                            start_chrom = ref['Chromosome'][i]
                            
                            # Find valid span that doesn't cross chromosome boundary
                            can_place = True
                            span_indices = []
                            
                            for j in range(num_windows_span):
                                idx = i + j
                                
                                # Check if we've exceeded total bins
                                if idx >= total_bin_lens:
                                    break
                                
                                # Check if we've crossed chromosome boundary
                                if ref['Chromosome'][idx] != start_chrom:
                                    break
                                
                                # Check if position is already occupied
                                if clone.cna_status[idx] is not None:
                                    can_place = False
                                    break
                                
                                span_indices.append(idx)
                            
                            if can_place and len(span_indices) > 0:
                                # Place the CNA event
                                for idx in span_indices:
                                    clone.cna_status[idx] = ("cna", len(span_indices), cna_event_id)
                                cna_event_id += 1
                                i += len(span_indices)  # Skip past the CNA we just placed
                            else:
                                i += 1
                        else:
                            i += 1
                    else:
                        i += 1

                # After generating all CNA statuses, iterate through all bins to process CNA events
                # First, initialize maternal_cnas and paternal_cnas with baseline value of 1
                for i in range(total_bin_lens):
                    clone.maternal_cnas.append(1)
                    clone.paternal_cnas.append(1)

                # Track processed event IDs to avoid duplicate processing
                processed_events = set()

                # Iterate through all bins to identify and process each CNA event
                i = 0
                while i < total_bin_lens:
                    if clone.cna_status[i] is None:
                        # No CNA event
                        i += 1
                        continue
                    
                    # Get current event information
                    cna_type, num_windows, event_id = clone.cna_status[i]

                    # If this event has already been processed, skip it
                    if event_id in processed_events:
                        i += 1
                        continue

                    # Mark this event as processed
                    processed_events.add(event_id)

                    # Find all bin indices for this event using num_windows
                    event_bins = []
                    for j in range(i, min(i + num_windows, total_bin_lens)):
                        if clone.cna_status[j] is not None and clone.cna_status[j][2] == event_id:
                            event_bins.append(j)
                                        
                    # Get the start and end positions of the event
                    event_start_idx = event_bins[0]
                    event_end_idx = event_bins[-1]
                    event_chrom = ref['Chromosome'][event_start_idx]
                    event_start_pos = ref['Start'][event_start_idx]
                    event_end_pos = ref['End'][event_end_idx]
                    m_sequence = maternal_genome[event_chrom][event_start_pos-1:event_end_pos]
                    p_sequence = paternal_genome[event_chrom][event_start_pos-1:event_end_pos]

                    # Handle Mirror CNA
                    if cna_type == "cnl":
                        if m_sequence != p_sequence:
                            # CNL_LOH: 1:0 or 0:1
                            if np.random.binomial(1, 0.5):  # Keep maternal, delete paternal
                                m_cna = 1
                                p_cna = 0
                            else:  # Keep paternal, delete maternal
                                m_cna = 0
                                p_cna = 1
                            
                            for bin_idx in event_bins:
                                clone.maternal_cnas[bin_idx] = m_cna
                                clone.paternal_cnas[bin_idx] = p_cna
                            
                            if m_cna == 0:
                                clone.changes.append([
                                    'normal', clone.name, 'maternal', 'CNL_LOH',
                                    f"{event_chrom}:{event_start_pos}-{event_end_pos}",str(event_end_pos-event_start_pos + 1),
                                    f'1->{m_cna}'
                                ])
                            
                            if p_cna == 0:
                                clone.changes.append([
                                    'normal', clone.name, 'paternal', 'CNL_LOH',
                                    f"{event_chrom}:{event_start_pos}-{event_end_pos}",str(event_end_pos-event_start_pos + 1),
                                    f'1->{p_cna}'
                                ])
                
                    # Handle CNN_LOH (Copy Number Neutral LOH with duplication)
                    elif cna_type == "cnn":
                        if m_sequence != p_sequence:
                            # CNN_LOH: 2:0 or 0:2
                            cnn_cna = np.clip(np.random.geometric(self.cna_copy_param), 2, int(self.max_cna_value))
                            
                            if np.random.binomial(1, 0.5):  # Maternal duplication, paternal deletion
                                m_cna = cnn_cna
                                p_cna = 0
                            else:  # Paternal duplication, maternal deletion
                                m_cna = 0
                                p_cna = cnn_cna
                            
                            for bin_idx in event_bins:
                                clone.maternal_cnas[bin_idx] = m_cna
                                clone.paternal_cnas[bin_idx] = p_cna
                            
                            event_type = 'CNN_LOH' if cnn_cna == 2 else 'CNG_LOH'
                            
                            clone.changes.append([
                                'normal', clone.name, 'maternal', event_type,
                                f"{event_chrom}:{event_start_pos}-{event_end_pos}",str(event_end_pos-event_start_pos + 1),
                                f'1->{m_cna}'
                            ])

                            clone.changes.append([
                                'normal', clone.name, 'paternal', event_type,
                                f"{event_chrom}:{event_start_pos}-{event_end_pos}",str(event_end_pos-event_start_pos + 1),
                                f'1->{p_cna}'
                            ])
                    
                    # Handle GOH (Gain of Heterozygosity)
                    elif cna_type == "goh":
                        if m_sequence == p_sequence:
                            # GOH: Both alleles gain, but with different copy numbers
                            m_cna = min(np.random.geometric(self.cna_copy_param), self.max_cna_value - 1)
                            if m_cna == 1:
                                p_cna = min(np.random.geometric(self.cna_copy_param), 2, self.max_cna_value - m_cna)
                            else:
                                p_cna = min(np.random.geometric(self.cna_copy_param), self.max_cna_value - m_cna)
                            
                            for bin_idx in event_bins:
                                clone.maternal_cnas[bin_idx] = m_cna
                                clone.paternal_cnas[bin_idx] = p_cna
                            
                            clone.changes.append([
                                'normal', clone.name, 'maternal', 'GOH',
                                f"{event_chrom}:{event_start_pos}-{event_end_pos}",str(event_end_pos-event_start_pos + 1),
                                f'1->{m_cna}'
                            ])
                            clone.changes.append([
                                'normal', clone.name, 'paternal', 'GOH',
                                f"{event_chrom}:{event_start_pos}-{event_end_pos}",str(event_end_pos-event_start_pos + 1),
                                f'1->{p_cna}'
                            ])
                            
                    # Handle chromosome arm-level CNA
                    elif cna_type == "chrom-arm-cna":
                        # Generate maternal CNA: decide whether it's deletion or duplication
                       
                        if np.random.binomial(1, self.del_prob):  # Deletion
                            m_cna = 0
                        else:  # Duplication
                            m_cna = min(np.random.geometric(self.cna_copy_param), int(self.max_cna_value))
                        
                        # Paternal CNA: independently decide deletion or duplication
                        if np.random.binomial(1, self.del_prob):  # Deletion
                            p_cna = 0
                        else:  # Duplication
                            if m_cna == 1:
                                p_cna = min(np.random.geometric(self.cna_copy_param), 2, int(self.max_cna_value - m_cna))
                            else:
                                p_cna = min(np.random.geometric(self.cna_copy_param), int(self.max_cna_value - m_cna))

                        
                        for bin_idx in event_bins:
                            clone.maternal_cnas[bin_idx] = m_cna
                            clone.paternal_cnas[bin_idx] = p_cna
                        
                        # Record changes and events for maternal
                        if m_cna != 1:
                            m_event_type = 'ARM_DEL' if m_cna == 0 else 'ARM_DUP'
                            clone.changes.append([
                                'normal', clone.name, 'maternal', m_event_type,
                                f"{event_chrom}:{event_start_pos}-{event_end_pos}",str(event_end_pos-event_start_pos + 1),
                                f'1->{m_cna}'
                            ])
                        
                        # Record changes and events for paternal
                        if p_cna != 1:
                            p_event_type = 'ARM_DEL' if p_cna == 0 else 'ARM_DUP'
                            clone.changes.append([
                                'normal', clone.name, 'paternal', p_event_type,
                                f"{event_chrom}:{event_start_pos}-{event_end_pos}",str(event_end_pos-event_start_pos + 1),
                                f'1->{p_cna}'
                            ])
                    
                    # Handle whole chromosome-level CNA
                    elif cna_type == "whole-chrom-cna":
                        # Generate maternal CNA: decide whether it's deletion or duplication
                        if np.random.binomial(1, self.del_prob):  # Deletion
                            m_cna = 0
                        else:  # Duplication
                            m_cna = min(np.random.geometric(self.cna_copy_param), int(self.max_cna_value))
                        
                        # Paternal CNA: independently decide deletion or duplication
                        if np.random.binomial(1, self.del_prob):  # Deletion
                            p_cna = 0
                        else:  # Duplication
                            if m_cna == 1:
                                p_cna = min(np.random.geometric(self.cna_copy_param), 2, int(self.max_cna_value - m_cna))
                            else:
                                p_cna = min(np.random.geometric(self.cna_copy_param), int(self.max_cna_value - m_cna))
                        
                        for bin_idx in event_bins:
                            clone.maternal_cnas[bin_idx] = m_cna
                            clone.paternal_cnas[bin_idx] = p_cna
                        
                        # Record changes and events for maternal
                        if m_cna != 1:
                            m_event_type = 'WCL' if m_cna == 0 else 'WCD'
                            clone.changes.append([
                                'normal', clone.name, 'maternal', m_event_type,
                                f"{event_chrom}:{event_start_pos}-{event_end_pos}",str(event_end_pos-event_start_pos + 1),
                                f'1->{m_cna}'
                            ])
                        
                        # Record changes and events for paternal
                        if p_cna != 1:
                            p_event_type = 'WCL' if p_cna == 0 else 'WCD'
                            clone.changes.append([
                                'normal', clone.name, 'paternal', p_event_type,
                                f"{event_chrom}:{event_start_pos}-{event_end_pos}",str(event_end_pos-event_start_pos + 1),
                                f'1->{p_cna}'
                            ])
                    
                    # Handle regular CNA
                    elif cna_type == "cna":
                        # Generate maternal CNA: decide whether it's deletion or duplication
                        if np.random.binomial(1, self.del_prob):  # Deletion
                            m_cna = 0
                        else:  # Duplication
                            m_cna = min(np.random.geometric(self.cna_copy_param), int(self.max_cna_value))
                        
                        # Paternal CNA: independently decide deletion or duplication
                        if np.random.binomial(1, self.del_prob):  # Deletion
                            p_cna = 0
                        else:  # Duplication
                            if m_cna == 1:
                                p_cna = min(np.random.geometric(self.cna_copy_param), 2, int(self.max_cna_value - m_cna))
                            else:
                                p_cna = min(np.random.geometric(self.cna_copy_param), int(self.max_cna_value - m_cna))
                        
                        for bin_idx in event_bins:
                            clone.maternal_cnas[bin_idx] = m_cna
                            clone.paternal_cnas[bin_idx] = p_cna

                        # check whether is CNL_LOH
                        if (m_sequence != p_sequence) and ((m_cna == 0 and p_cna !=0) or (m_cna != 0 and p_cna ==0)):
                            if m_cna == 1 or p_cna == 1:
                                p_event_type = 'CNL_LOH'
                            elif m_cna == 2 or p_cna == 2:
                                p_event_type = 'CNN_LOH'
                            else:
                                p_event_type = 'CNG_LOH'
                            if m_cna != 1:
                                clone.changes.append([
                                    'normal', clone.name, 'maternal', p_event_type,
                                    f"{event_chrom}:{event_start_pos}-{event_end_pos}",str(event_end_pos-event_start_pos + 1),
                                    f'1->{m_cna}'
                                ])
                            if p_cna != 1:
                                clone.changes.append([
                                    'normal', clone.name, 'paternal', p_event_type,
                                    f"{event_chrom}:{event_start_pos}-{event_end_pos}",str(event_end_pos-event_start_pos + 1),
                                    f'1->{p_cna}'
                                ])
                            continue
                        
                        # Record changes and events for maternal
                        if m_cna != 1:
                            m_event_type = 'DEL' if m_cna == 0 else 'DUP'
                            clone.changes.append([
                                'normal', clone.name, 'maternal', m_event_type,
                                f"{event_chrom}:{event_start_pos}-{event_end_pos}",str(event_end_pos-event_start_pos + 1),
                                f'1->{m_cna}'
                            ])
                        
                        # Record changes and events for paternal
                        if p_cna != 1:
                            p_event_type = 'DEL' if p_cna == 0 else 'DUP'
                            clone.changes.append([
                                'normal', clone.name, 'paternal', p_event_type,
                                f"{event_chrom}:{event_start_pos}-{event_end_pos}",str(event_end_pos-event_start_pos + 1),
                                f'1->{p_cna}'
                            ])
                    i += 1
            else:
                # First, inherit parent's CNA profile
                parent = clone.parent
                
                # Initialize with parent's CNA status and copy numbers
                for i in range(total_bin_lens):
                    clone.cna_status[i] = parent.cna_status[i]
                    clone.maternal_cnas.append(parent.maternal_cnas[i])
                    clone.paternal_cnas.append(parent.paternal_cnas[i])
               
                # Check if parent is WGD
                parent_is_wgd = any('wgd' in status for status in parent.cna_status if status is not None)
                
                cna_event_id = max([status[2] for status in parent.cna_status if isinstance(status, tuple)], default=0) + 1
                
                # If parent is WGD, only allow duplications (copy number increases)
                if parent_is_wgd:
                    # Randomly select positions for new CNAs with probability
                    for i in range(total_bin_lens):
                        if np.random.binomial(1, self.cna_prob * 0.1):  # Reduced probability for post-WGD mutations
                            num_windows_span = utils.generate_mixture_poisson(self.weights, self.lambdas)
                            
                            # Get chromosome of the start position
                            start_chrom = ref['Chromosome'][i]
                            
                            # Find valid span that doesn't cross chromosome boundary
                            can_place = True
                            span_indices = []
                            
                            for j in range(num_windows_span):
                                idx = i + j
                                
                                # Check if we've exceeded total bins
                                if idx >= total_bin_lens:
                                    break
                                
                                # Check if we've crossed chromosome boundary
                                if ref['Chromosome'][idx] != start_chrom:
                                    break

                                if clone.cna_status[idx] == 'post-wgd-dup':
                                    can_place = False
                                    break
                                
                                span_indices.append(idx)
                            
                            if can_place and len(span_indices) > 0:
                                # Get parent copy numbers at this position
                                parent_m_cna = parent.maternal_cnas[i]
                                parent_p_cna = parent.paternal_cnas[i]
                                
                                # Generate duplication CNAs (must be greater than current)
                                # Randomly choose to duplicate maternal, paternal, or both
                                mutation_choice = np.random.choice(['maternal', 'paternal', 'both'])
                                
                                if mutation_choice == 'maternal':
                                    new_m_cna = min(parent_m_cna + np.random.geometric(self.cna_copy_param), int(self.max_cna_value-parent_p_cna))
                                    new_p_cna = parent_p_cna
                                elif mutation_choice == 'paternal':
                                    new_m_cna = parent_m_cna
                                    new_p_cna = min(parent_p_cna + np.random.geometric(self.cna_copy_param), int(self.max_cna_value-parent_m_cna))
                                else:  # both
                                    new_m_cna = min(parent_m_cna + np.random.geometric(self.cna_copy_param), self.max_cna_value - parent_p_cna)
                                    new_p_cna = min(parent_p_cna + np.random.geometric(self.cna_copy_param), self.max_cna_value - new_m_cna)
                                
                                # Apply the mutation to all bins in this span
                                event_bins = []
                                for idx in span_indices:
                                    clone.maternal_cnas[idx] = new_m_cna
                                    clone.paternal_cnas[idx] = new_p_cna
                                    clone.cna_status[idx] = ("post-wgd-dup", len(span_indices), cna_event_id)
                                    event_bins.append(idx)

                                # Record the change
                                event_start_idx = event_bins[0]
                                event_end_idx = event_bins[-1]
                                event_chrom = ref['Chromosome'][event_start_idx]
                                event_start_pos = ref['Start'][event_start_idx]
                                event_end_pos = ref['End'][event_end_idx]
                                
                                if new_m_cna != parent_m_cna:
                                    clone.changes.append([
                                        parent.name, clone.name, 'maternal', 'DUP',
                                        f"{event_chrom}:{event_start_pos}-{event_end_pos}",str(event_end_pos-event_start_pos + 1),
                                        f'{parent_m_cna}->{new_m_cna}'
                                    ])
                                
                                if new_p_cna != parent_p_cna:
                                    clone.changes.append([
                                        parent.name, clone.name, 'paternal', 'DUP',
                                        f"{event_chrom}:{event_start_pos}-{event_end_pos}",str(event_end_pos-event_start_pos + 1),
                                        f'{parent_p_cna}->{new_p_cna}'
                                    ])
                                
                                cna_event_id += 1
                
                else:  # Parent is not WGD, allow all types of mutations on non-mutated regions
                    # Find available indices (where parent has no CNA)
                    available_indices = [i for i in range(total_bin_lens) if parent.cna_status[i] is None]
                    
                    if len(available_indices) > 0:
                        # Select LOH and GOH CNAs
                        cnl_loh_no = int(self.loh_cna_no / 3)
                        
                        cna_types_to_generate = (
                            ["cnl"] * cnl_loh_no + 
                            ["cnn"] * (self.loh_cna_no - cnl_loh_no) + 
                            ["goh"] * self.goh_cna_no
                        )
                        
                        random.shuffle(available_indices)
                        random.shuffle(cna_types_to_generate)
                        
                        cna_generated = 0
                        total_cnas = len(cna_types_to_generate)
                        
                        for start_idx in available_indices:
                            if cna_generated >= total_cnas:
                                break
                            
                            if clone.cna_status[start_idx] != parent.cna_status[start_idx] or parent.cna_status[start_idx] is not None:
                                continue
                            
                            num_windows_span = utils.generate_mixture_poisson(self.weights, self.lambdas)
                            
                            # Get chromosome of the start position
                            start_chrom = ref['Chromosome'][start_idx]
                            
                            # Find valid span that doesn't cross chromosome boundary
                            can_place = True
                            span_indices = []
                            
                            for j in range(num_windows_span):
                                idx = start_idx + j
                                
                                # Check if we've exceeded total bins
                                if idx >= total_bin_lens:
                                    break
                                
                                # Check if we've crossed chromosome boundary
                                if ref['Chromosome'][idx] != start_chrom:
                                    break
                                
                                # Check if position has already been modified or was modified in parent
                                if clone.cna_status[idx] != parent.cna_status[idx] or parent.cna_status[idx] is not None:
                                    can_place = False
                                    break
                                
                                span_indices.append(idx)
                            
                            if can_place and len(span_indices) > 0:
                                cna_type = cna_types_to_generate[cna_generated]
                                for idx in span_indices:
                                    clone.cna_status[idx] = (cna_type, len(span_indices), cna_event_id)
                                cna_event_id += 1
                                cna_generated += 1
                        
                        # Generate regular CNAs on remaining available positions
                        available_indices = [i for i in range(total_bin_lens) if clone.cna_status[i] == parent.cna_status[i] and parent.cna_status[i] is None]
                        
                        i = 0
                        while i < total_bin_lens:
                            if clone.cna_status[i] == parent.cna_status[i] and parent.cna_status[i] is None:
                                if np.random.binomial(1, self.cna_prob):
                                    num_windows_span = utils.generate_mixture_poisson(self.weights, self.lambdas)
                                    
                                    # Get chromosome of the start position
                                    start_chrom = ref['Chromosome'][i]
                                    
                                    # Find valid span that doesn't cross chromosome boundary
                                    can_place = True
                                    span_indices = []
                                    
                                    for j in range(num_windows_span):
                                        idx = i + j
                                        
                                        # Check if we've exceeded total bins
                                        if idx >= total_bin_lens:
                                            break
                                        
                                        # Check if we've crossed chromosome boundary
                                        if ref['Chromosome'][idx] != start_chrom:
                                            break
                                        
                                        # Check if position has already been modified or was modified in parent
                                        if clone.cna_status[idx] != parent.cna_status[idx] or parent.cna_status[idx] is not None:
                                            can_place = False
                                            break
                                        
                                        span_indices.append(idx)
                                    
                                    if can_place and len(span_indices) > 0:
                                        # Place the CNA event
                                        for idx in span_indices:
                                            clone.cna_status[idx] = ("cna", len(span_indices), cna_event_id)
                                        cna_event_id += 1
                                        i += len(span_indices)  # Skip past the CNA we just placed
                                    else:
                                        i += 1
                                else:
                                    i += 1
                            else:
                                i += 1
                    
                    # Process all new CNA events (those not inherited from parent)
                    processed_events = set()
                    
                    i = 0
                    while i < total_bin_lens:
                        # Skip if this is inherited from parent or already processed
                        if clone.cna_status[i] == parent.cna_status[i]:
                            i += 1
                            continue
                        
                        if clone.cna_status[i] is None:
                            i += 1
                            continue
                        
                        # Get current event information
                        cna_type, num_windows, event_id = clone.cna_status[i]
                        
                        # If this event has already been processed, skip it
                        if event_id in processed_events:
                            i += 1
                            continue
                        
                        # Mark this event as processed
                        processed_events.add(event_id)
                        
                        # Find all bin indices for this event
                        event_bins = []
                        for j in range(i, min(i + num_windows, total_bin_lens)):
                            if clone.cna_status[j] is not None and isinstance(clone.cna_status[j], tuple) and clone.cna_status[j][2] == event_id:
                                event_bins.append(j)
                        
                        # Get the start and end positions of the event
                        event_start_idx = event_bins[0]
                        event_end_idx = event_bins[-1]
                        event_chrom = ref['Chromosome'][event_start_idx]
                        event_start_pos = ref['Start'][event_start_idx]
                        event_end_pos = ref['End'][event_end_idx]
                        m_sequence = maternal_genome[event_chrom][event_start_pos-1:event_end_pos]
                        p_sequence = paternal_genome[event_chrom][event_start_pos-1:event_end_pos]
                        
                        # Get parent's copy numbers at this position
                        parent_m_cna = parent.maternal_cnas[event_start_idx]
                        parent_p_cna = parent.paternal_cnas[event_start_idx]
                        
                        # Handle CNL_LOH
                        if cna_type == "cnl":
                            if m_sequence != p_sequence:
                                # CNL_LOH: 1:0 or 0:1 relative to parent baseline
                                if np.random.binomial(1, 0.5):  # Keep maternal, delete paternal
                                    m_cna = parent_m_cna
                                    p_cna = 0
                                else:  # Keep paternal, delete maternal
                                    m_cna = 0
                                    p_cna = parent_p_cna
                                
                                for bin_idx in event_bins:
                                    clone.maternal_cnas[bin_idx] = m_cna
                                    clone.paternal_cnas[bin_idx] = p_cna
                                
                                if m_cna != parent_m_cna:
                                    clone.changes.append([
                                        parent.name, clone.name, 'maternal', 'CNL_LOH',
                                        f"{event_chrom}:{event_start_pos}-{event_end_pos}",str(event_end_pos-event_start_pos + 1),
                                        f'{parent_m_cna}->{m_cna}'
                                    ])
                                
                                if p_cna != parent_p_cna:
                                    clone.changes.append([
                                        parent.name, clone.name, 'paternal', 'CNL_LOH',
                                        f"{event_chrom}:{event_start_pos}-{event_end_pos}",str(event_end_pos-event_start_pos + 1),
                                        f'{parent_p_cna}->{p_cna}'
                                    ])
                        
                        # Handle CNN_LOH
                        elif cna_type == "cnn":
                            if m_sequence != p_sequence:
                                # CNN_LOH: duplication of one allele, deletion of the other
                                cnn_cna = np.clip(np.random.geometric(self.cna_copy_param), 2, int(self.max_cna_value))
                                
                                if np.random.binomial(1, 0.5):  # Maternal duplication, paternal deletion
                                    m_cna = cnn_cna  # Add copies to maintain or increase
                                    p_cna = 0
                                else:  # Paternal duplication, maternal deletion
                                    m_cna = 0
                                    p_cna = cnn_cna
                                
                                for bin_idx in event_bins:
                                    clone.maternal_cnas[bin_idx] = m_cna
                                    clone.paternal_cnas[bin_idx] = p_cna
                                
                                event_type = 'CNN_LOH' if cnn_cna == 2 else 'CNG_LOH'
                                
                                clone.changes.append([
                                    parent.name, clone.name, 'maternal', event_type,
                                    f"{event_chrom}:{event_start_pos}-{event_end_pos}",str(event_end_pos-event_start_pos + 1),
                                    f'{parent_m_cna}->{m_cna}'
                                ])
                                
                                clone.changes.append([
                                    parent.name, clone.name, 'paternal', event_type,
                                    f"{event_chrom}:{event_start_pos}-{event_end_pos}",str(event_end_pos-event_start_pos + 1),
                                    f'{parent_p_cna}->{p_cna}'
                                ])
                        
                        # Handle GOH
                        elif cna_type == "goh":
                            if m_sequence == p_sequence:
                                # GOH: Both alleles gain, but with different copy numbers
                                m_cna = min(np.random.geometric(self.cna_copy_param), self.max_cna_value - 1)
                                if m_cna == 1:
                                    p_cna = min(np.random.geometric(self.cna_copy_param), 2, self.max_cna_value - m_cna)
                                else:
                                    p_cna = min(np.random.geometric(self.cna_copy_param), self.max_cna_value - m_cna)
                                
                                for bin_idx in event_bins:
                                    clone.maternal_cnas[bin_idx] = m_cna
                                    clone.paternal_cnas[bin_idx] = p_cna
                                
                                if m_cna != parent_m_cna:
                                    clone.changes.append([
                                        parent.name, clone.name, 'maternal', 'GOH',
                                        f"{event_chrom}:{event_start_pos}-{event_end_pos}",str(event_end_pos-event_start_pos + 1),
                                        f'{parent_m_cna}->{m_cna}'
                                    ])
                                
                                if p_cna != parent_p_cna:
                                    clone.changes.append([
                                        parent.name, clone.name, 'paternal', 'GOH',
                                        f"{event_chrom}:{event_start_pos}-{event_end_pos}",str(event_end_pos-event_start_pos + 1),
                                        f'{parent_p_cna}->{p_cna}'
                                    ])
                        
                        # Handle regular CNA
                        elif cna_type == "cna":
                            # Generate maternal CNA
                            if np.random.binomial(1, self.del_prob):  # Deletion
                                m_cna = 0
                            else:  # Duplication
                                m_cna = min(np.random.geometric(self.cna_copy_param), int(self.max_cna_value))
                            
                            # Paternal CNA: independently decide deletion or duplication
                            if np.random.binomial(1, self.del_prob):  # Deletion
                                p_cna = 0
                            else:  # Duplication
                                if m_cna == 1:
                                    p_cna = min(np.random.geometric(self.cna_copy_param), 2, int(self.max_cna_value - m_cna))
                                else:
                                    p_cna = min(np.random.geometric(self.cna_copy_param), int(self.max_cna_value - m_cna))
                            
                            for bin_idx in event_bins:
                                clone.maternal_cnas[bin_idx] = m_cna
                                clone.paternal_cnas[bin_idx] = p_cna
                            
                            # Check whether it forms CNL_LOH, CNN_LOH, or CNG_LOH
                            if (m_sequence != p_sequence) and ((m_cna == 0 and p_cna != 0) or (m_cna != 0 and p_cna == 0)):
                                total_cn = m_cna + p_cna
                                if m_cna == 1 or p_cna == 1:
                                    p_event_type = 'CNL_LOH'
                                elif m_cna == 2 or p_cna == 2:
                                    p_event_type = 'CNN_LOH'
                                else:
                                    p_event_type = 'CNG_LOH'
                                if m_cna != 1:
                                    clone.changes.append([
                                        'normal', clone.name, 'maternal', p_event_type,
                                        f"{event_chrom}:{event_start_pos}-{event_end_pos}",str(event_end_pos-event_start_pos + 1),
                                        f'{parent_m_cna}->{m_cna}'
                                    ])
                                if p_cna != 1:
                                    clone.changes.append([
                                        'normal', clone.name, 'paternal', p_event_type,
                                        f"{event_chrom}:{event_start_pos}-{event_end_pos}",str(event_end_pos-event_start_pos + 1),
                                        f'{parent_p_cna}->{p_cna}'
                                    ])
                                continue
                            else:
                                # Record changes for maternal
                                if m_cna != parent_m_cna:
                                    m_event_type = 'DEL' if m_cna == 0 else 'DUP'
                                    clone.changes.append([
                                        parent.name, clone.name, 'maternal', m_event_type,
                                        f"{event_chrom}:{event_start_pos}-{event_end_pos}",str(event_end_pos-event_start_pos + 1),
                                        f'{parent_m_cna}->{m_cna}'
                                    ])
                                
                                # Record changes for paternal
                                if p_cna != parent_p_cna:
                                    p_event_type = 'DEL' if p_cna == 0 else 'DUP'
                                    clone.changes.append([
                                        parent.name, clone.name, 'paternal', p_event_type,
                                        f"{event_chrom}:{event_start_pos}-{event_end_pos}",str(event_end_pos-event_start_pos + 1),
                                        f'{parent_p_cna}->{p_cna}'
                                    ])
                        
                        i += 1
            
            ref[clone.name+'_maternal_cnas'] = clone.maternal_cnas
            ref[clone.name+'_paternal_cnas'] = clone.paternal_cnas
            queue.extend(clone.children)

        return ref, maternal_genome, paternal_genome

    def _generate_cell_cna_profiles(self, root, ref, maternal_genome, paternal_genome):
        """
        Generate cell-level CNA profiles for each clone.
        
        Args:
            root: Root node of the clone tree
            ref: Reference dataframe with genomic bins
            mutation_ratio: Proportion of cells in each clone that will acquire new mutations
            maternal_genome: Dictionary of maternal genome sequences by chromosome
            paternal_genome: Dictionary of paternal genome sequences by chromosome
        
        Returns:
            dict: Dictionary mapping cell IDs to their CNA profiles and changes
        """
        
        total_bin_lens = ref.shape[0]
        cell_profiles = {}
        
        # Process clones in BFS order
        queue = deque([root])
        
        while queue:
            clone = queue.popleft()
            # Determine number of cells with and without mutations
            total_cells = clone.cell_no
            cell_id_counter = 1
            mutated_cell_count = int(total_cells * self.unique_ratio)
            unmutated_cell_count = total_cells - mutated_cell_count
            
            # Generate unmutated cells (inherit clone's CNA profile exactly)
            for i in range(unmutated_cell_count):
                cell_id = f"{clone.name}_cell{cell_id_counter}"
                cell_id_counter += 1
                
                cell_profiles[cell_id] = {
                    'clone': clone.name,
                    'maternal_cnas': clone.maternal_cnas.copy(),
                    'paternal_cnas': clone.paternal_cnas.copy(),
                    'changes': [],
                    'has_mutation': False
                }
                
            if clone.cna_status is not None:
                is_wgd = any('wgd' in status for status in clone.cna_status if status is not None)
            else:
                is_wgd = False

            # Generate mutated cells
            for i in range(mutated_cell_count):
                cell_id = f"{clone.name}_cell{cell_id_counter}"
                cell_id_counter += 1
                
                # Initialize with clone's CNA profile
                cell_maternal_cnas = clone.maternal_cnas.copy()
                cell_paternal_cnas = clone.paternal_cnas.copy()
                cell_changes = []
                
                # Find available indices where clone has no CNA (copy number = 1 for both alleles)
                if is_wgd:
                    available_indices = [
                        idx for idx in range(total_bin_lens) 
                        if clone.cna_status[idx] != 'post-wgd-dup'
                    ]
                else:
                    available_indices = [
                        idx for idx in range(total_bin_lens) 
                        if (clone.cna_status is None) or (clone.maternal_cnas[idx] == 1 and clone.paternal_cnas[idx] == 1 and clone.cna_status[idx] is None)
                    ]
                
                if len(available_indices) == 0:
                    # No available positions for mutation, cell inherits clone profile
                    cell_profiles[cell_id] = {
                        'clone': clone.name,
                        'maternal_cnas': cell_maternal_cnas,
                        'paternal_cnas': cell_paternal_cnas,
                        'changes': cell_changes,
                        'has_mutation': False
                    }
                    continue
                
                # Generate random CNAs for this cell
                random.shuffle(available_indices)
                
                # Track which positions have been assigned CNAs
                cell_cna_status = [None] * total_bin_lens
                cna_event_id = 0
                
                # Iterate through available positions with probability
                i = 0
                while i < len(available_indices):
                    start_idx = available_indices[i]
                    
                    # Decide whether to place a CNA here
                    if clone.name == 'normal' or is_wgd:
                        temp_prob = np.random.binomial(1, self.cna_prob * 0.02)
                    else:
                        temp_prob = np.random.binomial(1, self.cna_prob * 0.1)
                    if temp_prob:
                        num_windows_span = utils.generate_mixture_poisson(self.weights, self.lambdas)
                        
                        # Check if we can place a CNA spanning num_windows_span bins
                        can_place = True
                        span_indices = []
                        
                        for j in range(num_windows_span):
                            start_chrom = ref['Chromosome'][start_idx]
                            potential_idx = start_idx + j
                            if potential_idx >= total_bin_lens:
                                break
                            
                            # Check if we've crossed chromosome boundary
                            if ref['Chromosome'][potential_idx] != start_chrom:
                                break

                            # Check if this position is available (unmutated in clone and not yet assigned)
                            if is_wgd:
                                if clone.cna_status[potential_idx] == 'post-wgd-dup':
                                    can_place = False
                                    break
                            else:
                                if (clone.maternal_cnas[potential_idx] != 1 or 
                                    clone.paternal_cnas[potential_idx] != 1 or 
                                    (clone.cna_status is not None and clone.cna_status[potential_idx] is not None)):
                                    can_place = False
                                    break
                            
                            span_indices.append(potential_idx)
                        
                        if can_place and len(span_indices) > 0:
                            # Mark these positions as having a CNA
                            for idx in span_indices:
                                cell_cna_status[idx] = ("cna", len(span_indices), cna_event_id)
                            clone_m_cna = clone.maternal_cnas[start_idx]
                            clone_p_cna = clone.paternal_cnas[start_idx]
                            if is_wgd:
                                # Generate duplication CNAs (must be greater than current)
                                # Randomly choose to duplicate maternal, paternal, or both
                                mutation_choice = np.random.choice(['maternal', 'paternal', 'both'])
                                
                                if mutation_choice == 'maternal':
                                    m_cna = min(clone_m_cna + np.random.geometric(self.cna_copy_param), int(self.max_cna_value-clone_p_cna))
                                    p_cna = clone_p_cna
                                elif mutation_choice == 'paternal':
                                    m_cna = clone_m_cna
                                    p_cna = min(clone_p_cna + np.random.geometric(self.cna_copy_param), int(self.max_cna_value-clone_m_cna))
                                else:  # both
                                    m_cna = min(clone_m_cna + np.random.geometric(self.cna_copy_param), self.max_cna_value - clone_p_cna)
                                    p_cna = min(clone_p_cna + np.random.geometric(self.cna_copy_param), self.max_cna_value - m_cna)
                            else:
                                # Generate maternal CNA: deletion or duplication
                                if np.random.binomial(1, self.del_prob):  # Deletion
                                    m_cna = 0
                                else:  # Duplication
                                    m_cna = min(np.random.geometric(self.cna_copy_param), int(self.max_cna_value))
                                
                                # Paternal CNA: deletion or duplication
                                if np.random.binomial(1, self.del_prob):  # Deletion
                                    p_cna = 0
                                else:  # Duplication
                                    if m_cna == 1:
                                        p_cna = min(np.random.geometric(self.cna_copy_param), 2, int(self.max_cna_value - m_cna))
                                    else:
                                        p_cna = min(np.random.geometric(self.cna_copy_param), int(self.max_cna_value - m_cna))
                            
                            # Apply the CNA to all bins in the span
                            for idx in span_indices:
                                cell_maternal_cnas[idx] = m_cna
                                cell_paternal_cnas[idx] = p_cna
                            
                            # Get genomic information for this event
                            event_start_idx = span_indices[0]
                            event_end_idx = span_indices[-1]
                            event_chrom = ref['Chromosome'][event_start_idx]
                            event_start_pos = ref['Start'][event_start_idx]
                            event_end_pos = ref['End'][event_end_idx]
                            
                            if is_wgd:
                                if m_cna != clone_m_cna:
                                    cell_changes.append([
                                        clone.name, cell_id, 'maternal', 'DUP',
                                        f"{event_chrom}:{event_start_pos}-{event_end_pos}",str(event_end_pos-event_start_pos + 1),
                                        f'{clone_m_cna}->{m_cna}'
                                    ])
                                
                                if p_cna != clone_p_cna:
                                    cell_changes.append([
                                        clone.name, cell_id,'paternal', 'DUP',
                                        f"{event_chrom}:{event_start_pos}-{event_end_pos}",str(event_end_pos-event_start_pos + 1),
                                        f'{clone_p_cna}->{p_cna}'
                                    ])
                            else:
                                # Get sequences to check if heterozygous
                                m_sequence = None
                                p_sequence = None
                                if maternal_genome is not None and paternal_genome is not None:
                                    m_sequence = maternal_genome[event_chrom][event_start_pos-1:event_end_pos]
                                    p_sequence = paternal_genome[event_chrom][event_start_pos-1:event_end_pos]
                                
                                # Check if this is a LOH event
                                is_loh = False
                                loh_type = None
                                
                                if m_sequence is not None and p_sequence is not None and m_sequence != p_sequence:
                                    # Sequences are different (heterozygous), check for LOH
                                    if (m_cna == 0 and p_cna != 0) or (m_cna != 0 and p_cna == 0):
                                        is_loh = True
                                        total_cn = m_cna + p_cna
                                        
                                        if total_cn == 1:
                                            loh_type = 'CNL_LOH'
                                        elif total_cn == 2:
                                            loh_type = 'CNN_LOH'
                                        else:
                                            loh_type = 'CNG_LOH'
                                
                                # Record the changes based on LOH status
                                if is_loh:
                                    # Record as LOH event
                                    if m_cna != 1:
                                        cell_changes.append([
                                            clone.name, cell_id, 'maternal', loh_type,
                                            f"{event_chrom}:{event_start_pos}-{event_end_pos}",
                                            str(event_end_pos-event_start_pos + 1),
                                            f'1->{m_cna}'
                                        ])
                                    
                                    if p_cna != 1:
                                        cell_changes.append([
                                            clone.name, cell_id, 'paternal', loh_type,
                                            f"{event_chrom}:{event_start_pos}-{event_end_pos}",
                                            str(event_end_pos-event_start_pos + 1),
                                            f'1->{p_cna}'
                                        ])
                                else:
                                    # Record as regular DEL/DUP events
                                    if m_cna != 1:
                                        m_event_type = 'DEL' if m_cna == 0 else 'DUP'
                                        cell_changes.append([
                                            clone.name, cell_id, 'maternal', m_event_type,
                                            f"{event_chrom}:{event_start_pos}-{event_end_pos}",
                                            str(event_end_pos-event_start_pos + 1),
                                            f'1->{m_cna}'
                                        ])
                                    
                                    if p_cna != 1:
                                        p_event_type = 'DEL' if p_cna == 0 else 'DUP'
                                        cell_changes.append([
                                            clone.name, cell_id, 'paternal', p_event_type,
                                            f"{event_chrom}:{event_start_pos}-{event_end_pos}",
                                            str(event_end_pos-event_start_pos + 1),
                                            f'1->{p_cna}'
                                        ])
                            
                            cna_event_id += 1
                    
                    i += 1
                
                # Store the cell profile
                cell_profiles[cell_id] = {
                    'clone': clone.name,
                    'maternal_cnas': cell_maternal_cnas,
                    'paternal_cnas': cell_paternal_cnas,
                    'changes': cell_changes,
                    'has_mutation': True if len(cell_changes) > 0 else False
                }
            
            # Add children to queue
            queue.extend(clone.children)
        
        return cell_profiles

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
                        m_cna = int(row[clone.name+'_maternal_cnas'])
                        p_cna = int(row[clone.name+'_paternal_cnas'])
                        start = int(row['Start'])
                        end = int(row['End'])

                        # handle CNN_LOH
                        cna_type = utils.find_segment_type(changes, clone.name, chrom, start, end)
                        m_sequence = maternal_genome[chrom][start-1:end]
                        p_sequence = paternal_genome[chrom][start-1:end]
                        # if cna_type == 'CNN_LOH':
                        #     if m_cna != 0:
                        #         new_m_cna = random.randint(1, m_cna -1)
                        #         new_p_cna = m_cna - new_m_cna
                        #         cna_m_sequence = m_sequence * new_m_cna
                        #         cna_p_sequence = m_sequence * new_p_cna
                        #     else:
                        #         new_m_cna = random.randint(1, p_cna -1)
                        #         new_p_cna = p_cna - new_m_cna
                        #         cna_m_sequence = p_sequence * new_m_cna
                        #         cna_p_sequence = p_sequence * new_p_cna
                        if cna_type == 'GOH':
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
                            cna_m_sequence = new_m_sequence * m_cna
                            cna_p_sequence = new_p_sequence * p_cna
                        else:   
                            cna_m_sequence = m_sequence * m_cna
                            cna_p_sequence = p_sequence * p_cna
                        m_output.write(cna_m_sequence)
                        p_output.write(cna_p_sequence)
                        clone.maternal_fasta_length += len(cna_m_sequence)
                        clone.paternal_fasta_length += len(cna_p_sequence)
                    m_output.write('\n')
                    p_output.write('\n')

        # merge maternal and paternal fasta
        # clone.fasta = os.path.join(outdir, clone.name+'.fasta')
        # command = """sed '/^>chr/ s/$/-A/' {0} > {1} && sed '/^>chr/ s/$/-B/' {2} >> {1}""".format(clone.maternal_fasta, clone.fasta, clone.paternal_fasta)
        # utils.runcmd(command, self.outdir)
        gfasta_bar.progress(advance=True, msg="Finish generating fasta file for {}".format(clone.name))
        return (clone)

    def _find_mirrored_clones(self, cna_profile):
        """
        Identify rows with mirrored-clone CNAs in a given CNV CSV file, excluding cases where the 
        allele-specific CNVs are equal (e.g., 1|1, 2|2). Consecutive bins are merged into regions.

        Parameters:
            cna_profile: pd.DataFrame

        Returns:
            pd.DataFrame: A DataFrame containing merged regions with mirrored-clone CNAs.
        """
        # Read the CSV file into a pandas DataFrame
        df = cna_profile
        
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
        
        # If no mirrored clones found, return empty DataFrame
        if mirrored_df.empty:
            return mirrored_df
        
        # Sort by Chromosome, Clone1, Clone2, and Start for merging
        mirrored_df = mirrored_df.sort_values(['Chromosome', 'Clone1', 'Clone2', 'Start']).reset_index(drop=True)
        
        # Merge consecutive bins
        merged_rows = []
        current_group = None
        
        for _, row in mirrored_df.iterrows():
            if current_group is None:
                # Start a new group
                current_group = row.to_dict()
            else:
                # Check if this row can be merged with the current group
                can_merge = (
                    current_group['Chromosome'] == row['Chromosome'] and
                    current_group['Clone1'] == row['Clone1'] and
                    current_group['Clone2'] == row['Clone2'] and
                    current_group['Clone1_CNA'] == row['Clone1_CNA'] and
                    current_group['Clone2_CNA'] == row['Clone2_CNA'] and
                    current_group['End'] + 1 == row['Start']  # Consecutive bins
                )
                
                if can_merge:
                    # Extend the end position
                    current_group['End'] = row['End']
                else:
                    # Save the current group and start a new one
                    merged_rows.append(current_group)
                    current_group = row.to_dict()
        
        # Don't forget to add the last group
        if current_group is not None:
            merged_rows.append(current_group)
        
        # Convert merged results to DataFrame
        merged_df = pd.DataFrame(merged_rows)
        
        return merged_df

    def _out_clone_cna_profile(self, root, ref, outdir):
        # out cna profile csv
        changes = []
        df = ref[['Chromosome', 'Start', 'End']]
        queue = deque([root])
        while queue:
            clone = queue.popleft()
            df[clone.name] = ref[clone.name+'_maternal_cnas'].astype(str) + '|' + ref[clone.name+'_paternal_cnas'].astype(str)
            changes += clone.changes
            queue.extend(clone.children)
        df.to_csv(os.path.join(outdir, 'clone_cna_profile.csv'), index=False)



        # out maternal cna matrix
        indexes = ref['Chromosome'] + ':' + ref['Start'].astype(str) + '-' + ref['End'].astype(str)
        m_cna = ref.filter(like='maternal_cnas')
        m_cna.index = indexes
        m_cna.to_csv(os.path.join(outdir, 'clone_maternal_cna_matrix.csv'))

        # out paternal cna matrix
        p_cna = ref.filter(like='paternal_cnas')
        p_cna.index = indexes
        p_cna.to_csv(os.path.join(outdir, 'clone_paternal_cna_matrix.csv'))

        # out changes profile
        columns = ['Parent', 'Child', 'Haplotype', 'Type', 'Segment', 'Length', 'Change']
        change_df = pd.DataFrame(data=changes, columns=columns)
        change_df.to_csv(os.path.join(outdir, 'clone_changes.csv'), index=False)
        ref.to_csv(os.path.join(outdir, 'reference.csv'), index=False)
        return df
    
    def _out_cell_cna_profile(self, cell_profiles, ref, outdir):
        """
        Output cell-level CNA profiles and changes to CSV files.
        
        Args:
            cell_profiles: Dictionary mapping cell IDs to their CNA profiles and changes
            ref: Reference dataframe with genomic bins
            outdir: Output directory path
        """
        
        # Prepare genomic coordinates
        indexes = ref['Chromosome'] + ':' + ref['Start'].astype(str) + '-' + ref['End'].astype(str)
        
        # 1. Output cell_cna_profile.csv
        df = ref[['Chromosome', 'Start', 'End']].copy()
        
        # Add columns for each cell (maternal|paternal format)
        cell_data = {}
        for cell_id, profile in cell_profiles.items():
            maternal_cnas = profile['maternal_cnas']
            paternal_cnas = profile['paternal_cnas']
            cell_data[cell_id] = [f"{m}|{p}" for m, p in zip(maternal_cnas, paternal_cnas)]
        df = pd.concat([df, pd.DataFrame(cell_data)], axis=1)    
        df.to_csv(os.path.join(outdir, 'cell_cna_profile.csv'), index=False)
        
        # 2. Output cell_maternal_cna_matrix.csv
        m_cna_data = {}
        for cell_id, profile in cell_profiles.items():
            m_cna_data[cell_id] = profile['maternal_cnas']
        
        m_cna_df = pd.DataFrame(m_cna_data)
        m_cna_df.index = indexes
        m_cna_df.to_csv(os.path.join(outdir, 'cell_maternal_cna_matrix.csv'))
        
        # 3. Output cell_paternal_cna_matrix.csv
        p_cna_data = {}
        for cell_id, profile in cell_profiles.items():
            p_cna_data[cell_id] = profile['paternal_cnas']
        
        p_cna_df = pd.DataFrame(p_cna_data)
        p_cna_df.index = indexes
        p_cna_df.to_csv(os.path.join(outdir, 'cell_paternal_cna_matrix.csv'))
        
        # 4. Output cell_changes.csv
        all_changes = []
        for cell_id, profile in cell_profiles.items():
            all_changes.extend(profile['changes'])
        
        columns = ['Clone', 'Cell', 'Haplotype', 'Type', 'Segment', 'Length', 'Change']
        change_df = pd.DataFrame(data=all_changes, columns=columns)
        change_df.to_csv(os.path.join(outdir, 'cell_changes.csv'), index=False)

    def _run_wgsim_for_window(self, args):
        """
        Worker function for generating reads for a single bin using wgsim
        
        Parameters:
            args: Tuple of (fasta_file, bin_info, coverage, params)
            
        Returns:
            (r1_fastq, r2_fastq, bin_idx, n_reads) or None if failed
        """
        clone, fasta_file, mode, bin_info, coverage = args
        chrom, start, end, bin_idx = bin_info
        bin_length = end - start
        wgsim_log = os.path.join(self.outdir, 'log/wgsim_log.txt')
        samtools_log = os.path.join(self.outdir, 'log/samtools_log.txt')

        # Calculate number of read pairs needed
        # coverage = (n_reads × 2 × read_length) / bin_length
        n_reads = int(coverage * bin_length / (2 * self.reads_len))
        
        # Skip bins with zero coverage
        if n_reads == 0:
            return None
        
        # Create temporary directory for this bin
        temp_dir = os.path.join(self.outdir, 'tmp')
        temp_fasta = os.path.join(temp_dir, f"{clone}_{mode}_window_{chrom}_{bin_idx}.fa")
        temp_fq1 = os.path.join(temp_dir, f"{clone}_{mode}_reads_{chrom}_{bin_idx}_1.fq")
        temp_fq2 = os.path.join(temp_dir, f"{clone}_{mode}_reads_{chrom}_{bin_idx}_2.fq")
        
        # Extract region using samtools faidx
        region = f"{chrom}:{start+1}-{end}"  # 1-based coordinate
        
        cmd = f"{self.samtools} faidx {fasta_file} {region} > {temp_fasta}"
        utils.runcmd(cmd, samtools_log)

        # Build wgsim command
        command = self.wgsim + " -e {0} -d {1} -s 35 -N {2} -1 {3} -2 {3} -S {4} -r0 -R0 -X0 {5} {6} {7}".format(self.error_rate,self.insertion_size,n_reads,self.reads_len,bin_idx,temp_fasta,temp_fq1,temp_fq2)
        utils.runcmd(command, wgsim_log)
        if os.path.exists(temp_fasta):
            os.remove(temp_fasta)
        return (temp_fq1,temp_fq2)
            
    def _generate_fastq_for_each_clone(self, job):
        """
        Generates biased FASTQ files for a single clone, keeping maternal and paternal files separate.
        """
        (clone, fasta_file, mode, outdir) = job
        self.log("Start generating fastq file for {}".format(clone + '_' + mode), level='PROGRESS')
        
        alpha, beta = utils.lorenz_to_beta(self.lorenz_x, self.lorenz_y)
        bins = utils.generate_bin_regions(fasta_file, self.window_size)

        relative_coverage = utils.sample_coverage_with_correlation( len(bins), alpha, beta, correlation_length=self.correlation_len)
        coverage_per_bin = relative_coverage * self.clone_coverage
        
        # Prepare tasks
        tasks = [
            (clone, fasta_file, mode, bin_info, cov/2)
            for bin_info, cov in zip(bins, coverage_per_bin)
        ]

        with Pool(processes=self.thread) as pool:
            results = pool.map(self._run_wgsim_for_window, tasks)
        all_temp_files = [r for r in results if r is not None]
        # Merge all temporary FASTQ files into final FASTQ files
        fq1 = os.path.join(outdir, f'{clone}_{mode}_r1.fq')
        fq2 = os.path.join(outdir, f'{clone}_{mode}_r2.fq')

        with open(fq1, 'w') as out1, open(fq2, 'w') as out2:
            for temp_fq1, temp_fq2 in all_temp_files:
                if os.path.exists(temp_fq1):
                    with open(temp_fq1, 'r') as f:
                        out1.write(f.read())
                if os.path.exists(temp_fq2):
                    with open(temp_fq2, 'r') as f:
                        out2.write(f.read())
                # Remove temporary files
                if os.path.exists(temp_fq1):
                    os.remove(temp_fq1)
                if os.path.exists(temp_fq2):
                    os.remove(temp_fq2)
        self.log("Finish generating fastq file for {}".format(clone + '_' + mode), level='PROGRESS')

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
        command = "{0} sort -@ {1} {2} -o {3}".format(self.samtools, self.thread, bam_file, sorted_bam_file)
        utils.runcmd(command, samtools_log)

        bam_file = os.path.join(bam_dir, clone+"_paternal.bam")
        sorted_bam_file = os.path.join(bam_dir, clone+"_paternal.sorted.bam")
        command = "{0} sort -@ {1} {2} -o {3}".format(self.samtools, self.thread, bam_file, sorted_bam_file)
        utils.runcmd(command, samtools_log)

        # align_bar.progress(advance=False, msg="Samtools merge maternal and paternal bam for {}".format(clone))
        # clone_bam = os.path.join(bam_dir, clone+".bam")
        # m_sorted_bam_file = os.path.join(bam_dir, clone+"_maternal.sorted.bam")
        # p_sorted_bam_file = os.path.join(bam_dir, clone+"_paternal.sorted.bam")
        # command = "{0} merge -@ {1} -f {2} {3} {4}".format(self.samtools, self.thread, clone_bam, m_sorted_bam_file, p_sorted_bam_file)
        # utils.runcmd(command, samtools_log)

        # clean sam and unsorted bam
        for tmp_file in tmp_files:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
        os.rename(os.path.join(bam_dir, clone+"_maternal.sorted.bam"), os.path.join(bam_dir, clone+"_maternal.bam"))
        os.rename(os.path.join(bam_dir, clone+"_paternal.sorted.bam"), os.path.join(bam_dir, clone+"_paternal.bam"))

        # index bam
        command = "{0} index {1}".format(self.samtools, os.path.join(bam_dir, clone+"_maternal.bam"))
        utils.runcmd(command, samtools_log)
        command = "{0} index {1}".format(self.samtools,  os.path.join(bam_dir, clone+"_paternal.bam"))
        utils.runcmd(command, samtools_log)
        align_bar.progress(advance=True, msg="Finish alignment process for {}".format(clone))

    def _merge_bams_in_batches(self, output_file, input_files, batch_size=1000):
        samtools_log = os.path.join(self.outdir, 'log/samtools_log.txt')
        dtmp = os.path.join(self.outdir, 'tmp')
        temp_merged = []
        
        for i in range(0, len(input_files), batch_size):
            batch = input_files[i:i+batch_size]
            temp_output = os.path.join(dtmp, f"temp_merged_{i}.bam")
            temp_bam_list_file = os.path.join(dtmp, f"temp_{i}_bam_list.txt")

            with open(temp_bam_list_file, 'w') as f:
                for temp_bam_file in batch:
                    f.write(temp_bam_file + '\n')
            merge_cmd = "{0} merge -@ {1} -f -b {2} {3}".format(
                self.samtools, self.thread, temp_bam_list_file, temp_output
            )
            utils.runcmd(merge_cmd, samtools_log)
            temp_merged.append(temp_output)
            if os.path.exists(temp_bam_list_file):
                os.remove(temp_bam_list_file)
        
        if len(temp_merged) == 1:
            os.rename(temp_merged[0], output_file)
        else:
            final_merge_cmd = "{0} merge -@ {1} -f {2} {3}".format(
                self.samtools, self.thread, output_file, ' '.join(temp_merged)
            )
            utils.runcmd(final_merge_cmd, samtools_log)
            
            for f in temp_merged:
                os.remove(f)

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
        tmp_files = [bam_file, query_sorted_bam_file, fixed_sorted_bam_file, rg_dedup_bam_file]

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
        # pbam_bar.progress(advance=False, msg="Samtools markdup for {}".format(cell))
        # command = '{0} markdup -@ {1} -r {2} {3}'.format(self.samtools, self.thread, sorted_bam_file, dedup_bam_file)
        # utils.runcmd(command, samtools_log)

        pbam_bar.progress(advance=False, msg="Samtools index for {}".format(cell))
        command = "{0} index -@ {1} {2}".format(self.samtools, self.thread, sorted_bam_file)
        utils.runcmd(command, samtools_log)

        # clean tmp bam file
        for tmp_file in tmp_files:
            if os.path.exists(tmp_file) and os.path.exists(sorted_bam_file):
                os.remove(tmp_file)
        
        # rename cell bam
        os.rename(sorted_bam_file, bam_file)
        os.rename(sorted_bam_file + '.bai', bam_file + '.bai')
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
        cov_bar.progress(advance=False, msg="Counting reads count on {}".format(cell))
        self._call_bedtools(cell_bam, ref_bed, coverage_file)
        cov_bar.progress(advance=True, msg="Finish reads count on {}".format(cell))

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

    def _merge_consecutive_bins(self, bins):
        """
        Merge consecutive bins into segments
        
        Parameters:
        -----------
        bins : list
            List of bins, format like ['chr1:1-100000', 'chr1:100001-200000', ...]
        
        Returns:
        --------
        list : List of merged region strings ['chr1:1-200000', ...]
        """
        if len(bins) == 0:
            return []
        
        segments = []
        current_segment = None
        
        for bin_range in bins:
            # Parse bin
            chrom, pos = bin_range.split(':')
            start, end = pos.split('-')
            start = int(start)
            end = int(end)
            
            if current_segment is None:
                # Start new segment
                current_segment = {
                    'chrom': chrom,
                    'start': start,
                    'end': end
                }
            elif (current_segment['chrom'] == chrom and 
                current_segment['end'] + 1 == start):  # Check if consecutive
                # Merge into current segment
                current_segment['end'] = end
            else:
                # Save current segment and start new one
                region_str = f"{current_segment['chrom']}:{current_segment['start']}-{current_segment['end']}"
                segments.append(region_str)
                
                current_segment = {
                    'chrom': chrom,
                    'start': start,
                    'end': end
                }
        
        # Add the last segment
        if current_segment is not None:
            region_str = f"{current_segment['chrom']}:{current_segment['start']}-{current_segment['end']}"
            segments.append(region_str)
        
        return segments

    def _group_segments_by_cnv_ratio(self, bins, clone_cnv, cell_cnv):
        """
        Group bins into segments based on CNV ratio
        
        Strategy:
        1. Group bins by CNV ratio (clone_cnv == cell_cnv as one group, others by ratio)
        2. Within each group, merge consecutive bins into segments
        
        Parameters:
        -----------
        bins : list
            List of bins, format like ['chr1:1-100000', 'chr1:100001-200000', ...]
        clone_cnv : list or np.array
            CNV values of clone
        cell_cnv : list or np.array
            CNV values of cell
        
        Returns:
        --------
        dict : Grouped segments by ratio
            Key: ratio value (or 'same' for clone_cnv == cell_cnv)
            Value: dict with 'segments', 'clone_cnv', 'cell_cnv'
        """
        if len(bins) == 0:
            return {}
        
        # First, group bins by their CNV characteristics (preserve order)
        temp_groups = {}
        
        for i, bin_range in enumerate(bins):
            clone_cnv_val = clone_cnv[i]
            cell_cnv_val = cell_cnv[i]
            
            # Skip zero values
            if clone_cnv_val == 0 or cell_cnv_val == 0:
                continue
            
            # Determine the group key
            if clone_cnv_val == cell_cnv_val:
                group_key = 'same'
            else:
                ratio = cell_cnv_val / clone_cnv_val
                group_key = f'ratio_{ratio:.6f}'
            
            if group_key not in temp_groups:
                temp_groups[group_key] = {
                    'bins': [],
                    'clone_cnv': clone_cnv_val,
                    'cell_cnv': cell_cnv_val
                }
            
            temp_groups[group_key]['bins'].append(bin_range)
        
        # Second, merge consecutive bins within each group into segments
        groups = {}
        for group_key, group_data in temp_groups.items():
            segments = self._merge_consecutive_bins(group_data['bins'])
            groups[group_key] = {
                'segments': segments,
                'clone_cnv': group_data['clone_cnv'],
                'cell_cnv': group_data['cell_cnv']
            }
        
        return groups

    def _downsampling_cell_bam(self, job):
        """Optimized version: grouping by CNV ratio + merging consecutive bins"""
        (clone, cell, mode, clone_bam_file, clone_cnv, cell_cnv, bins, cell_index) = job

        samtools_log = os.path.join(self.outdir, 'log/samtools_log.txt')
        dcell = os.path.join(self.outdir, 'cell_bams')
        dtmp = os.path.join(self.outdir, 'tmp')

        downsam_bar.progress(advance=False, msg=f"Downsampling cell bam for {cell} ({mode})")

        if clone_cnv == cell_cnv:
            # Simple downsampling for the whole bam
            cell_bam_file = os.path.join(dcell, f'{cell}_{mode}.bam')
            ratio = self.cell_coverage / self.clone_coverage
            random_seed = cell_index + ratio
            command = "{0} view -b -@ {1} -s {2} {3} -o {4}".format(
                self.samtools, 
                self.thread,
                random_seed, 
                clone_bam_file, 
                cell_bam_file
            )
            utils.runcmd(command, samtools_log)
        else:
            # Group bins by CNV ratio and merge consecutive bins
            groups = self._group_segments_by_cnv_ratio(bins, clone_cnv, cell_cnv)
            temp_bam_files = []
            tasks = []
            
            for group_idx, (group_key, group_data) in enumerate(groups.items()):
                segments = group_data['segments']
                clone_cnv_value = group_data['clone_cnv']
                cell_cnv_value = group_data['cell_cnv']
                
                # Calculate ratio
                if group_key == 'same':
                    # For regions where clone_cnv == cell_cnv
                    ratio = self.cell_coverage / self.clone_coverage
                else:
                    # For regions where clone_cnv != cell_cnv
                    ratio = (cell_cnv_value / clone_cnv_value) * (self.cell_coverage / self.clone_coverage)
                
                if ratio <= 1:
                    # Simple downsampling for all segments in this group
                    temp_bam_file = os.path.join(dtmp, f"{cell}_{mode}_group{group_idx:05d}.bam")
                    random_seed = cell_index + ratio
                    
                    # Build samtools command with multiple segments
                    segments_str = ' '.join(segments)
                    command = "{0} view -b -@ {1} -s {2} {3} {4} -o {5}".format(
                        self.samtools, 
                        self.thread,
                        random_seed, 
                        clone_bam_file, 
                        segments_str,
                        temp_bam_file
                    )
                    tasks.append((command, samtools_log))
                    temp_bam_files.append(temp_bam_file)
                    
                else:
                    # ratio > 1: need multiple sampling
                    full_copies = int(ratio)
                    fractional_part = ratio - full_copies
                    
                    segments_str = ' '.join(segments)
                    
                    # Full copies
                    for i in range(full_copies):
                        temp_bam = os.path.join(dtmp, f"{cell}_{mode}_group{group_idx:05d}_copy{i}.bam")
                        command = "{0} view -b -@ {1} {2} {3} -o {4}".format(
                            self.samtools,
                            self.thread,
                            clone_bam_file, 
                            segments_str,
                            temp_bam
                        )
                        tasks.append((command, samtools_log))
                        temp_bam_files.append(temp_bam)
                    
                    # Fractional part
                    if fractional_part > 0:
                        temp_bam = os.path.join(dtmp, f"{cell}_{mode}_group{group_idx:05d}_frac.bam")
                        random_seed = cell_index + fractional_part + full_copies
                        command = "{0} view -b -@ {1} -s {2} {3} {4} -o {5}".format(
                            self.samtools,
                            self.thread,
                            random_seed,
                            clone_bam_file, 
                            segments_str,
                            temp_bam
                        )
                        tasks.append((command, samtools_log))
                        temp_bam_files.append(temp_bam)
            
            for task in tasks:
                command, log_file = task
                utils.runcmd(command, log_file)
            
            # Merge all temp bam files
            cell_bam_file = os.path.join(dcell, f'{cell}_{mode}.bam')
            self._merge_bams_in_batches(cell_bam_file, temp_bam_files, batch_size=1000)

            # Clean up temporary files
            for temp_bam_file in temp_bam_files:
                if os.path.exists(temp_bam_file):
                    os.remove(temp_bam_file)

        downsam_bar.progress(advance=True, msg=f"Finished downsampling cell bam for {cell} ({mode})")

    @utils.log_runtime
    def gprofile(self):
        self.log('Setting directories', level='PROGRESS')
        dprofile, dfasta, dfastq, dclone, dcell, dbarcode, drdr, dbaf, dtmp, dlog = self.setup_dir()

        # set related files
        m_fasta = os.path.join(dfasta, 'reference_maternal.fasta')
        p_fasta = os.path.join(dfasta, 'reference_paternal.fasta')
        # phase_file = os.path.join(dprofile, 'phases.tsv')
        allele_phase_file = os.path.join(dprofile, 'snp_phases.csv')
        tree_newick = os.path.join(dprofile, 'tree.newick')
        tree_pdf = os.path.join(dprofile, 'tree.pdf')
        tree_json = os.path.join(dprofile, 'tree.json')

        # generate random clone tree and set root as normal clone
        if self.tree_newwick and os.path.exists(self.tree_newwick):
            self.log('Loading tree from newick file: {}'.format(self.tree_newwick), level='PROGRESS')
            newwick_str = ''
            with open(self.tree_newwick, 'r') as f:
                newick_string = f.read()
            root = random_tree.load_tree_from_newick(newick_string)
        else:
            self.log('Generating random cell-lineage tree...', level='PROGRESS')
            root = random_tree.generate_tree_beta(cell_num=self.cell_no, num_clones=self.clone_no-1, alpha=self.tree_alpha, beta=self.tree_beta, treedepth=self.max_tree_depth, treedepthsigma=self.tree_depth_sigma, max_children=self.max_node_children, balance_factor=self.tree_balance_factor, mode=self.tree_mode)

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
        self.log("Building reference fasta file with SNPs data...", level='PROGRESS')
        self._buildGenome(m_fasta, p_fasta, allele_phase_file) 

        # generate cna for each clone
        self.log('Generating CNV profile for each clone...', level='PROGRESS')
        loop_no = 1
        unique_mirrored_subclonal_cnas_no = 0
        while unique_mirrored_subclonal_cnas_no < 3 and loop_no < 5:
            ref = self._split_chr_to_bins('all')
            new_ref, maternal_genome, paternal_genome = self._generate_cna_profile_for_each_clone(root, ref, m_fasta, p_fasta)
            cell_profiles = self._generate_cell_cna_profiles(root, new_ref,maternal_genome, paternal_genome)
            cna_profile = self._out_clone_cna_profile(root, new_ref, dprofile)
            cell_profiles = self._out_cell_cna_profile(cell_profiles, new_ref, dprofile)

            mirrored_subclonal_cnas = self._find_mirrored_clones(cna_profile)
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
        changes_file = os.path.join(dprofile, 'clone_changes.csv')
        maternal_fasta = os.path.join(dfasta, 'reference_maternal.fasta')
        paternal_fasta = os.path.join(dfasta, 'reference_paternal.fasta')

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
        clone_fasta_dict = {}
        for clone in pool.imap_unordered(self._generate_fasta_for_each_clone, jobs):
            clone_fasta_dict[clone.name] = {
                'maternal_fasta': clone.maternal_fasta,
                'paternal_fasta': clone.paternal_fasta,
                'maternal_fasta_length': clone.maternal_fasta_length,
                'paternal_fasta_length': clone.paternal_fasta_length
            }
        pool.close()
        pool.join()

        # update clone fasta file info in tree
        for clone in random_tree.collect_all_nodes(root):
            clone.maternal_fasta = clone_fasta_dict[clone.name]['maternal_fasta']
            clone.paternal_fasta = clone_fasta_dict[clone.name]['paternal_fasta']
            clone.maternal_fasta_length = clone_fasta_dict[clone.name]['maternal_fasta_length']
            clone.paternal_fasta_length = clone_fasta_dict[clone.name]['paternal_fasta_length']

        self.log('Storing the tree to json file...', level='PROGRESS')
        random_tree.save_tree_to_file(root, tree_json)
        self.log('gfasta BYEBYE')

    @utils.log_runtime
    def gfastq(self):
        self.log('Setting directories', level='PROGRESS')
        dprofile, dfasta, dfastq, dclone, dcell, dbarcode, drdr, dbaf, dtmp, dlog = self.setup_dir()

        tree_json = os.path.join(dprofile, 'tree.json')
    
        utils.check_exist(tree_json=tree_json)

        # load object from file
        root = random_tree.load_tree_from_file(tree_json)
        all_clones = random_tree.collect_all_nodes(root)

        jobs = []
        # check fasta file for each clone
        for clone in all_clones:
            utils.check_exist(maternal_fasta=clone.maternal_fasta)
            utils.check_exist(paternal_fasta=clone.paternal_fasta)
            jobs.append((clone.name, clone.maternal_fasta, 'maternal', dfastq))
            jobs.append((clone.name, clone.paternal_fasta, 'paternal', dfastq))

        for job in jobs:
            self._generate_fastq_for_each_clone(job)
        # # set parallel jobs for each clone
        # lock = Lock()
        # counter = Value('i', 0)
        # init_args = (lock, counter, len(jobs))
        # pool = Pool(processes=1, initializer=init_gfastq, initargs=init_args)
        
        # self.log('Generating fastq file for each clone...', level='PROGRESS')
        # for _ in pool.imap_unordered(self._generate_fastq_for_each_clone, jobs):
        #     pass
        # pool.close()
        # pool.join()

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
        all_clones = random_tree.collect_all_nodes(root)

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
        
        # Load object from file
        root = random_tree.load_tree_from_file(tree_json)
        all_clones = random_tree.collect_all_nodes(root)

        # Assign cells for each clone and generate job list
        barcodes = []
        jobs = []
                
        for mode in ['maternal', 'paternal']:
            clone_cnv_df = pd.read_csv(os.path.join(dprofile, f'clone_{mode}_cna_matrix.csv'), index_col=0)
            cell_cnv_df = pd.read_csv(os.path.join(dprofile, f'cell_{mode}_cna_matrix.csv'), index_col=0)
            for clone in all_clones:
                # Downsample maternal and paternal bam file for each cell
                clone_bam_file = os.path.join(dclone, f'{clone.name}_{mode}.bam')
                clone_cnv = clone_cnv_df[f'{clone.name}_{mode}_cnas'].tolist()
                bins = clone_cnv_df.index.tolist()
                for i in range(clone.cell_no):
                    cell_name = clone.name + '_cell' + str(i+1)
                    cell_cnv = cell_cnv_df[cell_name].tolist()
                    jobs.append((clone.name, cell_name, mode, clone_bam_file, clone_cnv, cell_cnv, bins, i))
                    barcodes.append(cell_name)
                
        lock = Lock()
        counter = Value('i', 0)
        init_args = (lock, counter, len(jobs))
        pool = Pool(processes=1, initializer=init_downsam, initargs=init_args)
        
        self.log('Processing cell bam...', level='PROGRESS')
        for _ in pool.imap_unordered(self._downsampling_cell_bam, jobs):
            pass
        pool.close()
        pool.join()
        
        # Merge cell maternal and paternal bam files
        self.log('Merging maternal and paternal cell bam files...', level='PROGRESS')
        unique_barcodes = sorted(set(barcodes))
        
        for cell in unique_barcodes:
            maternal_bam = os.path.join(dcell, f'{cell}_maternal.bam')
            paternal_bam = os.path.join(dcell, f'{cell}_paternal.bam')
            merged_bam = os.path.join(dcell, f'{cell}.bam')
            samtools_log = os.path.join(dlog, 'samtools_log.txt')
            
            command = "{0} merge -@ {1} -f {2} {3} {4}".format(
                self.samtools, self.thread, merged_bam, maternal_bam, paternal_bam)
            utils.runcmd(command, samtools_log)
            
            # Index merged bam file
            command = "{0} index -@ {1} {2}".format(self.samtools, self.thread, merged_bam)
            utils.runcmd(command, samtools_log)
        
            if os.path.exists(merged_bam):
                os.remove(maternal_bam)
                os.remove(paternal_bam)
            if os.path.exists(merged_bam + '.bai'):
                if os.path.exists(maternal_bam + '.bai'):
                    os.remove(maternal_bam + '.bai')
                if os.path.exists(paternal_bam + '.bai'):
                    os.remove(paternal_bam + '.bai')
        self.log('Writing cell list to barcode.txt...', level='PROGRESS')
        barcodes_file = os.path.join(dprofile, 'barcodes.txt')
        with open(barcodes_file, 'w') as output:
            for barcode in sorted(set(barcodes)):
                output.write(barcode+'\n')

        self.log('Storing the tree to json file...', level='PROGRESS')
        random_tree.save_tree_to_file(root, tree_json)
        self.log('downsam COMPLETE!')

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
        ref = ref[['Chromosome', 'Start', 'End']]
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
        ref = ref[['Chromosome', 'Start', 'End']]
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
import os
import sys
import random
import datetime
import subprocess as sp
import pandas as pd
import numbers
from collections import defaultdict

import numpy as np
import pandas as pd
import time
import functools

from scipy.stats import beta as beta_dist

# check part
def check_exist(**params):
    """Check that files are exist as expected

    Raises
    ------
    ValueError : unacceptable choice of parameters
    """
    for p in params:
        if not os.path.exists(params[p]):
            raise ValueError(
                "{} file or directory {} does not exist.".format(p, params[p]))

def check_positive(**params):
    """Check that parameters are positive as expected

    Raises
    ------
    ValueError : unacceptable choice of parameters
    """
    for p in params:
        if params[p] <= 0:
            raise ValueError(
                "Expected {} > 0, got {}".format(p, params[p]))

def check_lt_zero(**params):
    """Check that parameters are larger than zero as expected

    Raises
    ------
    ValueError : unacceptable choice of parameters
    """
    for p in params:
        if params[p] < 0:
            raise ValueError(
                "Expected {} >= 0, got {}".format(p, params[p]))


def check_int(**params):
    """Check that parameters are integers as expected

    Raises
    ------
    ValueError : unacceptable choice of parameters
    """
    for p in params:
        if not isinstance(params[p], numbers.Integral):
            raise ValueError(
                "Expected {} integer, got {}".format(p, params[p]))


def check_bool(**params):
    """Check that parameters are bools as expected

    Raises
    ------
    ValueError : unacceptable choice of parameters
    """
    for p in params:
        if params[p] is not True and params[p] is not False:
            raise ValueError(
                "Expected {} boolean, got {}".format(p, params[p]))


def check_between(v_min, v_max, **params):
    """Checks parameters are in a specified range

    Parameters
    ----------

    v_min : float, minimum allowed value (inclusive)

    v_max : float, maximum allowed value (inclusive)

    params : object
        Named arguments, parameters to be checked

    Raises
    ------
    ValueError : unacceptable choice of parameters
    """
    for p in params:
        if params[p] < v_min or params[p] > v_max:
            raise ValueError("Expected {} between {} and {}, "
                             "got {}".format(p, v_min, v_max, params[p]))

def check_in(choices, **params):
    """Checks parameters are in a list of allowed parameters
    Parameters
    ----------
    choices : array-like, accepted values
    params : object
        Named arguments, parameters to be checked
    Raises
    ------
    ValueError : unacceptable choice of parameters
    """
    for p in params:
        if params[p] not in choices:
            raise ValueError(
                "{} value {} not recognized. Choose from {}".format(
                    p, params[p], choices))

def randomSNPList(chrom_sizes, snp_ratio):
    snpList = {}
    for chrom, chrom_len in chrom_sizes.items():
        snpList[chrom] = {snp : random.sample(['A','T','C','G'], 2) for snp in random.sample(range(1, chrom_len+1), int(round(chrom_len * snp_ratio)))}
    return snpList

def parseSNPList(snpfile):
    snpList = {}
    with open(snpfile, 'r') as input:
        for line in input:
            info = line.strip().split('\t')
            chrom = info[0]
            pos = info[1]
            allele1 = info[2].upper()
            allele2 = info[3].upper()
            if allele1 not in ['A', 'C', 'G', 'T'] or allele2 not in ['A', 'C', 'G', 'T']:
                continue
            if chrom not in snpList:
                snpList[chrom] = {}
            snpList[chrom][pos] = (allele1, allele2)
    return snpList

def parseIgnoreList(ignorefile):
    ignorelist = []
    with open(ignorefile, 'r') as input:
        for line in input:
            if line != '':
                ignorelist.append(line.strip())
    return ignorelist

def random_cna():
    # 定义数字和对应的概率e
    numbers = [0, 2, 3, 4, 5]
    probabilities = [0.2, 0.3, 0.25, 0.15, 0.1]

    # 使用choices函数进行随机选择，weights参数指定概率
    return random.choices(numbers, weights=probabilities)[0]

def random_mirrored_cna():
    # 定义数字和对应的概率e
    numbers = [2, 3, 4, 5, 6, 7, 8, 9]
    probabilities = [0.05, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05, 0.05]

    # 使用choices函数进行随机选择，weights参数指定概率
    return random.choices(numbers, weights=probabilities)[0]

def random_WGD():
    # 定义数字和对应的概率e
    numbers = [2, 3, 4, 5]
    probabilities = [0.5, 0.3, 0.15, 0.05]

    # 使用choices函数进行随机选择，weights参数指定概率
    return random.choices(numbers, weights=probabilities)[0]

def random_CNL():
    # 定义数字和对应的概率e
    numbers = [1, 2, 3, 4, 5]
    probabilities = [0.4, 0.3, 0.15, 0.1, 0.05]

    # 使用choices函数进行随机选择，weights参数指定概率
    return random.choices(numbers, weights=probabilities)[0]

def assign_cells_to_clones(cell_no, clone_no):
    # # Ensure at least one cell for each clone
    # clones = [1] * clone_no
    # remaining_cells = cell_no - clone_no

    # # Assign remaining cells to clones
    # for _ in range(remaining_cells):
    #     clone_index = random.randint(0, clone_no - 1)
    #     clones[clone_index] += 1

    # return clones
    # Initialize a list to hold the number of cells assigned to each clone
    clones = [0] * clone_no
    
    # Distribute the cells to the clones
    for i in range(cell_no):
        # Determine which clone to assign the current cell to
        clone_index = i % clone_no
        # Increment the cell count for the determined clone
        clones[clone_index] += 1
    
    return clones

def random_sampling(m, n, k):
    # m for total reads no, n for cell no, k for per reads no in cell
    if n * k < m:
        # Randomly sample without replacement
        sampled_lines = random.sample(range(m), n * k)
    else:
        # Randomly sample with replacement
        sampled_lines = [random.randint(0, m - 1) for _ in range(n * k)]

    # Reshape the sampled lines into a 2D array
    sampled_lines_2d = [sorted(sampled_lines[i:i+k]) for i in range(0, len(sampled_lines), k)]
    
    # add three lines to each line index
    new_sampled_lines_2d = []
    for lines in sampled_lines_2d:
        temp = []
        for line in lines:
            temp += [line*4+1, line*4+2, line*4+3, line*4+4]
        new_sampled_lines_2d.append(temp)
    return new_sampled_lines_2d

def root_path():
    return os.path.dirname(os.path.abspath(__file__))

class bcolors:
    HEADER = '\033[95m'  # 用于高亮显示标题或头部信息的颜色代码，通常为浅紫色。
    OKBLUE = '\033[94m'  # 一种蓝色，可能用于表示正常或信息性的消息。
    BBLUE = '\033[96m'  # 另一种蓝色，可能用于表示警告或其他需要引起注意的信息。
    OKGREEN = '\033[92m'  # 用于表示成功或正面的消息，通常为绿色。
    WARNING = '\033[93m'  # 用于表示警告或需要注意的信息，通常为黄色。
    FAIL = '\033[91m'  # 用于表示错误或失败的信息，通常为红色。
    ENDC = '\033[0m'  # 用于重置终端文本颜色，使其恢复到默认颜色。
    BOLD = '\033[1m'  # 用于加粗文本。
    UNDERLINE = '\033[4m'  # 用于下划线


# ProgressBar 在命令行界面显示进度条，向用户提供任务完成的百分比和状态更新。
class ProgressBar:

    def __init__(self, total, length, lock=None, counter=0, verbose=False, decimals=1, fill=chr(9608),
                 prefix='Progress:', suffix='Complete'):
        """
        :param total: 任务的总步骤数。
        :param length: 进度条的长度，以字符为单位。
        :param lock: 用于线程安全的锁，如果进度条在多线程环境中使用，需要这个参数。
        :param counter: 当前任务完成的步骤数，默认为0。
        :param verbose: 是否显示详细的信息，默认为False。
        :param decimals: 进度百分比的小数位数。
        :param fill: 进度条中填充部分的字符，默认为 █ ,这是一个标准的块状字符
        :param prefix: 进度条前缀，默认为"Progress:"。
        :param suffix: 进度条后缀，默认为"Complete"。
        """
        self.total = total
        self.length = length
        self.decimals = decimals
        self.fill = fill
        self.prefix = prefix
        self.suffix = suffix
        self.lock = lock
        self.counter = counter
        assert lock is not None or counter == 0
        self.verbose = verbose

    def progress(self, advance=True, msg=""):
        """
        更新进度条的状态。
        :param advance: 是否增加计数器，进度增加，默认为True。
        :param msg: 要显示的附加信息。
        """
        if self.lock is None:
            self.progress_unlocked(advance, msg)
        else:
            self.progress_locked(advance, msg)
        return True

    def progress_unlocked(self, advance, msg):
        """
        更新进度条的状态，不使用锁。
        :param advance: 是否增加计数器，进度增加，默认为True。
        :param msg: 要显示的附加信息。
        :return:
        """
        flush = sys.stderr.flush
        write = sys.stderr.write
        if advance:
            self.counter += 1
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * (self.counter / float(self.total)))
        filledLength = int(self.length * self.counter // self.total)
        bar = self.fill * filledLength + '-' * (self.length - filledLength)
        rewind = '\x1b[2K\r'
        result = f'{self.prefix} |{bar}| {percent}% {self.suffix}'
        msg = f'[{datetime.datetime.now():%Y-%b-%d %H:%M:%S}] {msg}'
        if not self.verbose:
            toprint = rewind + result + f" [{msg}]"
        else:
            toprint = rewind + msg + "\n" + result
        write(toprint)
        flush()
        if self.counter == self.total:
            write("\n")
            flush()

    def progress_locked(self, advance, msg):
        """
        更新进度条的状态，*使用锁。
        :param advance: 是否增加计数器，进度增加，默认为True。
        :param msg: 要显示的附加信息。
        :return:
        """
        flush = sys.stderr.flush
        write = sys.stderr.write
        # 1. 更新操作
        if advance:
            # 1.1.使用 self.counter.get_lock() 获取锁，确保线程安全地更新计数器。
            with self.counter.get_lock():
                # 1.2.进度条的当前进度
                self.counter.value += 1
        # 2.计算进度条的完成百分比，使用格式化字符串将结果保留到指定的小数位数。
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * (self.counter.value / float(self.total)))
        # 3.计算进度条中填充部分的长度，使用整数除法 // 将计数器的值除以总步骤数，然后乘以进度条的长度。
        filledLength = int(self.length * self.counter.value // self.total)
        # 4.bar 根据 filledLength 构建进度条的字符串表示：
        # 使用 self.fill 指定的字符填充已完成部分，使用 '-' 表示未完成部分。
        bar = self.fill * filledLength + '-' * (self.length - filledLength)
        # # 5.rewind 是一个 ANSI 转义序列，用于清除当前行并返回行首，这样每次更新进度条时都能在同一行显示
        rewind = '\x1b[2K\r'
        # 6.result 构建进度条的字符串，包括前缀、进度条本身、完成百分比和后缀。
        # msg 格式化消息字符串，包括当前的时间戳和传入的 msg 参数。
        result = f'{self.prefix} |{bar}| {percent}% {self.suffix}'
        msg = f'[{datetime.datetime.now():%Y-%b-%d %H:%M:%S}] {msg}'
        # 7.根据 self.verbose 的值，决定是否在进度条前显示额外的详细信息。
        # 如果False，只显示进度条和进度条提示消息；如果True，先显示工作函数的处理消息，然后是进度条。
        if not self.verbose:
            toprint = rewind + result + f" [{msg}]"
        else:
            toprint = rewind + msg + "\n" + result
        # 8.使用 self.lock 确保在写入和刷新时不会与其他线程冲突。
        with self.lock:
            sys.stderr.write(toprint)
            sys.stderr.flush()
            # 9.如果 self.counter.value 等于 self.total，表示任务已经完成。
            # 在任务完成后，输出一个换行符并刷新标准错误流，以确保进度条后面没有残留的内容。
            if self.counter.value == self.total:
                write("\n")
                flush()

# def error(msg):
#     log(msg=msg, level="ERROR")
#     sys.exit(0)

def which(program):
    """
    查找并返回一个可执行文件的路径。如果找到了，它就返回该文件的路径；如果没有找到，它就返回None。
    :param program:
    :return:
    """
    import os
    def is_exe(fpath):
        # 检查给定的文件路径是否存在并且具有执行权限。
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)

    # 提供了完整路径（fpath不为空）
    if fpath:
        # 该路径下的文件是否存在且可执行
        if is_exe(program):
            return program
    else:
        # 函数会搜索环境变量PATH中列出的所有目录
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            # 寻找第一个匹配且可执行的文件。
            if is_exe(exe_file):
                return exe_file

    return None

def runcmd(cmd, log):
    '''
    simple version of run command
    run shell command and write error log
    '''
    with open(log, 'a') as errlog:
        try:
            # Run the command
            proc = sp.Popen(
                cmd,
                shell=True,  # Allows the use of shell-specific syntax
                stdout=sp.PIPE,  # Captures standard output
                stderr=errlog  # Redirects standard error to the log file
            )
            proc.wait()  # Wait for the process to complete
            
        except Exception as e:
            print(f"Error executing command: {cmd}\n{e}")

def read_phase(allele_phase_file):
    # Use defaultdict to create a dictionary where each key maps to another dictionary
    phased = {}
    
    # Read the allele phase file use pandas
    phased_df = pd.read_csv(allele_phase_file, header=None)
    for index, row in phased_df.iterrows():
        chrom, pos, phase = row[0], row[1], row[4]
        phased[(chrom, pos)] = phase
    
    return phased

def process_snp_count_file(snp_count_file, phased_snps):
    # process the SNP count file
    snps_counts = {}
    with open(snp_count_file, 'r') as input:
        for line in input:
            fields = line.split("\t")
            chrom, pos, ref, alt, ad = fields[:5]
            pos = int(pos)
            ad = list(map(int, ad.split(",")))  # Convert AD values to integers

            # Skip positions not in phased SNPs
            if (chrom, pos) not in phased_snps:
                continue

            # Determine A and B alleles based on the phase
            phase = phased_snps[(chrom, pos)]
            a_count, b_count = 0, 0

            if phase == "0|1":
                # A allele is REF, B allele is ALT
                a_count = ad[0]
                b_count = ad[1] if len(ad) > 1 else 0
            elif phase == "1|0":
                # A allele is ALT, B allele is REF
                a_count = ad[1] if len(ad) > 1 else 0
                b_count = ad[0]

            # Store counts
            snps_counts[(chrom, pos)] = (a_count, b_count)
    return snps_counts

def write_snp_bed_file(snp_bed_file, phased_snps):
    # Write the phased SNPs to a BED file
    with open(snp_bed_file, 'w') as output:
        for snp, phase in phased_snps.items():
            chrom, pos = snp
            # phased snp is 1-based and the BED file must use 0-based start positions and 1-based end positions
            output.write('{}\t{}\t{}\n'.format(chrom, pos-1, pos))

def assign_bin(row, ref):
    # Find the bin where the SNP belongs
    bin_row = ref[
        (ref['Chromosome'] == row['chrom']) & 
        (ref['Start'] <= row['pos']) & 
        (ref['End'] >= row['pos'])
    ]
    if not bin_row.empty:
        return f"{row['chrom']}:{bin_row.iloc[0]['Start']}-{bin_row.iloc[0]['End']}"
    return None

def merge_count_files(count_files, phased_snps, ref):
    # merge count files to a matrix, row index is chrom-pos, column index is cell
    a_allele_counts = []
    b_allele_counts = []
    for count_file in count_files:
        # get cell name from path  of count file
        cell = os.path.basename(count_file).split(".")[0]
        snps_counts = process_snp_count_file(count_file, phased_snps)
        for snp, counts in snps_counts.items():
            chrom, pos = snp
            index = f"{chrom}-{pos}"
            a_allele_counts.append((chrom, pos, cell, counts[0]))
            b_allele_counts.append((chrom, pos, cell, counts[1]))
    a_allele_counts = pd.DataFrame(a_allele_counts, columns=['chrom', 'pos', 'cell', 'count'])
    b_allele_counts = pd.DataFrame(b_allele_counts, columns=['chrom', 'pos', 'cell', 'count'])

    a_allele_counts['bin'] = a_allele_counts.apply(assign_bin, axis=1, ref=ref)

    # Step 4: Group by bins and cells, summing the counts
    a_allele_counts_grouped = a_allele_counts.groupby(['bin', 'cell'])['count'].sum().reset_index()
    b_allele_counts_grouped = b_allele_counts.groupby(['bin', 'cell'])['count'].sum().reset_index()

    # Step 5: Pivot to create the count matrix
    a_allele_martrix = a_allele_counts_grouped.pivot(index='bin', columns='cell', values='count')
    b_allele_martrix = b_allele_counts_grouped.pivot(index='bin', columns='cell', values='count')

    # Step 6: Fill missing values with 0
    a_allele_martrix = a_allele_martrix.fillna(0)
    b_allele_martrix = b_allele_martrix.fillna(0)

    baf_matrix = b_allele_martrix / (a_allele_martrix + b_allele_martrix)
    return a_allele_martrix, b_allele_martrix, baf_matrix

def extract_chr_and_start(bin_value):
    chrom, positions = bin_value.split(":")
    start, _ = positions.split("-")
    
    # extract chromosome
    if chrom.startswith("chr"):
        chrom = chrom[3:]
    chrom = int(chrom) if chrom.isdigit() else chrom
    
    # extract start position
    start = int(start)
    
    return chrom, start

def merge_coverage_files(coverage_files):
    merged_df = pd.DataFrame()
    # merge coverage files to a matrix, row index is chrom-pos, column index is cell
    for cov_file in coverage_files:
        cell_name = os.path.basename(cov_file).split(".")[0]
        
        df = pd.read_csv(cov_file, sep="\t", header=None, names=["chr", "start", "end", "coverage"])
        df["start"] =  df["start"] + 1
        df["bin"] = df["chr"] + ":" + df["start"].astype(str) + "-" + df["end"].astype(str)
        
        df = df[["bin", "coverage"]].set_index("bin")
        df.rename(columns={"coverage": cell_name}, inplace=True)
        
        if merged_df.empty:
            merged_df = df
        else:
            merged_df = merged_df.join(df, how="outer")
    
    merged_df = merged_df.sort_values(by="bin", key=lambda col: col.map(extract_chr_and_start))
    return merged_df

def create_allele_count_matrices(count_files, phased_snps_filtered, a_allele_bed, b_allele_bed):
    """
    Process SNP count files and create TSV matrices for A and B allele counts.
    
    Args:
        count_files: List of paths to count files
        phased_snps_filtered: Dictionary or set of filtered phased SNPs
        output_dir: Directory to save output files
    
    Returns:
        tuple: Paths to the created A and B allele count matrices
    """
    # Initialize dictionaries to store counts for each allele
    a_allele_counts = defaultdict(dict)
    b_allele_counts = defaultdict(dict)
    
    # Process each count file
    with open(a_allele_bed, 'w') as a_allele, open(b_allele_bed, 'w') as b_allele:
        for count_file in count_files:
            # Extract cell name from filename
            cell = os.path.basename(count_file).split(".")[0]
            
            # Process the count file to get SNP counts
            snps_counts = process_snp_count_file(count_file, phased_snps_filtered)
            
            # Store counts in the appropriate dictionaries
            for snp, counts in snps_counts.items():
                chrom, pos = snp
                snp_id = f"{chrom}:{pos}"
                
                # Store A allele count
                a_allele_counts[snp_id][cell] = counts[0]
                
                # Store B allele count
                b_allele_counts[snp_id][cell] = counts[1]
                a_allele.write('\t'.join([chrom, str(pos), str(pos+1), cell, str(counts[0])]) + '\n')
                b_allele.write('\t'.join([chrom, str(pos), str(pos+1), cell, str(counts[1])]) + '\n')
    
    # Convert dictionaries to pandas DataFrames
    a_allele_df = pd.DataFrame.from_dict(a_allele_counts, orient='index')
    b_allele_df = pd.DataFrame.from_dict(b_allele_counts, orient='index')
    
    # Fill NaN values with 0 (for cells with no counts at a particular SNP)
    a_allele_df = a_allele_df.fillna(0).astype(int)
    b_allele_df = b_allele_df.fillna(0).astype(int)
    
    # Sort the rows by chromosome and position
    def sort_key(snp_id):
        chrom, pos = snp_id.split(':')
        # Handle chromosome sorting (numeric, then X, Y, etc.)
        if chrom.startswith('chr'):
            chrom = chrom[3:]
        
        if chrom.isdigit():
            chrom_val = int(chrom)
        elif chrom == 'X':
            chrom_val = 100
        elif chrom == 'Y':
            chrom_val = 101
        else:
            chrom_val = 999
            
        return (chrom_val, int(pos))
    
    # Sort the DataFrames
    a_allele_df = a_allele_df.loc[sorted(a_allele_df.index, key=sort_key)]
    b_allele_df = b_allele_df.loc[sorted(b_allele_df.index, key=sort_key)]
    
    return a_allele_df, b_allele_df

def log_runtime(func):
    """A decorator to log and print the execution time of a member method."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Check if the instance has a 'log' method for robustness.
        if not hasattr(self, 'log'):
            print(f"Warning: {type(self).__name__} instance is missing a 'log' method. Cannot log runtime.")
            return func(self, *args, **kwargs)

        func_name = func.__name__
        self.log(f"Starting method: {func_name}...", level='PROGRESS')
        
        # 1. Record the start time
        start_time = time.perf_counter()
        
        # 2. Execute the original function
        # *args and **kwargs ensure that any arguments are passed correctly.
        result = func(self, *args, **kwargs)
        
        # 3. Record the end time
        end_time = time.perf_counter()
        
        # 4. Calculate the runtime
        runtime = end_time - start_time
        
        # 5. Log using self.log
        self.log(f"Method '{func_name}' finished, runtime: {runtime:.4f} seconds.", level='INFO')
        
        return result
    return wrapper

def generate_mixture_poisson():
    """Generate a random sample from a mixture of Poisson distributions with specified weights and lambda parameters.
    """
    weights = [0.3, 0.4, 0.2, 0.1]
    lambdas = [5, 20, 100, 300]
    component = np.random.choice(len(lambdas), p=weights)
    sample = max(1, np.random.poisson(lambdas[component]))
    return sample

def set_random_seed(seed):
    """Set random seed for reproducibility."""
    if seed is None:
        # If no seed provided, generate a random one and save it
        seed = int.from_bytes(os.urandom(4), byteorder="little")
    
    # Set seeds for both random and numpy
    random.seed(seed)
    np.random.seed(seed)
    
    return seed

def lorenz_curve(x, x0, y0):
    """Lorenz curve function"""
    if x <= x0:
        return (y0 / x0) * x
    else:
        return ((1 - y0) / (1 - x0)) * (x - x0) + y0


def fit_beta_to_lorenz(x0, y0, num_points=1000):
    """Fit Beta distribution parameters to Lorenz curve"""
    x_vals = np.linspace(0, 1, num_points)
    lorenz_vals = np.array([lorenz_curve(x, x0, y0) for x in x_vals])
    
    # Calculate derivatives to get PDF
    pdf_vals = np.diff(lorenz_vals) / np.diff(x_vals)
    pdf_vals = np.append(pdf_vals, pdf_vals[-1])
    
    # Normalize PDF
    pdf_vals = pdf_vals / np.sum(pdf_vals)
    
    # Estimate Beta parameters
    mean_val = np.sum(x_vals * pdf_vals)
    var_val = np.sum((x_vals - mean_val)**2 * pdf_vals)
    
    if var_val > 0:
        alpha = mean_val * ((mean_val * (1 - mean_val)) / var_val - 1)
        beta = (1 - mean_val) * ((mean_val * (1 - mean_val)) / var_val - 1)
        alpha = max(0.1, alpha)
        beta = max(0.1, beta)
    else:
        alpha, beta = 2.0, 2.0
    
    return alpha, beta

def gen_readcount(cov, l, window_size, num_windows, Alpha, Beta):
    """
    Generate read counts for each window using standard Metropolis-Hastings
    
    Note: Parameter 'u' is kept for API compatibility but not used in standard MH.
    Standard MH uses probabilistic acceptance based on the ratio.
    """
    
    # Calculate mean read count per window
    x0 = Alpha / (Alpha + Beta)
    mean_read = int(float(cov * window_size) / float(l))
    
    # Initialize
    readcounts = []
    
    # Starting point
    x_p = x0
    prob_x_p = beta_dist.pdf(x_p, Alpha, Beta)
    
    for i in range(num_windows):
        if i == 0:
            # First window uses initial value
            read_p = x_p / x0 * mean_read
            readcounts.append(int(read_p))
        else:
            # Proposal: sample from normal distribution centered at current value
            proposal_std = 0.1
            new_x = np.random.normal(x_p, proposal_std)
            
            # Ensure new_x is in valid range (0, 1)
            # Use reflection at boundaries
            while new_x <= 0 or new_x >= 1:
                if new_x <= 0:
                    new_x = -new_x
                if new_x >= 1:
                    new_x = 2 - new_x
            
            # Calculate probability of new value under Beta distribution
            new_x_p = beta_dist.pdf(new_x, Alpha, Beta)
            
            # Calculate acceptance probability
            prob_ratio = new_x_p / prob_x_p if prob_x_p > 0 else 1.0
            
            # Proposal is symmetric (normal distribution), so prop_ratio = 1
            acceptance_prob = min(1, prob_ratio)
            
            # Standard MH: accept with probability acceptance_prob
            if np.random.random() < acceptance_prob:
                x_p = new_x
                prob_x_p = new_x_p
            # else: keep current x_p (rejection)
            
            # Convert to read count
            read_p = x_p / x0 * mean_read
            readcounts.append(int(read_p))
    
    return readcounts

def write_fasta(chr_name, sequence, filename):
    """Write a single chromosome to FASTA file"""
    with open(filename, 'w') as f:
        f.write(f">{chr_name}\n")
        # Write sequence in 60bp lines
        for i in range(0, len(sequence), 60):
            f.write(sequence[i:i+60] + '\n')

def read_fasta(fasta_file):
    """Read all chromosomes from FASTA file"""
    chromosomes = {}
    current_chr = None
    current_seq = []
    
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('>'):
                # Save previous chromosome
                if current_chr is not None:
                    chromosomes[current_chr] = ''.join(current_seq)
                
                # Start new chromosome
                current_chr = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line.upper())
        
        # Save last chromosome
        if current_chr is not None:
            chromosomes[current_chr] = ''.join(current_seq)
    
    return chromosomes


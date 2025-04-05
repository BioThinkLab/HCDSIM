import os, sys
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
import argparse
import shlex
import shutil
import glob
import re
import traceback
import subprocess as sp
import multiprocessing as mp

from itertools import product
from collections import defaultdict
from heapq import nlargest

import numpy as np

from hcdsim.chisel_utils import *


def parse_args():
    # 1. 
    description = '''CHISEL command to create a barcoded BAM file from single-cell FASTQs (or gz-compressed FASTQs), single-cell BAMs, or a
        `RG:Z:`-barcoded BAM files without `CB:Z:` tags. When single-cell FASTQs or BAMs are provided
        a CELL name is assigned to each file (through either filename or table) and the same cell barcode will be assigned to all corresponding reads, but
        a different RG tag as they are considered as different repetitions of sequencing of the same cell. Specifically, when a table of inputs is not provied,
        for FASTQs each CELL name is extracted from the filename through the provided regular expression (default matches Illumina standard format), for BAMs
        basename is used as CELL name. When single-cell FASTQs are provided a READ value is also assigned to each file (through either filename or table) and 
        files with the same filename when removing READ values are considered as pairs of sequencing read mates.
        Input files, CELL names, and possible READ values can be provided through a table of inputs.'''
    
    # 2. 
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("INPUT", nargs='+', type=str, help='''Input FASTQs, BAMs, or TSV file with different behaviors: ......................................... 
    (1) FASTQs -- specified in a directory DIR as `DIR/*.fastq` or `DIR/*.fastq.gz` -- will be barcoded and aligned with (optionally) marked duplicates into a barcoded BAM file;
    .................................
    (2) BAMs -- specified in a directory DIR as `DIR/*.bam` -- will be barcoded and aligned with (optionally) marked duplicates into a barcoded BAM file; 
    ..............................................
    (3) a single BAM file with unique cells names in the field `RG:Z:` will be converted into a barcoded BAM file with the additional `CB:Z:` tag; 
    ..............
    (4) a tab-separated table of inputs (TSV with optional header starting with `#`) with two columns: the first column is an input file (FASTQ or BAM) and the
    second column is the corresponding cell name. When FASTQs are provided, a third column can be optionally specified to indicate the read name in paired-end
    sequencing, e.g., indicating either R1 or R2 for the first or second mate of paired-end reads, respectively. If a third column is not present, FASTQs are
    assumed to be from single-end sequencing.''')
    # 参考基因组，这是在 FASTQ 模式下必需的（默认：无）
    parser.add_argument("-r","--reference", type=str, required=False, help="Reference genome, which is mandatory in FASTQ mode (default: None)")
    # 运行目录（默认：当前目录
    parser.add_argument("-x","--rundir", required=False, default='./', type=str, help="Running directory (default: current directory)")
    # 运行目录中的输出名称（默认：barcodedcells.bam）
    parser.add_argument("-o","--output", required=False, default='barcodedcells.bam', type=str, help="Output name in running directory (default: barcodedcells.bam)")
    # --rexpname：用于提取输入 FASTQ 文件名中的细胞名称的正则表达式。
    parser.add_argument("--rexpname", required=False, default='(.*)_S.*_L00.*_R[1|2]_001.fastq.*', type=str, help="Regulare expression to extract cell name from input FASTQ filenames (default: `(.*)_S.*_L.*_R[1|2]_001.fastq.*`)")
    # --rexpread：用于提取输入 FASTQ 文件名中的READ名称的正则表达式。
    parser.add_argument("--rexpread", required=False, default='.*_S.*_L00.*_(R[1|2])_001.fastq.*', type=str, help="Regulare expression to extract cell name from input FASTQ filenames (default: `.*_S.*_L.*_(R[1|2])_001.fastq.*`)")
    # --barcodeonly：当设置为 True 时，仅计算条形码而不运行比对流程。
    parser.add_argument("--barcodeonly", required=False, default=False, action='store_true', help="Only compute barcodes but do not run aligning pipeline (default: False)")
    # --noduplicates：当设置为 True 时，不执行标记重复和使用 Picard 工具进行重校准。
    parser.add_argument("--noduplicates", required=False, default=False, action='store_true', help="Do not perform marking duplicates and recalibration with Picard tools (default: False)")
    # --keeptmpdir：当设置为 True 时，不删除临时目录。
    parser.add_argument("--keeptmpdir", required=False, default=False, action='store_true', help="Do not erase temporary directory (default: False)")
    # --barcodelength：设置条形码的长度，默认为 12。
    parser.add_argument("--barcodelength", required=False, type=int, default=12, help="Length of barcodes (default: 12)")
    # --bcftools：指定 bcftools 可执行文件的路径。
    parser.add_argument("--bcftools", required=False, default=None, type=str, help="Path to the directory to \"bcftools\" executable (default: in $PATH)")
    #  --samtools：指定 samtools 可执行文件的路径。
    parser.add_argument("--samtools", required=False, default=None, type=str, help="Path to the directory to \"samtools\" executable (default: in $PATH)")
    # --bwa：指定 bwa 可执行文件的路径。
    parser.add_argument("--bwa", required=False, default=None, type=str, help="Path to the directory to \"bwa\" executable (default: in $PATH)")
    # 设置并行工作线程数，默认等于可用处理器的数量。
    parser.add_argument("-j","--jobs", required=False, type=int, default=0, help="Number of parallele jobs to use (default: equal to number of available processors)")
    parser.add_argument("--seed", required=False, type=int, default=None, help="Random seed for replication (default: None)")

    args = parser.parse_args() #1.参数解析
    
    if args.seed is not None: #2.设置随机种子
        np.random.seed(args.seed)

    # 3. 输入文件检查：遍历 args.INPUT 中的每个输入文件，检查它们是否存在且不是目录。如果文件不存在，则抛出 ValueError。
    # inputs = map(os.path.abspath, filter(lambda f : len(f) > 1, args.INPUT))
    # 展开通配符以获取所有匹配的文件
    inputs = []
    for pattern in args.INPUT:
        matched_files = glob.glob(pattern)
        inputs.extend(matched_files)
    # 将相对路径转换为绝对路径
    inputs = list(map(os.path.abspath, inputs))

    for inp in inputs:
        if not (os.path.exists(inp) and not os.path.isdir(inp)):
            raise ValueError(error(f"This input file does not exist: {inp}"))

    # 4. 运行目录检查：检查 args.rundir 指定的运行目录是否存在。如果不存在，则抛出 ValueError。
    if not os.path.isdir(args.rundir):
        raise ValueError(error(f"Running directory does not exists: {args.rundir}"))
    
    # 5. 参考基因组检查：如果提供了参考基因组文件（args.reference），检查该文件是否存在。
    # 如果不存在，则抛出 ValueError。如果参考基因组文件存在，还会检查 BWA 索引文件是否齐全，如果不齐全，则同样抛出 ValueError。
    if args.reference is not None:
        if not os.path.isfile(args.reference):
            raise ValueError(error(f"Reference genome file does not exist: {args.reference}"))
        refidx = ['{}.{}'.format(args.reference, ix) for ix in ['amb', 'ann', 'bwt', 'pac', 'sa']]
        if not all(os.path.isfile(f) for f in refidx):
            raise ValueError(error("Some of the BWA index files are missing, please make sure these are available and generated through the command ``bwa index FASTA-REFERENCE''. Expected files are: {}".format('\n'.join(refidx))))

    # 6. 工作线程数设置：如果用户没有指定 --jobs 参数（即工作线程数），则 args.jobs 默认设置为 CPU 的核心数。如果 args.jobs 小于 1，则抛出 ValueError。
    if not args.jobs:
        args.jobs = mp.cpu_count()
    if args.jobs < 1:
        raise ValueError(error("The number of jobs must be positive!"))

    # 7. 外部工具路径检查：检查 bcftools(VCF/BCF 文件)、samtools(bam文件) 和 bwa(对FASTA格式) 工具是否存在。
    # 如果这些工具的路径没有通过对应的命令行参数指定，或者在系统路径中找不到这些工具，且 --barcodeonly 参数没有设置为 True，则抛出 ValueError。
    bcftools = args.bcftools
    if not bcftools:
        bcftools = "bcftools"
    if which(bcftools) is None and not args.barcodeonly:
        raise ValueError(error("bcftools has not been found or is not executable!\n\nIf you are within a CHISEL conda environment ${ENV} you can install it with:\n\tconda install -c bioconda -n ${ENV} bcftools\n\nOtherwise, please provide with the flag `--bcftools` the full path to the directory containing bcftools exacutable."))

    samtools = args.samtools
    if not samtools:
        samtools = "samtools"
    if which(samtools) is None and not args.barcodeonly:
        raise ValueError(error("samtools has not been found or is not executable!\n\nIf you are within a CHISEL conda environment ${ENV} you can install it with:\n\tconda install -c bioconda -n ${ENV} samtools\n\nOtherwise, please provide with the flag `--samtools` the full path to the directory containing samtools exacutable."))

    bwa = args.bwa
    if not bwa:
        bwa = "bwa"
    if which(bwa) is None and not args.barcodeonly:
        raise ValueError(error("bwa has not been found or is not executable!\n\nIf you are within a CHISEL conda environment ${ENV} you can install it with:\n\tconda install -c bioconda -n ${ENV} bwa\n\nOtherwise, please provide with the flag `--bwa` the full path to the directory containing bwa exacutable."))

    # 8.输出文件名确定：根据用户是否指定了输出文件的扩展名来确定最终的输出文件名。
    output = os.path.basename(args.output if args.output[-4:] == '.bam' else f"{args.output}.bam")

    return {
        "inputs" : inputs,
        "rundir" : os.path.abspath(args.rundir),
        "reference" : os.path.abspath(args.reference) if args.reference is not None else None,
        "barcodeonly" : args.barcodeonly,
        "noduplicates" : args.noduplicates,
        "keeptmpdir" : args.keeptmpdir,
        "barlength" : args.barcodelength,
        "bcftools" : bcftools,
        "samtools" : samtools,
        "bwa" : bwa,
        "rexpname" : args.rexpname,
        "rexpread" : args.rexpread,
        "jobs" : args.jobs,
        "output" : os.path.join(os.path.abspath(args.rundir), output)
    }
    
"""
入口文件
"""
def main():
    # 表示开始解析和检查参数。
    log('Parsing and checking arguments', level='STEP')
    # 函数解析命令行参数
    args = parse_args()
    log('\n'.join(['Arguments:'] + ['\t{} : {}'.format(a, args[a]) for a in args if a != 'inputs']) + '\n', level='INFO')

    # 表示开始设置环境。
    log('Setting up', level='STEP')
    # 1. 根据 args['rundir'] 创建一个临时目录的路径。
    tmpdir = os.path.join(args['rundir'], '_TMP_CHISEL_PREP')
    # 检查这个临时目录是否已经存在。
    if os.path.exists(tmpdir):
        raise ValueError(error(f"Temporary directory {tmpdir} already exists, please move or rename it!"))
    os.mkdir(tmpdir)
    # 2. 创建一个用于存放错误信息的目录路径。
    errdir = os.path.join(args['rundir'], '_ERR_CHISEL_PREP')
    # 如果错误目录已存在，同样抛出异常。
    if os.path.exists(errdir):
        raise ValueError(error(f"Temporary error directory {errdir} already exists, please move or rename it!"))
    os.mkdir(errdir)

    # 3. 检查输入参数 args['inputs'] 是否只有**一个元素**
    # 并且这个元素不是 BAM 文件 / 不是 FASTQ 格式 / 不是 gzip 压缩的 FASTQ 格式 / 不是 gzip 压缩文件
    # 那这个就是单独说明的那份 TSV 文件
    if len(args['inputs']) == 1 and args['inputs'][0][-4:] != '.bam'\
                                and args['inputs'][0][-6:] != '.fastq'\
                                and args['inputs'][0][-9:] != '.fastq.gz'\
                                and args['inputs'][0][-3:] != '.gz':
        # 如果输入是一个表格文件，记录日志表示开始读取。
        log('Reading provided table of inputs', level='STEP')
        # 3.1.读取输入表格文件
        # read_table函数：当输入参数 args['inputs'] 包含一个 TSV 文件时，用于解析该文件并获取输入文件的详细信息。
        #   返回输入文件列表（绝对路径）和信息字典，信息字典中包含每个文件的细胞名称、重复次数（和读取名称-在FASTQ格式下输出三个）
        args['inputs'], info = read_table(args['inputs'][0])
    else:
        info = None

    # 4. 检查所有输入文件是否为 FASTQ 或 gzip 压缩的 FASTQ 格式。
    if all(f[-6:] == '.fastq' or f[-9:] == '.fastq.gz' for f in args['inputs']):
         # 4.1. 如果参考基因组未指定，并且没有设置仅计算条形码的标志，则抛出异常。
        if args['reference'] is None and not args['barcodeonly']:
            shutil.rmtree(tmpdir)  # 删除临时目录
            shutil.rmtree(errdir)  # 删除错误目录
            # 抛出异常，提示在 FASTQ 模式下需要指定参考基因组。
            raise ValueError(error('Reference genome is required when running in FASTQ mode!'))
        
        # 4.2. 如果 info 为 None，即输入不是表格文件。
        if info is None:
            # 获取文件配对信息。
            files, fastqinfo, ispaired = match_fastq(args['inputs'], args)
        else:
            # 根据表格文件创建 FASTQ 信息。
            files, fastqinfo, ispaired = make_fastqinfo(args['inputs'], info, args)
        
        # 4.3.根据是否是配对的 FASTQ 文件，记录不同的日志信息。
        if ispaired:
            # 以 paired-end FASTQ 模式运行。
            log('Running in paired-end FASTQ mode', level='STEP')
        else:
            # 以 single-end FASTQ 模式运行。
            log('Running in single-end FASTQ mode', level='STEP')

        # 4.4. 调用 run_q 函数执行与 FASTQ 文件相关的处理，并将结果赋值给 barcoded 和 cells。
        barcoded, cells = run_q(args, tmpdir, errdir, files, fastqinfo, args['barcodeonly'])
        header = '#CELL\tBARCODE\tREPETITION\tREADS\tFILES'  # 设置日志信息的表头。
        
        # 4.5. 对 cells 列表进行排序。
        cells = sorted(cells, key=lambda c: (c[2], fastqinfo[c[0][0]]['reps']))
        
        # 4.6. 创建 loginfo，通过 map() 转换为列表（map() 返回的是迭代器，在 Python 3 需要转换为列表）。
        loginfo = list(map(lambda c: (
            fastqinfo[c[0][0]]['cell'], 
            c[2], 
            fastqinfo[c[0][0]]['reps'], 
            ','.join([fastqinfo[f]['read'] for f in c[0]]), 
            ','.join(c[0])
        ), cells))

    # 5. 逻辑与处理 FASTQ 文件类似，但是调用不同的函数。
    # ** 一般用这个 处理bam文件的函数
    elif all(f[-4:] == '.bam' for f in args['inputs']):
        # 5.1. 如果 info 为 None，即输入不是表格文件。
        if info is None:
            info = make_baminfo(args['inputs'])
        barcoded, cells, info = run_B(args, tmpdir, errdir) if len(args['inputs']) == 1 else run_b(args, tmpdir, errdir, info, args['barcodeonly'])
        header = '#CELL\tBARCODE\tREPETITION\tFILE'
        cells = sorted(cells, key=(lambda c : (c[2], info[c[0]]['reps'])))
        
        # 在Python3中将map转换为列表
        loginfo = list(map(lambda c: (info[c[0]]['cell'], c[2], info[c[0]]['reps'], c[0]), cells))
        
    else:
        raise ValueError(error("Input files are of wrong format or mixed formats"))

    # 6. 如果 barcodeonly 参数没有设置为 True
    if not args['barcodeonly']:
        # 记录日志，表示开始索引和完成最终的条形码 BAM 文件。
        log('Indexing and finalizing final barcoded BAM file', level='STEP')
        # 6.1. 调用 indexing 函数进行 BAM 文件的索引和最终处理。
        indexing(args['samtools'], args['jobs'], tmpdir, errdir, barcoded, args['output'])

        # 记录日志，表示开始检索最终条形码 BAM 文件的统计信息。
        log('Retrieving stats about the resulting barcoded BAM file', level='STEP')
        # 6.2. 格式化命令行字符串，用于获取 BAM 文件的统计信息。
        cmd = f'{args["samtools"]} flagstat -@ {args["jobs"]} {args["output"]}'
        # 使用 subprocess.Popen 执行命令，并捕获输出和错误。
        process = sp.Popen(shlex.split(cmd), stdout=sp.PIPE, stderr=sp.PIPE)
        stdout, stderr = process.communicate()
        # 记录命令的输出作为日志信息。
        stdout = stdout.decode()  # 解码为字符串
        stderr = stderr.decode()  # 解码为字符串
        if stdout:
            log(f'{stdout}', level='INFO')  # 使用 f-string 替换 .format()
        if stderr:
            log(f'{stderr}', level='ERROR')  # 如果有错误，记录错误信息

    # 7. 构造**包含条形码信息**的日志文件的路径。
    floginfo = os.path.join(args['rundir'], '{}.info.tsv'.format(args['output'][:-4]))
    # 记录日志，表示开始写入条形码细胞的最终总结。
    log('Writing final summary of barcoded cells', level='STEP')
    # 记录条形码细胞的数量。
    log(f'Number of barcoded cells: {len(set(c[-1] for c in cells))}', level='INFO')

    with open(floginfo, 'w', encoding='utf-8') as o:  # 指定文件编码
        o.write(f"{header}\n")  
        for l in loginfo:
            o.write("".join([str(item) + '\t' for item in l]) + '\n')

    # 记录日志，表示最终总结已经写入文件。
    log(f'Final summary is written in {floginfo}', level='INFO')

    # 8. 如果 keeptmpdir 参数没有设置为 True，执行以下步骤：
    if not args['keeptmpdir']:
        # 记录日志，表示开始清理临时文件。
        log('Cleaning remaining temporary files', level='STEP')
        shutil.rmtree(tmpdir)
        shutil.rmtree(errdir)
        
    log('KTHXBYE', level='STEP')
    

def read_table(file):
    """
    读取一个表格文件，并解析出输入文件的路径、细胞名称（CELL name）、重复次数（repetition，可能指的是测序重复或文件副本）以及可能的读取名称（READ name）。
    :param file:表格文件（通常是 TSV 文件，即制表符分隔的文件）
    :return:文件的路径 / CELL name / repetition / READ name
    """
    # 1.
    with open(file, 'r') as i:
        # 读取文件中的每一行line，对每行去除首尾空白字符（包括换行符），然后按空格分割。
        read = [l.strip().split() for l in i if l[0] != '#' and len(l) > 1]

    # 2.1 .定义一个 lambda 函数 get_ext，用于根据文件扩展名获取文件类型
    get_ext = (lambda s : 'b' if s[-4:] == '.bam' else ('q' if s[-6:] == '.fastq' or s[-9:] == '.fastq.gz' else None))
    # 2.2.使用 get_ext 函数获取所有输入文件的扩展名类型，并存储在一个集合 exts 中，集合会自动去除重复项。
    exts = set(get_ext(r[0]) for r in read)

    # 3.检查集合 exts 中是否包含 None，即是否有未知格式的文件。
    if None in exts:
        raise ValueError(error('Unknown format, different than .bam, .fastq, or .fastq.gz has been provided!'))
    
    # 4.使用 filter 函数找出所有分割后长度小于 2 的行，即不满足至少有文件路径和细胞名称两列的行。
    # list显示转换
    wrong = list(filter(lambda r : len(r) < 2, read))  # Convert filter result to list
    if len(wrong) > 0:
        raise ValueError(error(f'When a TSV is provided, every row must have at least two fields, specifying FILE and CELL-NAME, but an error was found, e.g.:\n\n{wrong[0]}'))
    
    # 5. 为所有输入文件生成绝对路径
    # list显示转换
    inputs = list(map(lambda r : os.path.abspath(r[0]), read)) 

    # 6.遍历所有输入文件的绝对路径。
    for inp in inputs:
        if not os.path.isfile(inp):
            raise ValueError(error(f"This input file does not exist: {inp}"))
    
    # 7. 如果所有文件都是 BAM 格式。
    if {'b'} == exts:
        binfo = {os.path.abspath(r[0]) : r[1] for r in read}
        reps = defaultdict(lambda : [])
        # list显示转换
        list(map(lambda f : reps[binfo[f]].append((f, len(reps[binfo[f]]))), sorted(binfo)))  # Convert map result to list
        reps = {s : dict(reps[s]) for s in reps}
        return inputs, {f : {'cell' : binfo[f], 'reps' : reps[binfo[f]][f]} for f in binfo}
    

    # 8.所有的输入文件都是 FASTQ 格式。
    elif {'q'} == exts:
        qinfo = {os.path.abspath(r[0]) : {'cell' : r[1], 'read' : r[2] if len(r) > 2 else 'R1'} for r in read}
        reps = defaultdict(lambda : [])
        found = set()
        proc = (lambda Q, f, R : reps[Q['cell']].append((R, len(reps[Q['cell']]))) or found.add(R) if R not in found else None)
        # list显示转换
        list(map(lambda f : proc(qinfo[f], f, f.replace(qinfo[f]['read'], '')), sorted(qinfo)))  # Convert map result to list
        reps = {s : dict(reps[s]) for s in reps}
        return inputs, {f : {'cell' : qinfo[f]['cell'], 'reps' : reps[qinfo[f]['cell']][f.replace(qinfo[f]['read'], '')], 'read' : qinfo[f]['read']} for f in qinfo}
    
    else:
        raise ValueError(error('The provided table contains files of mixed format (i.e. both BAM and FASTQ)!'))


def match_fastq(inputs, args):
    # 对输入的 FASTQ 文件进行格式验证和信息提取，确保文件符合预期的格式，并生成包含细胞和读取类型信息的 qinfo 字典

    # 1. 根据给定的正则表达式，从文件名中提取细胞（cell）和读取类型（read）
    mname = lambda s: re.search(args['rexpname'], os.path.basename(s))
    mread = lambda s: re.search(args['rexpread'], os.path.basename(s))

    # 使用 list(map()) 代替 map()，确保返回列表
    match = list(map(lambda f: (f, mname(f), mread(f)), inputs))

    # 2. 使用 all() 检查所有文件是否都成功匹配了正则表达式
    if all(m[1] is not None and m[2] is not None for m in match):
        qinfo = {m[0]: {'cell': m[1].group(1), 'read': m[2].group(1)} for m in match}
        reps = defaultdict(list)
        found = set()

        # proc 函数将每个文件的读取信息与其细胞关联，并确保每个读取类型（R1、R2）只有一个对应的文件
        def proc(Q, f, R):
            if R not in found:
                reps[Q['cell']].append((R, len(reps[Q['cell']])))
                found.add(R)

        # 使用 list(map()) 代替 map()
        list(map(lambda f: proc(qinfo[f], f, f.replace(qinfo[f]['read'], '')), sorted(qinfo)))

        reps = {s: dict(reps[s]) for s in reps}

        # 更新 qinfo 字典
        qinfo = {
            f: {'cell': qinfo[f]['cell'], 'reps': reps[qinfo[f]['cell']][f.replace(qinfo[f]['read'], '')], 'read': qinfo[f]['read']}
            for f in qinfo
        }
        return make_fastqinfo(inputs, qinfo, args)
    
    # 3. 如果某些文件未能匹配正则表达式
    else:
         # 如果某些文件未能匹配正则表达式，日志警告并默认这些文件为独立的单端FASTQ
        log('The filenames of the provided FASTQ files do not match the format with given expressions and will be all considered as independent single-end FASTQs', level='WARN')

        rmext = lambda s: s[-9:] if s[-9:] == '.fastq.gz' else s[-6:]
        qinfo = {m[0]: {'cell': rmext(m[0]), 'reps': 0, 'read': 'R1'} for m in match}

        # list显示转换
        files = list(map(lambda f: (f,), inputs))
        return files, qinfo, False
    
        
def make_fastqinfo(inputs, fastqinfo, args):
    # 根据这些信息进一步对文件进行分类，检查文件是单端还是配对端，并返回适当的文件对

    # 1. 将每个文件的细胞（cell）和重复信息（reps）作为键，文件名作为值
    pairs = defaultdict(list)

    # 使用 list(map()) 代替 map()
    list(map(lambda f: pairs[(fastqinfo[f]['cell'], fastqinfo[f]['reps'])].append(f), fastqinfo))

    # 排序每一组文件
    list(map(lambda p: pairs[p].sort(key=lambda f: fastqinfo[f]['read']), pairs))
   
    # 2. 单端检查
    if all(len(pairs[p]) == 1 for p in pairs):
        # list显示转换
        files = list(map(lambda f: (f,), inputs))
        return files, fastqinfo, False
    
    # 3. 配对检查
    elif all(len(set(pairs[p])) == 2 and len(pairs[p]) == 2 for p in pairs):
        # list显示转换
        files = list(map(lambda p: (pairs[p][0], pairs[p][1]), pairs))
        assert set(f for p in files for f in p) == set(fastqinfo.keys())
        return files, fastqinfo, True
    
    # 4. 异常处理
    else:
        for p in pairs:
            if len(pairs[p]) < 2:
                raise ValueError(f"Found more files from the same cell with same filenames and identical reads!\n{','.join(pairs[p])}")
            elif len(pairs[p]) > 2:
                raise ValueError(f"Found more than TWO files with the same cell and filenames, which cannot indicate paired-end reads!\n{','.join(pairs[p])}")
            else:
                assert False


def make_baminfo(inputs):
    # 根据输入的文件路径列表（inputs），创建一个字典，其中包含每个 BAM 文件的相关信息。
    # 根据文件的路径和文件名，按细胞（cell）名称进行分组，并为每个文件附加一些计数信息。

    # 1.
    binfo = {os.path.abspath(f): os.path.basename(f) for f in inputs}
    # print("binfo:", binfo)  # 打印binfo字典
    
    lanes = defaultdict(lambda: [])
    
    # 2. 使用for循环代替map，确保lanes字典被正确更新
    # 将同一个细胞名称（来自文件名）归类到同一组，并为每个文件计算其在该细胞中的“重复”信息。
    for f in sorted(binfo):
        cell_name = binfo[f]
        # 这里的“重复”信息是通过当前细胞中已有文件的数量来计算的
        lanes[cell_name].append((f, len(lanes[cell_name])))
    # print("lanes before conversion:", lanes)
    
    # 2.转换lanes中的列表为字典
    lanes = {s: dict(lanes[s]) for s in lanes}
    # print("lanes after conversion:", lanes)
    
    # 创建返回的字典
    result = {}
    for f in binfo:
        cell_name = binfo[f]
        if cell_name in lanes:
            if f in lanes[cell_name]:
                rep_info = lanes[cell_name][f]
                result[f] = {'cell': cell_name, 'reps': rep_info}
            else:
                print(f"Warning: {f} not found in lanes[{cell_name}]")  # 警告信息
        else:
            print(f"Warning: {cell_name} not found in lanes")  # 警告信息

    # print("Result dictionary:", result)  # 打印最终结果字典
    return result


def run_q(args, tmpdir, errdir, files, fastqinfo, barcodeonly):
    # 对 FASTQ 文件进行条形码提取、比对、排序和合并等操作

    par = {}
    par['files'] = files
    # 将 map 对象转换为列表
    par['names'] = list(map(lambda f: fastqinfo[f[0]]['cell'], files))
    par['lanes'] = list(map(lambda f: fastqinfo[f[0]]['reps'], files))
    par['barcodes'] = mkbarcodes(par['files'], args['barlength'], fastqinfo)
    par['tmpdir'] = tmpdir
    par['errdir'] = errdir
    par['ref'] = args['reference']
    par['samtools'] = args['samtools']
    par['bwa'] = args['bwa']
    par['J'] = args['jobs']

    barcoded = None
    
    if not barcodeonly:
        if args['noduplicates']:
            log('Alignment, barcoding and sorting is running for every cell', level='STEP')
            bams = align(**par)
        else:
            # noduplicates 默认 False
            log('Alignment, barcoding, sorting, and marking duplicates is running for every cell', level='STEP')
            bams = align_marked(**par)

        log('Merging all cells', level='STEP')
        # 不论是否进行去重，都会调用 merging() 函数将所有细胞的数据合并。该函数接受对齐结果文件（bams）以及其它参数
        barcoded = merging(args['samtools'], bams, args['jobs'], tmpdir, errdir)
    
    # 返回的结果将包括 barcoded 和每个文件的 (文件名, 细胞名, 条形码) 三元组
    # zip() 返回的是一个迭代器，在 Python 3 中是懒加载的。
    return barcoded, [(f, name, barcode) for f, name, barcode in zip(par['files'], par['names'], par['barcodes'])]

def run_b(args, tmpdir, errdir, binfo, barcodeonly):
    # 主要用于处理多个 BAM 文件的情况，执行条形码处理、排序和去重等任务。
    log('Running in multiple BAM files mode', level='STEP')
    
    # 1.从 args（包含用户提供的输入文件、工具路径等参数）和 binfo（包含关于 BAM 文件的元数据，如细胞名、重复次数等）中提取信息
    par = {}
    par['files'] = args['inputs']
    par['names'] = list(map(lambda f: binfo[f]['cell'], par['files']))  # 显式转换为列表
    par['lanes'] = list(map(lambda f: binfo[f]['reps'], par['files']))  # 显式转换为列表
    par['barcodes'] = mkbarcodes(par['files'], args['barlength'], binfo)
    par['tmpdir'] = tmpdir
    par['errdir'] = errdir
    par['samtools'] = args['samtools']
    par['J'] = args['jobs']
    barcoded = None

    # 2. barcodeonly 为 False(默认，即不只是生成条形码，还需要进行数据处理：
    if not barcodeonly:
        if args['noduplicates']:
            log('Barcoding and sorting is running for every cell', level='STEP')
            bams = barcode(**par)
        else:
            # 默认 False
            # 调用 barcode_marked 函数，除了条形码和排序，还会执行去重操作
            log('Barcoding, sorting, and marking duplicates is running for every cell', level='STEP')
            bams = barcode_marked(**par)

        log('Merging all cells', level='STEP')
        # 使用 merging 函数将处理后的数据合并。
        barcoded = merging(args['samtools'], bams, args['jobs'], tmpdir, errdir)
    
    # 3. 返回合并后的结果（barcoded）、BAM 文件、细胞名称和条形码的元数据。
    return barcoded, list(zip(par['files'], par['names'], par['barcodes'])), binfo


def run_B(args, tmpdir, errdir):
    # 主要用于处理单个 BAM 文件的情况
    log('Running in single BAM file mode', level='STEP')
    
    # 1. 使用 samtools split 按照 RG（Read Group）标签将输入的 BAM 文件拆分成多个文件，生成临时的 BAM 文件（存放在 tmpdir 中）。
    log('Splitting reads in BAM file by RG tag', level='STEP')
    sform = os.path.join(tmpdir, '%!.bam')
    cmd = f"{args['samtools']} split -f '{sform}' -@ {args['jobs']} {args['inputs'][0]}"
    proc = sp.Popen(shlex.split(cmd), stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise ValueError(error(f"Merging failed with messages:\n{stdout.decode()}\n{stderr.decode()}\n"))
    
    # 2. 更新 args['inputs']，指向拆分后的 BAM 文件。
    # 使用 list() 来确保 args['inputs'] 是一个列表，这对于后续代码（如 glob.glob()）是必要的
    args['inputs'] = list(map(os.path.abspath, glob.glob(os.path.join(tmpdir, '*.bam'))))
    
    getname = lambda f: os.path.splitext(os.path.basename(f))[0]
    
    # 3. 从拆分后的文件中提取文件名，假设文件名中包含细胞名称，并将其映射到 binfo 字典。
    binfo = {f : {'cell' : getname(f), 'reps' : 0} for f in args['inputs']}
    
    # 4. 调用 run_b 函数进行条形码处理和进一步的数据分析。
    # 确保传递 barcodeonly 参数
    barcodeonly = args.get('barcodeonly', None)  # barcodeonly 在 args 中
    return run_b(args, tmpdir, errdir, binfo, barcodeonly)
    
    
def mkbarcodes(files, length, info):
    # 为多个文件生成唯一的条形码，每个条形码对应一个文件。
    
    # 1. 通过 random_sample_iter 随机生成长度为 length 的序列，字符是 A、T、C、G，并生成足够数量的条形码（与文件数相同）
    random_sample_iter = (lambda it, k : (x for _, x in nlargest(k, ((np.random.random(), x) for x in it))))
    # 使用 list() 来将 map 对象转为列表
    barcodes = list(random_sample_iter(product(['A', 'T', 'C', 'G'], repeat=length), len(files)))
    barcodes = list(map(lambda b : ''.join(b), barcodes))  # 将 barcodes 转为列表
    assert len(barcodes) == len(files), '{} != {}'.format(len(files), len(barcodes))
    
    getfile = (lambda f : f[0] if type(f) == tuple else f)
    reps = defaultdict(lambda : [])
    # 使用 list() 来确保 map 被处理为一个列表
    list(map(lambda f : reps[info[getfile(f)]['cell']].append(f), sorted(files)))
    
    dup = [s for s in reps if len(reps[s]) != len(set(info[getfile(f)]['reps'] for f in reps[s]))]
    if len(dup) > 0:
        raise ValueError(error(f"Two or more of these files have the same cell name but also the same rep number:\n" + '\n'.join(reps[dup[0]])))


    # 使用 dict() 进行文件到条形码的映射
    assign = dict(zip(files, barcodes))
    return list(map(lambda f : assign[reps[info[getfile(f)]['cell']][0]], sorted(files)))
   
    
def align(files, names, barcodes, lanes, tmpdir, errdir, ref, bwa, samtools, J):
    jobs = zip(files, names, barcodes, lanes)
    bar = ProgressBar(total=len(files), length=30, verbose=False)
    initargs = (tmpdir, errdir, bwa, ref, samtools)
    pool = mp.Pool(processes=min(J, len(names)), initializer=init_align, initargs=initargs)
    progress = (lambda e, c : bar.progress(advance=True, msg=f"{e}") if c is None else error_code(e, c, errdir))
    
    # 使用 map() 获取处理结果，并确保只选择成功的对齐文件
    bams = [b for e, b, c in pool.imap_unordered(aligning, jobs) if progress(e, c)]
    
    # 关闭进程池并等待其完成
    pool.close()
    pool.join()
    return bams
    
    
def init_align(_tmpdir, _errdir, _bwa, _ref, _samtools):
    global cmd_bwa, cmd_arg, cmd_sor, tmpdir, errdir
    cmd_bwa = f"{_bwa} mem -M {_ref} {{}}"
    # 添加标记
    cmd_arg = '{} addreplacerg - -r \'ID:{}\' -r \'SM:{}\' -r \'FO:{}\' -r \'PG:CHISEL_PREP\' -Osam'.format(_samtools, '{}', '{}', '{}')
    # 排序命令
    cmd_sor = f"{_samtools} sort - -Obam -o {{}} -T {{}}"
    tmpdir = _tmpdir
    errdir = _errdir
    
    
def aligning(job):
    fil, name, barcode, lane = job
    bam = os.path.join(tmpdir, f"{name}_{lane}.bam")
    curr_tmp = os.path.join(tmpdir, f"_SORT_{name}_{lane}")
    os.makedirs(curr_tmp, exist_ok=True)  # 使用 `os.makedirs` 自动创建目录并避免重复创建
    
    # 构建命令
    curr_cmd_bwa = cmd_bwa.format(' '.join(fil))
    curr_cmd_arg = cmd_arg.format(f"CB:Z:{barcode}-{lane}", name, lane)
    curr_cmd_sor = cmd_sor.format(bam, curr_tmp)
    blog = os.path.join(errdir, f"{name}_{lane}_BWA.rdrlog")
    rlog = os.path.join(errdir, f"{name}_{lane}_SAMTOOLS_ADDREPLACERG.rdrlog")
    olog = os.path.join(errdir, f"{name}_{lane}_SAMTOOLS_SORT.rdrlog")

    # 执行命令并记录日志
    with open(blog, 'w') as ebwa, open(rlog, 'w') as earg, open(olog, 'w') as osor:
        pbwa = sp.Popen(shlex.split(curr_cmd_bwa), stdout=sp.PIPE, stderr=ebwa)
        parg = sp.Popen(shlex.split(curr_cmd_arg), stdin=pbwa.stdout, stdout=sp.PIPE, stderr=earg)
        psor = sp.Popen(shlex.split(curr_cmd_sor), stdin=parg.stdout, stdout=sp.PIPE, stderr=osor)
        
        # 等待进程结束并返回退出码
        rcodes = [p.wait() for p in [pbwa, parg, psor]]
    return f"{name}_{lane}", bam, check_rcodes(rcodes, [blog, rlog, olog], [curr_tmp])


def align_marked(files, names, barcodes, lanes, tmpdir, errdir, ref, bwa, samtools, J):
    jobs = zip(files, names, barcodes, lanes)
    bar = ProgressBar(total=len(files), length=30, verbose=False)
    initargs = (tmpdir, errdir, bwa, ref, samtools)
    pool = mp.Pool(processes=min(J, len(names)), initializer=init_align_marked, initargs=initargs)
    progress = (lambda e, c : bar.progress(advance=True, msg=f"{e}") if c is None else error_code(e, c, errdir))
    
    bams = [b for e, b, c in pool.imap_unordered(aligning_marked, jobs) if progress(e, c)]
    pool.close()
    pool.join()
    return bams
    

def init_align_marked(_tmpdir, _errdir, _bwa, _ref, _samtools):
    global cmd_bwa, cmd_nam, cmd_fix, cmd_arg, cmd_sor, cmd_mar, tmpdir, errdir
    cmd_bwa = f"{_bwa} mem -M {_ref} {{}}"
    cmd_nam = f"{_samtools} sort - -n -T {{}} -Osam"
    cmd_fix = f"{_samtools} fixmate -m - - -Osam"
    cmd_arg = '{} addreplacerg - -r \'ID:{}\' -r \'SM:{}\' -r \'FO:{}\' -r \'PG:CHISEL_PREP\' -Osam'.format(_samtools, '{}', '{}', '{}')
    cmd_sor = f"{_samtools} sort - -T {{}} -Osam"
    cmd_mar = f"{_samtools} markdup -T {{}} -Obam - {{}}"
    tmpdir = _tmpdir
    errdir = _errdir
    

def aligning_marked(job):
    fil, name, barcode, lane = job
    bam = os.path.join(tmpdir, f"{name}_{lane}.bam")
    nam_tmp = os.path.join(tmpdir, f"_NAME_{name}_{lane}")
    sor_tmp = os.path.join(tmpdir, f"_SORT_{name}_{lane}")
    mar_tmp = os.path.join(tmpdir, f"_MARK_{name}_{lane}")

    # 确保目录存在，避免创建已存在的目录时报错
    os.makedirs(nam_tmp, exist_ok=True)
    os.makedirs(sor_tmp, exist_ok=True)
    os.makedirs(mar_tmp, exist_ok=True)

    # 生成命令字符串
    curr_cmd_bwa = cmd_bwa.format(' '.join(fil))
    curr_cmd_nam = cmd_nam.format(nam_tmp)
    curr_cmd_fix = cmd_fix
    curr_cmd_arg = cmd_arg.format(f"CB:Z:{barcode}-{lane}", '{}'.format(name), '{}'.format(lane))
    curr_cmd_sor = cmd_sor.format(sor_tmp)
    curr_cmd_mar = cmd_mar.format(mar_tmp, bam)

    # 生成日志文件路径
    blog = os.path.join(errdir, f"{name}_{lane}_BWA.rdrlog")
    nlog = os.path.join(errdir, f"{name}_{lane}_SAMTOOLS_SNAME.rdrlog")
    flog = os.path.join(errdir, f"{name}_{lane}_SAMTOOLS_FIX.rdrlog")
    rlog = os.path.join(errdir, f"{name}_{lane}_SAMTOOLS_ADDREPLACERG.rdrlog")
    olog = os.path.join(errdir, f"{name}_{lane}_SAMTOOLS_SORT.rdrlog")
    mlog = os.path.join(errdir, f"{name}_{lane}_SAMTOOLS_MARK.rdrlog")

    # 打开日志文件并启动进程
    with open(blog, 'w') as ebwa, open(nlog, 'w') as enam, open(flog, 'w') as efix, \
         open(rlog, 'w') as earg, open(olog, 'w') as osor, open(mlog, 'w') as emar:
            
            pbwa = sp.Popen(shlex.split(curr_cmd_bwa), stdout=sp.PIPE, stderr=ebwa)
            pnam = sp.Popen(shlex.split(curr_cmd_nam), stdin=pbwa.stdout, stdout=sp.PIPE, stderr=enam)
            pfix = sp.Popen(shlex.split(curr_cmd_fix), stdin=pnam.stdout, stdout=sp.PIPE, stderr=efix)
            parg = sp.Popen(shlex.split(curr_cmd_arg), stdin=pfix.stdout, stdout=sp.PIPE, stderr=earg)
            psor = sp.Popen(shlex.split(curr_cmd_sor), stdin=parg.stdout, stdout=sp.PIPE, stderr=osor)
            pmar = sp.Popen(shlex.split(curr_cmd_mar), stdin=psor.stdout, stdout=sp.PIPE, stderr=emar)

            # 使用列表推导代替 map
            rcodes = [p.wait() for p in [pbwa, pnam, pfix, parg, psor, pmar]]
    return f"{name}_{lane}", bam, check_rcodes(rcodes, [blog, nlog, flog, rlog, olog, mlog], [nam_tmp, sor_tmp, mar_tmp])


def barcode(files, names, barcodes, lanes, tmpdir, errdir, samtools, J):
    # 显示转换
    jobs = list(zip(files, names, barcodes, lanes))

    bar = ProgressBar(total=len(files), length=30, verbose=False)
    initargs = (tmpdir, errdir, samtools)
    pool = mp.Pool(processes=min(J, len(names)), initializer=init_barcoding, initargs=initargs)
    progress = (lambda e, c : bar.progress(advance=True, msg=f"{e}") if c is None else error_code(e, c, errdir))
    bams = [b for e, b, c in pool.imap_unordered(barcoding, jobs) if progress(e, c)]
    
    pool.close()
    pool.join()
    return bams
    
    
def init_barcoding(_tmpdir, _errdir, _samtools):
    global cmd_arg, tmpdir, errdir
    cmd_arg = f"{_samtools} addreplacerg {{}} -r 'ID:{{}}' -r 'SM:{{}}' -r 'FO:{{}}' -r 'PG:CHISEL_PREP' -o {{}}"
    tmpdir = _tmpdir
    errdir = _errdir
    
    
def barcoding(job):
    fil, name, barcode, lane = job
    bam = os.path.join(tmpdir, f"{name}_{lane}.bam")
    curr_cmd_arg = cmd_arg.format(fil, f'CB:Z:{barcode}-{lane}', name, lane, bam)
    rlog = os.path.join(errdir, f"{name}_{lane}_SAMTOOLS_ADDREPLACERG.rdrlog")
    with open(rlog, 'w') as earg:
        parg = sp.Popen(shlex.split(curr_cmd_arg), stdout=sp.PIPE, stderr=earg)
        # 显示转换
        rcodes = list(map(lambda p: p.wait(), [parg]))
    return f"{name}_{lane}", bam, check_rcodes(rcodes, [rlog], [])


def barcode_marked(files, names, barcodes, lanes, tmpdir, errdir, samtools, J):
    # 具体是使用 samtools 对 BAM 文件进行处理和标记条形码

    # 将 names 转换为列表
    names = list(names)  # 显式地将 names 转换为列表
    jobs = zip(files, names, barcodes, lanes)
    
    bar = ProgressBar(total=len(files), length=30, verbose=False)
    initargs = (tmpdir, errdir, samtools)
    
    # 创建一个多进程池
    pool = mp.Pool(processes=min(J, len(names)), initializer=init_barcoding_marked, initargs=initargs)
    
    # 定义进度回调函数
    progress = (lambda e, c: bar.progress(advance=True, msg=f"{e}") if c is None else error_code(e, c, errdir))

    
    # 执行并发任务，并收集返回的 BAM 文件
    bams = [b for e, b, c in pool.imap_unordered(barcoding_marked, jobs) if progress(e, c)]
    
    # 关闭池，等待所有进程完成
    pool.close()
    pool.join()
    return bams
    
    
def init_barcoding_marked(_tmpdir, _errdir, _samtools):
    global cmd_nam, cmd_fix, cmd_arg, cmd_sor, cmd_mar, tmpdir, errdir
    # 用 samtools 创建的多个命令模板，用于处理 BAM 文件。
    # 1. 对文件进行排序。
    cmd_nam = '{} sort {} -n -T {} -Osam'.format(_samtools, '{}', '{}')
    # 2. 对文件进行修复。
    cmd_fix = '{} fixmate -m - - -Osam'.format(_samtools)
    # 3. 添加或替换 RG 标签（例如，ID、SM、FO 和 PG）。
    cmd_arg = '{} addreplacerg - -r \'ID:{}\' -r \'SM:{}\' -r \'FO:{}\' -r \'PG:CHISEL_PREP\' -Osam'.format(_samtools, '{}', '{}', '{}')
    # 4. 对文件进行排序。
    cmd_sor = '{} sort - -T {} -Osam'.format(_samtools, '{}')
    # 5. 对文件进行标记重复。
    cmd_mar = '{} markdup -T {} - {} -Obam'.format(_samtools, '{}', '{}')
    # 传递给 init_barcoding_marked 的参数，指定临时文件夹和错误日志目录。
    tmpdir = _tmpdir
    errdir = _errdir
    
    
def barcoding_marked(job):
    # 接收一个任务元组 job
    fil, name, barcode, lane = job
    # 1. 生成用于存储最终结果的 BAM 文件路径。
    bam = os.path.join(tmpdir, f'{name}_{lane}.bam')
    # 2. 为每个处理步骤（命名、排序、标记重复）创建独立的临时目录
    nam_tmp = os.path.join(tmpdir, f"_NAME_{name}_{lane}")
    os.makedirs(nam_tmp, exist_ok=True)
    sor_tmp = os.path.join(tmpdir, f"_SORT_{name}_{lane}")
    os.makedirs(sor_tmp, exist_ok=True)
    mar_tmp = os.path.join(tmpdir, f"_MARK_{name}_{lane}")
    os.makedirs(mar_tmp, exist_ok=True)

    # 3. 使用预定义的命令模板（来自 init_barcoding_marked）格式化具体的命令
    curr_cmd_nam = cmd_nam.format(fil, nam_tmp)
    curr_cmd_fix = cmd_fix
    curr_cmd_arg = cmd_arg.format(f'CB:Z:{barcode}-{lane}', name, lane)
    curr_cmd_sor = cmd_sor.format(sor_tmp)
    curr_cmd_mar = cmd_mar.format(mar_tmp, bam)

    # 4. 为每个步骤生成日志文件，记录 samtools 操作的标准输出和错误输出。
    nlog = os.path.join(errdir, f"{name}_{lane}_SAMTOOLS_SNAME.rdrlog")
    flog = os.path.join(errdir, f"{name}_{lane}_SAMTOOLS_FIX.rdrlog")
    rlog = os.path.join(errdir, f"{name}_{lane}_SAMTOOLS_ADDREPLACERG.rdrlog")
    olog = os.path.join(errdir, f"{name}_{lane}_SAMTOOLS_SORT.rdrlog")
    mlog = os.path.join(errdir, f"{name}_{lane}_SAMTOOLS_MARK.rdrlog")

    # 5. 使用 subprocess.Popen 启动多个子进程，每个子进程执行一个命令。
    with open(nlog, 'w') as enam, open(flog, 'w') as efix, open(rlog, 'w') as earg, open(olog, 'w') as osor, open(mlog, 'w') as emar:
        pnam = sp.Popen(shlex.split(curr_cmd_nam), stdout=sp.PIPE, stderr=enam)
        pfix = sp.Popen(shlex.split(curr_cmd_fix), stdin=pnam.stdout, stdout=sp.PIPE, stderr=efix)
        parg = sp.Popen(shlex.split(curr_cmd_arg), stdin=pfix.stdout, stdout=sp.PIPE, stderr=earg)
        psor = sp.Popen(shlex.split(curr_cmd_sor), stdin=parg.stdout, stdout=sp.PIPE, stderr=osor)    
        pmar = sp.Popen(shlex.split(curr_cmd_mar), stdin=psor.stdout, stdout=sp.PIPE, stderr=emar)
        
        # 用列表推导代替 map 和 lambda
        rcodes = [p.wait() for p in [pnam, pfix, parg, psor, pmar]]
    # 返回样本的名称和通道号（如 name_lane）以及生成的 BAM 文件路径。如果有错误，还会记录错误日志。
    return f'{name}_{lane}', bam, check_rcodes(rcodes, [nlog, flog, rlog, olog, mlog], [nam_tmp, sor_tmp, mar_tmp])


def merging(samtools, bams, jobs, tmpdir, errdir):
    # 将多个处理后的 BAM 文件合并成一个单一的 BAM 文件。

    # 1. 定义输出文件路径
    barcoded = os.path.join(tmpdir, 'barcodedcells.bam')
    cellbams = os.path.join(tmpdir, 'cellbams.tsv')
    # 2. 写入 BAM 文件列表
    with open(cellbams, 'w') as o:
        o.write('\n'.join(bams))
    # 3. 构建合并命令 samtools
    cmd = '{} merge {} -b {} -@ {}'.format(samtools, barcoded, cellbams, jobs)
    # 4. 使用 subprocess 模块执行构建的合并命令
    proc = sp.Popen(shlex.split(cmd), stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, stderr = proc.communicate()
    # 5. 检查命令是否成功执行（返回码为 0 表示成功）
    if proc.returncode != 0:
        # 标准输出和标准错误流通常会以字节流的形式传输
        raise ValueError(error(f'Merging failed with messages:\n{stdout.decode()}\n{stderr.decode()}\n'))
    # 6.函数返回合并后的 BAM 文件的路径。
    return barcoded


def indexing(samtools, jobs, tmpdir, errdir, barcoded, output):
    # 使用 samtools 工具对一个文件进行 索引操作。

    # 移动文件
    shutil.move(barcoded, output)
    
    # 构建命令字符串，使用 f-string 格式化
    cmd = f'{samtools} index {output} -@ {jobs}'
    
    # 启动进程并捕获标准输出和标准错误
    proc = sp.Popen(shlex.split(cmd), stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, stderr = proc.communicate()
    
    # 检查命令执行是否成功
    if proc.returncode != 0:
        # 解码 stdout 和 stderr 为字符串，并抛出异常
        raise ValueError(error(f'Indexing failed with messages:\n{stdout.decode()}\n{stderr.decode()}\n'))
    return


def error_code(e, c, errdir):
    error_message = f"Commands failed for {e} with error:\n{c}\nCheck the errors in the corresponding rdrlog files in:\n{errdir}"
    raise ValueError(error(error_message))


def check_rcodes(rcodes, logs, tdir, cmd=None):
    # 检查一组进程的返回码（rcodes），并根据返回码判断是否需要处理日志和清理文件。
    # 如果返回码非零，表示进程出现了错误，并根据日志输出详细信息。
    res = ''
    # 1. 检查每个进程的返回码
    for code, log in zip(rcodes, logs):
        if code != 0:
            # 如果提供了 cmd，直接返回 cmd
            if cmd is not None:
                return cmd
            # 否则，读取错误日志，并将其内容累加到结果中
            with open(log, 'r') as i:
                res += 'From {}:\n'.format(log) + '\n'.join(i.readlines()) + '\n'
    # 如果没有错误（res 为空），删除所有日志文件和临时目录。
    if res == '':
        # 使用 list() 来确保 map 执行
        list(map(lambda f: os.remove(f), logs))
        list(map(lambda d: shutil.rmtree(d), tdir))
        return None
    else:
        return res



if __name__ == '__main__':
    main()

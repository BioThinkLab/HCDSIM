#!/usr/bin/env python3
import subprocess
import os
import sys
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time

def downsample_single_bin(args):
    """
    处理单个bin的downsample
    
    Parameters:
    -----------
    args : tuple
        包含所有需要的参数
    
    Returns:
    --------
    tuple : (success, temp_bam_path, bin_info)
    """
    (idx, clone_bam, clone_cnv, cell_cnv, clone_coverage, cell_coverage, 
     bin_range, temp_dir, seed, lock) = args
    
    # 解析bin range
    if isinstance(bin_range, str):
        region_str = bin_range
        chrom, pos = bin_range.split(':')
        start, end = pos.split('-')
    elif isinstance(bin_range, (tuple, list)):
        chrom, start, end = bin_range
        region_str = f"{chrom}:{start}-{end}"
    else:
        return (False, None, f"不支持的bin_range格式: {bin_range}")
    
    # 计算downsample比例
    if clone_cnv == 0 or clone_coverage == 0:
        with lock:
            print(f"跳过 {region_str}: clone_cnv={clone_cnv}, clone_coverage={clone_coverage}", 
                  file=sys.stderr)
        return (False, None, f"Skipped {region_str}: zero CNV or coverage")
    
    ratio = (cell_coverage / clone_coverage) * (cell_cnv / clone_cnv)
    
    # samtools -s 参数范围是0-1
    if ratio > 1:
        with lock:
            print(f"警告: {region_str} 的ratio={ratio:.4f} > 1, 设置为1.0", file=sys.stderr)
        ratio = 1.0
    elif ratio <= 0:
        with lock:
            print(f"警告: {region_str} 的ratio={ratio:.4f} <= 0, 跳过此bin", file=sys.stderr)
        return (False, None, f"Skipped {region_str}: ratio <= 0")
    
    # 临时输出文件
    temp_bam = os.path.join(temp_dir, f"bin_{idx:06d}.bam")
    
    # 使用samtools进行downsample
    cmd = [
        "samtools", "view",
        "-b",  # 输出BAM格式
        "-s", f"{seed}.{ratio:.6f}",  # downsample比例
        clone_bam,
        region_str,
        "-o", temp_bam
    ]
    
    with lock:
        print(f"处理 {region_str}: ratio={ratio:.6f}, clone_cnv={clone_cnv}, cell_cnv={cell_cnv}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return (True, temp_bam, f"Success {region_str}")
    except subprocess.CalledProcessError as e:
        with lock:
            print(f"错误处理 {region_str}: {e.stderr}", file=sys.stderr)
        # 清理失败的临时文件
        if os.path.exists(temp_bam):
            os.remove(temp_bam)
        return (False, None, f"Failed {region_str}: {e.stderr}")


def downsample_clone_to_cell(clone_bam, cell_bam_out, clone_coverage, cell_coverage,
                              clone_cnv_vector, cell_cnv_vector, bin_ranges, 
                              temp_dir="temp_bins", n_threads=4, seed=42):
    """
    根据CNV和coverage从clone BAM downsample生成cell BAM (多线程版本)
    
    Parameters:
    -----------
    clone_bam : str
        Clone的BAM文件路径
    cell_bam_out : str
        输出的cell BAM文件路径
    clone_coverage : float
        Clone的平均coverage
    cell_coverage : float
        期望的cell coverage
    clone_cnv_vector : list or numpy.array
        Clone的CNV值向量，例如 [8, 8, 8, 0, 0, ...]
    cell_cnv_vector : list or numpy.array
        Cell的CNV值向量，例如 [1, 1, 1, 0, 0, ...]
    bin_ranges : list of tuples or list of str
        Bin的范围，格式可以是：
        - List of tuples: [('chr1', 1, 100000), ('chr1', 100001, 200000), ...]
        - List of strings: ['chr1:1-100000', 'chr1:100001-200000', ...]
    temp_dir : str, optional
        临时文件目录，默认为 "temp_bins"
    n_threads : int, optional
        线程数，默认为4
    seed : int, optional
        随机种子，默认为42
    
    Returns:
    --------
    str : 输出的BAM文件路径
    (cell_cov/clone_cov) * (cell_cnv/clone_cnv) 比例进行downsample
    Example:
    --------
    >>> clone_cnv = [8, 8, 8, 0, 0, 0]
    >>> cell_cnv = [1, 1, 1, 0, 0, 0]
    >>> bins = ['chr1:1-100000', 'chr1:100001-200000', 'chr1:200001-300000',
    ...         'chr1:300001-400000', 'chr1:400001-500000', 'chr1:500001-600000']
    >>> downsample_clone_to_cell('clone1.bam', 'cell_out.bam', 30, 1,
    ...                          clone_cnv, cell_cnv, bins, n_threads=8)
    """
    
    start_time = time.time()
    
    # 转换为numpy数组以便计算
    clone_cnv_vector = np.array(clone_cnv_vector)
    cell_cnv_vector = np.array(cell_cnv_vector)
    
    # 检查向量长度是否一致
    if len(clone_cnv_vector) != len(cell_cnv_vector) or len(clone_cnv_vector) != len(bin_ranges):
        raise ValueError(f"向量长度不一致: clone_cnv={len(clone_cnv_vector)}, "
                        f"cell_cnv={len(cell_cnv_vector)}, bins={len(bin_ranges)}")
    
    # 创建临时目录
    Path(temp_dir).mkdir(exist_ok=True)
    
    # 创建线程锁用于同步打印
    print_lock = Lock()
    
    # 准备所有任务的参数
    tasks = []
    for idx in range(len(bin_ranges)):
        task_args = (
            idx,
            clone_bam,
            clone_cnv_vector[idx],
            cell_cnv_vector[idx],
            clone_coverage,
            cell_coverage,
            bin_ranges[idx],
            temp_dir,
            seed,
            print_lock
        )
        tasks.append(task_args)
    
    # 使用线程池并行处理
    temp_bams = []
    successful_bins = 0
    failed_bins = 0
    
    print(f"\n使用 {n_threads} 个线程处理 {len(tasks)} 个bins...\n")
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        # 提交所有任务
        future_to_idx = {executor.submit(downsample_single_bin, task): i 
                        for i, task in enumerate(tasks)}
        
        # 收集结果
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                success, temp_bam, info = future.result()
                if success:
                    temp_bams.append(temp_bam)
                    successful_bins += 1
                else:
                    failed_bins += 1
            except Exception as exc:
                with print_lock:
                    print(f'Bin {idx} 产生异常: {exc}', file=sys.stderr)
                failed_bins += 1
    
    print(f"\n处理完成: 成功 {successful_bins} 个bins, 失败 {failed_bins} 个bins")
    
    # 检查是否有成功的bin
    if not temp_bams:
        raise ValueError("没有成功处理任何bin，无法生成输出BAM文件")
    
    # 按文件名排序以保证顺序
    temp_bams.sort()
    
    print(f"\n合并 {len(temp_bams)} 个BAM文件...")
    
    # 合并所有临时BAM文件
    merge_cmd = ["samtools", "merge", "-f", "-@", str(n_threads), cell_bam_out] + temp_bams
    try:
        subprocess.run(merge_cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"合并BAM文件时出错: {e.stderr}", file=sys.stderr)
        raise
    
    # 排序
    print("排序BAM文件...")
    sorted_bam = cell_bam_out.replace(".bam", ".sorted.bam")
    sort_cmd = ["samtools", "sort", "-@", str(n_threads), "-o", sorted_bam, cell_bam_out]
    try:
        subprocess.run(sort_cmd, check=True, capture_output=True, text=True)
        # 用排序后的文件替换原文件
        os.rename(sorted_bam, cell_bam_out)
    except subprocess.CalledProcessError as e:
        print(f"排序BAM文件时出错: {e.stderr}", file=sys.stderr)
        raise
    
    # 建立索引
    print("建立索引...")
    index_cmd = ["samtools", "index", "-@", str(n_threads), cell_bam_out]
    try:
        subprocess.run(index_cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"建立索引时出错: {e.stderr}", file=sys.stderr)
        # 索引失败不影响主流程
        pass
    
    # 清理临时文件
    print("\n清理临时文件...")
    for temp_bam in temp_bams:
        if os.path.exists(temp_bam):
            os.remove(temp_bam)
    
    # 可选：删除临时目录
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass  # 目录不为空时忽略
    
    elapsed_time = time.time() - start_time
    print(f"\n完成! 总耗时: {elapsed_time:.2f} 秒")
    print(f"输出文件: {cell_bam_out}")
    
    return cell_bam_out


# 示例使用
if __name__ == "__main__":
    # 示例数据
    clone_cnv = [8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cell_cnv = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    bin_ranges = [
        'chr1:1-100000',
        'chr1:100001-200000',
        'chr1:200001-300000',
        'chr1:300001-400000',
        'chr1:400001-500000',
        'chr1:500001-600000',
        'chr1:600001-700000',
        'chr1:700001-800000',
        'chr1:800001-900000',
        'chr1:900001-1000000',
        'chr1:1000001-1100000',
        'chr1:1100001-1200000'
    ]
    
    # 调用函数，使用8个线程
    downsample_clone_to_cell(
        clone_bam='clone1.bam',
        cell_bam_out='cell_downsampled.bam',
        clone_coverage=30,
        cell_coverage=1,
        clone_cnv_vector=clone_cnv,
        cell_cnv_vector=cell_cnv,
        bin_ranges=bin_ranges,
        temp_dir='temp_downsample',
        n_threads=8  # 使用8个线程
    )
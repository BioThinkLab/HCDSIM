import os
import datetime
import multiprocessing
import sys
import time
import shlex
import re
import subprocess as sp
from multiprocessing import Value, Lock, Pool


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


def log(msg, level='STEP', lock=None):
    """
    输出日志信息到标准错误输出。

    :param msg: 需要输出的日志消息。
    :param level: 日志级别，决定日志的输出样式和颜色。
    :param lock: 用于并发控制的日志锁，确保日志输出的顺序性。
    """
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
        sys.stderr.write(f"{log_msg}\n")
    else:
        with lock:
            sys.stderr.write(f"{log_msg}\n")


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

def error(msg):
    log(msg=msg, level="ERROR")
    sys.exit(0)

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

def runcmd(cmd, xdir, out=None, log="log"):
    """
    在指定目录下执行一个命令，并将输出和错误日志保存到文件中。
    :param cmd: 要执行的命令字符串。
    :param xdir: 执行命令的目录路径。
    :param out: 选参数，指定输出文件的名称。如果为 None，则不保存标准输出。
    :param log: 日志文件的名称，默认为 "log"。
    :return:
    """
    j = os.path.join
    tmp = log + '_TMP'


    # 1. 确定标准输出文件，如果 out 参数为 None，则使用 os.devnull
    sout_path = j(xdir, out) if out else os.devnull
    tmp_path = j(xdir, tmp)
    log_path = j(xdir, log)

    # 2. 使用 with open 自动管理文件句柄
    with open(sout_path, 'w') as sout, open(tmp_path, 'w') as serr:
        # 使用 subprocess.Popen 启动子进程执行命令

        proc = sp.Popen(shlex.split(cmd), stdout=sout, stderr=sp.PIPE)

        # 逐行读取标准错误输出并写入到 stderr 和临时文件
        for line in iter(proc.stderr.readline, b''):  # readline 返回字节数据
            sys.stderr.write(line.decode())  # 将字节解码为字符串并写入标准错误
            serr.write(line.decode())  # 将标准错误写入临时文件

        # 等待子进程完成
        proc.stderr.close()
        proc.wait()

    # 读取临时文件内容并清理 ANSI 转义序列
    with open(tmp_path, 'r') as i, open(log_path, 'w') as o:
        for line in i:
            if 'Progress' not in line:
                # 移除 ANSI 转义序列
                o.write(re.sub(r'\033\[[0-9]*m', '', line))

    # 删除临时文件
    os.remove(tmp_path)

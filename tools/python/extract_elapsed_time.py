import re
import sys


def extract_elapsed_time_sum(log_text):
    """
    从日志文本中提取所有包含"[RecursiveMatching] Enumerating()"的行，
    并统计elapsed时间的总和
    """
    pattern = r'.*\[RecursiveMatching\] Enumerating\(\) elapsed: (\d+\.\d+) sec'
    matches = re.findall(pattern, log_text)
    elapsed_times = [float(time) for time in matches]
    return elapsed_times, sum(elapsed_times)


def main():
    if len(sys.argv) != 2:
        print("用法: python extract_elapsed_time.py <logfile>")
        sys.exit(1)

    logfile = sys.argv[1]

    try:
        with open(logfile, 'r', encoding='utf-8') as file:
            log_text = file.read()
    except FileNotFoundError:
        print(f"错误：文件 {logfile} 未找到")
        sys.exit(1)
    except Exception as e:
        print(f"读取文件时出错：{e}")
        sys.exit(1)

    elapsed_times, total_time = extract_elapsed_time_sum(log_text)

    print(f"找到 {len(elapsed_times)} 个匹配的时间记录：")
    for i, time in enumerate(elapsed_times, 1):
        print(f"  {i}. {time} sec")
    print(f"\nelapsed时间总和：{total_time:.6f} sec")


if __name__ == "__main__":
    main()

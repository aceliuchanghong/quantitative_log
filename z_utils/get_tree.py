import os
import fnmatch


def generate_tree(startpath, exclude_patterns=None):
    """
    生成一个标准的目录树结构，并支持从 .gitignore 和自定义列表中排除文件/目录。

    :param startpath: str, 要扫描的根目录路径。
    :param exclude_patterns: set, 需要排除的文件/目录模式集合，支持 fnmatch 通配符。
    """
    # 默认排除的常见模式
    if exclude_patterns is None:
        exclude_patterns = set()

    default_excludes = {
        ".git",
        ".vscode",
        "__pycache__",
        "node_modules",
        "*.pyc",
        "*.swp",
    }
    exclude_patterns.update(default_excludes)

    # 从根目录读取 .gitignore 文件并添加其规则
    gitignore_path = os.path.join(startpath, ".gitignore")
    if os.path.isfile(gitignore_path):
        with open(gitignore_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    # fnmatch 不需要 gitignore 模式末尾的 '/'
                    if line.endswith("/"):
                        line = line[:-1]
                    exclude_patterns.add(line)

    # 打印根目录
    print(f"{os.path.basename(os.path.abspath(startpath))}/")

    # 开始递归生成树
    _recursive_tree_gen(startpath, "", exclude_patterns)


def _recursive_tree_gen(directory, prefix, exclude_patterns):
    """
    内部递归函数，用于生成和打印目录树。
    """
    try:
        all_items = os.listdir(directory)
    except OSError as e:
        print(f"{prefix}└── [Error: {e}]")
        return

    # 过滤掉需要排除的项目
    filtered_items = []
    for item in all_items:
        is_excluded = any(
            fnmatch.fnmatch(item, pattern) for pattern in exclude_patterns
        )
        if not is_excluded:
            filtered_items.append(item)

    # 排序：文件在前，目录在后，然后按字母顺序
    filtered_items.sort(
        key=lambda x: (os.path.isdir(os.path.join(directory, x)), x.lower())
    )

    # 打印条目
    pointers = ["├── "] * (len(filtered_items) - 1) + ["└── "]
    for pointer, item in zip(pointers, filtered_items):
        path = os.path.join(directory, item)
        print(f"{prefix}{pointer}{item}")

        if os.path.isdir(path):
            # 递归进入子目录，并更新前缀
            extension = "│   " if pointer == "├── " else "    "
            _recursive_tree_gen(path, prefix + extension, exclude_patterns)


if __name__ == "__main__":
    """
    uv run z_utils/get_tree.py
    """

    path = "../quantitative_log"
    exclude_dirs_set = {
        ".env",
        ".gitignore",
        ".python-version",
        "pyproject.toml",
        "LICENSE",
        "README.md",
        "uv.lock",
        "prompt.md",
        "package-lock.json",
        "node_modules",
        "z_using_files",
        "z_utils",
    }
    generate_tree(path, exclude_patterns=exclude_dirs_set)

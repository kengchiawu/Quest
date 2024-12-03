from git import Repo
import os

def get_repo_info(repo_path='~/quest'):
    try:
        # 打开 Git 仓库
        repo = Repo(repo_path)
        
        # 检查是否为裸仓库（bare repository）
        if repo.bare:
            return "Repository is bare.", None
        
        # 获取仓库名称（即顶级目录名）
        repo_name = os.path.basename(repo.working_tree_dir)
        
        # 获取当前分支名称
        try:
            current_branch = repo.active_branch.name
        except TypeError:
            # 如果没有活动分支（例如在分离头指针状态下），返回 None 或其他提示信息
            current_branch = "HEAD is in detached state."
        
        return repo_name, current_branch
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

# 调用函数并打印结果
repo_name, current_branch = get_repo_info()
if repo_name and current_branch:
    print(f"Repository name is: {repo_name}")
    print(f"Current branch is: {current_branch}")
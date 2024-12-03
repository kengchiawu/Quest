from git import Repo
import os
import requests

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
def send_message():
    repo_name, current_branch = get_repo_info()
    headers = {"Authorization": "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjUyNTAwMiwidXVpZCI6IjE5YzhiMWY0LWM2ZDQtNGRmZC04NmExLWZjMTEzNzcyMjIxMCIsImlzX2FkbWluIjpmYWxzZSwiYmFja3N0YWdlX3JvbGUiOiIiLCJpc19zdXBlcl9hZG1pbiI6ZmFsc2UsInN1Yl9uYW1lIjoiIiwidGVuYW50IjoiYXV0b2RsIiwidXBrIjoiIn0.VFeaZd23xdnopLR0cOLdPsUpTKT4fZh3ltTzsjNOPa1Xy-94ekt1wv7ucqCzM_Ka8SZ62O-nuQPEcDmB71oyNw"}
    resp = requests.post("https://www.autodl.com/api/v1/wechat/message/send",
                         json={
                             "title": "仿真完成",
                             "name": f"{repo_name}.{current_branch}",
                         }, headers = headers)
    print(resp.content.decode())

if __name__ == "__main__":
    send_message()
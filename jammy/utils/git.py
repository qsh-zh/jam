import git
import subprocess
import os
import os.path as osp

from jammy.logging import get_logger

logger = get_logger()

__all__ = ["is_git", "git_rootdir", "git_hash"]


def is_git(path):
    try:
        _ = git.Repo(path, search_parent_directories=True).git_dir
        return True
    except git.exc.InvalidGitRepositoryError:
        return False


def git_rootdir(path=""):
    if is_git(os.getcwd()):
        git_repo = git.Repo(os.getcwd(), search_parent_directories=True)
        root = git_repo.git.rev_parse("--show-toplevel")
        return osp.join(root, path)
    logger.info("not a git repo")
    return osp.join(os.getcwd(), path)


def git_hash(path):
    if is_git(path):
        git_repo = git.Repo(path, search_parent_directories=True)
        return git_repo.head.object.hexsha
    logger.info("not a git repo")
    return None


def git_repo(path):
    if is_git(path):
        git_repo = git.Repo(path, search_parent_directories=True)
        return git_repo
    logger.info("not a git repo")
    return None


def log_repo(path):
    repo = git_repo(path)
    if repo:
        return repo.head.object.hexsha, repo.git.diff()
    # if not repo, return None sha, empty diff
    return None, ""

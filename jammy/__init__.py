from jammy.utils.init import init_main
import jammy.utils.git as git

init_main()

del init_main

__hash__ = git.git_hash(__file__)

from gpustat import GPUStatCollection

import subprocess
def get_pname(id):
    p = subprocess.Popen(["ps -o cmd= {}".format(id)], stdout=subprocess.PIPE, shell=True)
    return str(p.communicate()[0])

def cuda_process():
    gpu_stat = GPUStatCollection.new_query()



## Libraries
import platform
import socket
import os

## Set up the directory automatically

def path_init():
    if socket.gethostname()=='LAPTOP-IQ8L6L26':
        subdir="C:/Users/emily/Documents/ML_spectroscopy_thesis/"
    elif socket.gethostname()=='rainbow':
        #subdir = "/net/ipa-gate/export/ipa/quanz/user_accounts/egarvin/Thesis/"
        subdir = "/home/ipa/quanz/user_accounts/egarvin/Thesis/"
    elif socket.gethostname() == 'sunray':
        subdir = "/home/ipa/quanz/user_accounts/egarvin/Thesis/"
    elif socket.gethostname() == 'bluesky':
        subdir = "/home/ipa/quanz/user_accounts/egarvin/Thesis/"
    elif socket.gethostname() == 'spaceml4':
        subdir = "/home/ipa/quanz/user_accounts/egarvin/Thesis/"
    elif socket.gethostname()[:-2] == 'eu-login-':
        subdir = "/cluster/home/egarvin/Thesis/"
    return subdir

#def path_init():
#    subdir = "/Thesis/"
#    return subdir

def global_settings():
    x=0 #default=0 # 6 #4
    alpha = 4
    beta =  50
    version = 5
    return [x, alpha, beta, version]



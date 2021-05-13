####################################################################
# Fetch and download the data needed for training the emulators
####################################################################
from os import makedirs
from os.path import join
import urllib.request

# User options
DATA_TARGET_DIR = "test"
DATA_PATH = join("/Volumes/Extreme SSD/MatrixElements", DATA_TARGET_DIR)

####################################################################
# Figshare has ints associated with each public file
# (These may not be stable, but hopefully are)
# Don't change these unless you have a reason
####################################################################
ID_DICT = {
    #'E1A_3BFS_ONLY' : 27925692,
    'E1A_NN_AND_3BFS' : 27925698,
    'H_H3_3BFS_ONLY' : 27925701,
    'H_H3_NN_AND_3BFS' : 27925704,
    'H_HE3_3BFS_ONLY' : 27925707,
    'H_HE3_NN_AND_3BFS' : 27925710,
    'H_HE4_3BFS_ONLY' : 27925713,
    'H_HE4_NN_AND_3BFS' : 27925716,
    'R2_HE4' : 27925719
}

FILENAMES = {
    #'E1A_3BFS_ONLY' : 'E1A_cut450_Nmax40_hw36_NNLO_450_vary_3bfs.h5',
    'E1A_NN_AND_3BFS': 'E1A_cut450_Nmax40_hw36_NNLO_450_vary_nn-and-3bfs.h5',
    'H_H3_3BFS_ONLY' : 'H_H3_NNLO_450_Nmax40_hw36_vary_3bfs.h5',
    'H_H3_NN_AND_3BFS' : 'H_H3_NNLO_450_Nmax40_hw36_vary_nn-and-3bfs.h5',
    'H_HE3_3BFS_ONLY' : 'H_He3_NNLO_450_Nmax40_hw36_vary_3bfs.h5',
    'H_HE3_NN_AND_3BFS': 'H_He3_NNLO_450_Nmax40_hw36_vary_nn-and-3bfs.h5',
    'H_HE4_3BFS_ONLY'  : 'H_He4_NNLO_450_Nmax18_hw36_vary_3bfs.h5',
    'H_HE4_NN_AND_3BFS': 'H_He4_NNLO_450_Nmax18_hw36_vary_nn-and-3bfs.h5',
    'R2_HE4' : 'r2_He4_Nmax18_hw36.h5'
}
DOWNLOAD_ROOT = "https://ndownloader.figshare.com/files/"

choice = input('Are you sure you want to go ahead and download these?\n enter y/n: ')
if choice != 'y':
    quit()

makedirs(DATA_PATH, exist_ok=True)

for key in ID_DICT:
    print('> Fetching',key,'with ID',ID_DICT[key])
    print('> Be patient...')
    data_url = DOWNLOAD_ROOT + str(ID_DICT[key])
    #print(data_url)
    urllib.request.urlretrieve(data_url, join(DATA_PATH, FILENAMES[key]))


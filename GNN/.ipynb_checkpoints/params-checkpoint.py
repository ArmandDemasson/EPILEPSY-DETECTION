import os
################################################# DATA ################################################# 

#Path to where the binary files containing windows are savec
path_extracted_data = '/sps/crnl/pmouches/data/MEG_PHRC_2006_Ap18_preprocessed/'
#Path to save the model files and model results
path_writing_data = '/sps/crnl/ademasson/data/GNN/'

# subjects = [str(int(sub[9:12])) for sub in os.listdir(path_extracted_data) if (os.path.isfile(os.path.join(path_extracted_data, sub)) and ("_new_labels" in sub))]
subjects = ['97', '83', '69', '68', '61', '54', '38', '27', '25', '22', '18']

sfreq = 150  # sampling frequency of the data in Hz
window_size = 0.2
dim = (int(sfreq*window_size), 274,1) # sample shape
save_emb = False

################################################# CV ################################################# 
splits = len(subjects)
nb_valid_subjects = 2

################################################# MODEL ################################################# 
#Name of the model to run, model archtectures are defined in the Models folder.
#Non exhaustive list of availaible models "CNNGCN", "CNNGCN" , "EMSNet" , 'CNN_Tjepkema', 'EMSNet', 'ResBlock_Meg1', 'ResBlock_Meg2', 'ResBlock_Meg2', 'SpikeNet', 'FAMED'
#See Model folder to see all possible models
model = "EEGCNNOnly"
#Whether to train the model on balanced data (majority class= windows without spikes is downsampled to match the number of windows with spikes)
balanced = True 
#Set to true only to run the CNN-Graph model called "CNNGCN"
need_features = True
batch_size = 64
augmentation = False

################################################# LOSS ################################################# 
learning_rate = 0.001#0.0001
weight_decay = 0.001
alpha_fl = 0.95
gamma_fl = 2

num_epochs = 20
n_epochs_stop = 10

# Warm up / step pour la modification du leanring rate
warmup_epochs = 10
beta = 0.1
step_size = 30

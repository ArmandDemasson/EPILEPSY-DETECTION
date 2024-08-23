import numpy as np
import os
import os.path as op
import gc
import csv
import pickle
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score, f1_score, confusion_matrix
# import matplotlib.pyplot as plt

def save_obj(obj, name, path):
    with open(op.join(path, name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name, path):
    with open(op.join(path, name), 'rb') as f:
        return pickle.load(f)

def save_model_results(file_path,model_name,y_test, y_pred, fold):

  tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
  specificity = tn / (tn+fp)

  fields=[fold,accuracy_score(y_test,y_pred),f1_score(y_test,y_pred),specificity,recall_score(y_test,y_pred),confusion_matrix(y_test, y_pred).ravel()]   

  print(fields)
  add_header=False
  if not (os.path.exists(file_path+model_name+'_cvresults.csv')):
    add_header=True

  with open(file_path+model_name+'_cvresults.csv', 'a', newline='') as f:
    writer = csv.writer(f)           
    if add_header:
        writer.writerow(["fold","accuracy","f1-score","specificity","sensitivity","confusion matrix"])
    writer.writerow(fields)

def save_model_predictions(X_test_ids,path_extracted_data,file_path,model_name,y_test,y_pred):
  sub = X_test_ids[:,1]
  win = X_test_ids[:,0]
  lab = X_test_ids[:,2]

  prevsub = 1000

  for ind, i in enumerate(sub):
      if (i != prevsub):
          y_timing_data = load_obj('data_raw_'+str(i).zfill(3)+'_b3_timing.pkl',path_extracted_data)
          y_block_data = load_obj('data_raw_'+str(i).zfill(3)+'_b3_blocks.pkl',path_extracted_data)
          y_label_data = load_obj('data_raw_'+str(i).zfill(3)+'_b3_new_labels.pkl',path_extracted_data)

      y_timing = y_timing_data[win[ind]]
      y_block = y_block_data[win[ind]]
      y_label = y_label_data[win[ind]]

      add_header=False
      if not (os.path.exists(file_path+model_name+'_cvpredictions.csv')):
          add_header=True

      with open(file_path+model_name+'_cvpredictions.csv', 'a', newline='') as f:
          writer = csv.writer(f)           
          if add_header:
              writer.writerow(["subject","block","timing","true","test","pred"])
          writer.writerow([i,y_block,y_timing,y_label,y_test[ind],y_pred[ind]])

      prevsub=i

def save_model_embeddings(reading_embeddings_cnn, reading_labels, reading_outputs, path_writing_data, model_name, fold):
  np.save(path_writing_data+ model_name + str(fold) +'_labelsh_umap.npy', reading_labels)
  np.save(path_writing_data+ model_name + str(fold) +'_embeddings_cnn_umap.npy', reading_embeddings_cnn)
  np.save(path_writing_data+ model_name + str(fold) +'_outputsh_umap_balanced_test.npy', reading_outputs)

# def plot_epochs_metric(train, valid, file_path, model_name, fold):
#     plt.figure()
#     plt.plot(np.linspace(0,len(train),num=len(train)), train)
#     plt.plot(np.linspace(0,len(valid),num=len(valid)), valid)
#     plt.title('model ' + model_name + str(fold))
#     plt.ylabel('loss', fontsize='large')
#     plt.xlabel('epoch', fontsize='large')
#     plt.legend(['train', 'val'], loc='upper left')
#     plt.savefig(file_path + model_name + str(fold) + '.png', bbox_inches='tight')
#     plt.close()


 

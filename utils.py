import csv
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
class Logger(object):
    def __init__(self, path, header):
        self.log_file = open(path, 'a')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del__(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()

def compute_ood_metrics(in_scores, out_scores):
    # Placeholder for actual OOD metric computation
    # You can use libraries like sklearn to compute AUROC, AUPR, etc.
    labels = [1] * len(in_scores) + [0] * len(out_scores)  # 1 for in-distribution, 0 for out-of-distribution
    scores = np.concatenate([in_scores, out_scores])
    
    auroc = roc_auc_score(labels, scores)
    aupr = average_precision_score(labels, scores)
    
    return {"AUROC": auroc, "AUPR": aupr}
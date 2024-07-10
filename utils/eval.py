from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def get_confusion_matrix_figure(y_true, y_pred, labels=None, title="Confusion matrix"):
    plt.figure(dpi=600)
    label_codes = np.arange(len(labels)) if labels is not None else None
    cm = confusion_matrix(y_true, y_pred, labels=label_codes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    values_format = None # Format specification for values in confusion matrix. If `None`, the format specification is 'd' or '.2g' whichever is shorter.
    disp.plot(
        include_values=True,
        cmap=plt.cm.Blues, # 'viridis'
        ax=None, 
        xticks_rotation='horizontal',
        values_format=values_format
    )
    fig = disp.figure_
    fig.suptitle(title)
    return fig


import torch
import numpy as np
from torchmetrics import Metric
from torchmetrics.functional import auroc


class PathologyLocalizationAUC(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

        # ImaGenome triplets
        self.attributes = ['lung opacity','pleural effusion','atelectasis','enlarged cardiac silhouette','pulmonary edema/hazy opacity'
                ,'pneumothorax','consolidation','fluid overload/heart failure','pneumonia']
        self.locations = ['right lung','right apical zone','right upper lung zone','right mid lung zone','right lower lung zone','right hilar structures'
                ,'left lung','left apical zone','left upper lung zone','left mid lung zone','left lower lung zone','left hilar structures'
                ,'right costophrenic angle','left costophrenic angle','mediastinum','upper mediastinum','cardiac silhouette','trachea']
        relation = 'in the'
        self.triplets = [f'{a} {relation} {l}' for a in self.attributes for l in self.locations]

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.preds.append(preds)
        self.target.append(target)

    def compute(self):
        preds = torch.cat(self.preds)
        target = torch.cat(self.target)

        num_attributes = len(self.attributes)
        num_locations = len(self.locations)

        # Reshape the flattened tensors
        preds = preds.view(-1, num_locations, num_attributes)
        target = target.view(-1, num_locations, num_attributes)

        aucs = {}
        for j in range(num_attributes):
            aucs[j] = {}
            for i in range(num_locations):
                if any(target[:, i, j] == -1):
                    aucs[j][i] = float('nan')
                else:
                    auc = auroc(preds[:, i, j], target[:, i, j], task="binary").item()
                    aucs[j][i] = auc if auc != 0 else float('nan')

        avg_aucs = {f"mAUC {attribute.replace('/', ' ')}": np.nanmean(list(aucs[j].values())) for j, attribute in enumerate(self.attributes)}
        avg_aucs['mAUC mean'] = np.nanmean(list(avg_aucs.values()))
        return avg_aucs

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from matplotlib.pyplot import figure
import numpy as np

class Confusion_Matrix:
    def __init__(self,y_true,y_pred,classes):
        self.y_true = y_true
        self.y_pred = y_pred
        self.cm=confusion_matrix(y_true=self.y_true,y_pred=self.y_pred)
        self.classes=classes

    def cm_plot(self,
                cmap=plt.cm.Blues,
                title='Matriz de Confusão',
                x_label='Classe Predita',
                y_label='Classe Real',
                fig_size=(10,7),
                dpi=80,
                normalize=False):
        figure(figsize=fig_size, dpi=dpi)
        plt.imshow(self.cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, self.classes, rotation=45)
        plt.yticks(tick_marks, self.classes)

        if normalize:
            self.cm = self.cm.astype('float')/self.cm.sum(axis=1)[:, np.newaxis]
            print("Matriz de confusão com normalização")
        else:
            print("Matriz de confusão sem normalização")

        thresh = self.cm.max()/2.

        for i,j in itertools.product(range(self.cm.shape[0]), range(self.cm.shape[1])):
            plt.text(j, i, str(self.cm[i,j]), horizontalalignment="center", color='white' if self.cm[i,j]> thresh else "black")

        plt.tight_layout()
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.show()


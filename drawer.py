from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


class Drawer:
    def __init__(self, true_label, pred_label, classes):
        self.true_label = true_label
        self.pred_label = pred_label
        self.classes = classes
        self.confusion = confusion_matrix(true_label, pred_label)

    def plot_confusion_matrix(self):

        plt.imshow(self.confusion, cmap=plt.cm.Accent)
        classes = range(self.classes)
        indices = range(len(self.confusion))
        plt.xticks(indices, classes)
        plt.yticks(indices, classes)
        plt.colorbar()
        plt.xlabel('True label')
        plt.ylabel('Predicted label')
        plt.title('Confusion matrix')

        for first_index in range(len(self.confusion)):
            for second_index in range(len(self.confusion[first_index])):
                plt.text(first_index, second_index, self.confusion[second_index][first_index])
        plt.show()


if __name__ == '__main__':
    true_label = [1, 1, 1, 1, 2, 0, 0, 2, 0, 1]
    pred_label = [2, 2, 2, 1, 2, 1, 1, 2, 0, 0]
    classes = 3

    drawer = Drawer(true_label, pred_label, classes)
    drawer.plot_confusion_matrix()

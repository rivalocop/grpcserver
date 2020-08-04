import numpy as np


class ImMotion:
    def __init__(self, face, labels, model):
        self.face = face
        self.labels = labels
        self.model = model
        self.result = self.__predict()

    def __predict(self):
        predictions = self.model.predict(self.face)[0]
        j = np.argmax(predictions)
        j = int(j)
        label = self.labels.classes_[j]
        return {'label': label, 'confidence': predictions[j]}

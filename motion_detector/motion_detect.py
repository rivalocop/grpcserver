import numpy as np


class ImMotion:
    def __init__(self, face, labels, model, expected_label):
        self.face = face
        self.labels = labels
        self.model = model
        self.threshold = 0.6
        self.expected_label = expected_label

    def predict(self):
        result = {"isPredicted": False}
        predictions = self.model.predict(self.face)[0]
        j = np.argmax(predictions)
        j = int(j)
        label = self.labels.classes_[j]
        if predictions[j] > self.threshold and self.expected_label is label:
            result["isPredicted"] = True
        return result

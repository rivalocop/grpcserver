import cv2
import numpy as np
import tensorflow as tf


def prepare_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (224, 224))
    image = cv2.flip(image, 1)
    return image


class ImFace:
    def __init__(self, image, net):
        self.image = prepare_image(image)
        self.net = net
        self.height = self.__get_shape()[0]
        self.width = self.__get_shape()[1]
        self.face = self.__extract_face()

    def __convert_blob_image(self):
        blob = cv2.dnn.blobFromImage(cv2.resize(self.image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        return blob

    def __extract_face(self):
        face = None
        blob_image = self.__convert_blob_image()
        self.net.setInput(blob_image)
        detections = self.net.forward()

        if len(detections) > 0:
            # we're making the assumption that each image has only ONE
            # face, so find the bounding box with the largest probability
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
                # the face and extract the face ROI
                box = detections[0, 0, i, 3:7] * \
                    np.array([self.width, self.height,
                              self.width, self.height])
                (start_x, start_y, end_x, end_y) = box.astype("int")
                face = self.image[start_y:end_y, start_x:end_x]
                face = cv2.resize(face, (64, 64))
                face = face.astype("float") / 255.0
                # face = tf.keras.preprocessing.image.img_to_array(face)
                face = np.expand_dims(face, axis=0)

        return face

    def __get_shape(self):
        (h, w) = self.image.shape[:2]
        return h, w

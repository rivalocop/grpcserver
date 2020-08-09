import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# initialize Redis connection settings
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
# initialize constants used to control image spatial dimensions and
# data type
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANS = 3
IMAGE_DTYPE = "float32"
IMAGE_WIDTH_IPHONESE = 480  # 288
IMAGE_HEIGHT_IPHONESE = 640  # 352
# 640x480
# initialize constants used for server queuing
IMAGE_QUEUE = "image_queue"
BATCH_SIZE = 32
SERVER_SLEEP = 0.25
CLIENT_SLEEP = 0.25

# File path to opencv face detection deep learning
CAFFE_MODEL = BASE_DIR + '/models/res10_300x300_ssd_iter_140000.caffemodel'
DEPLOY_FILE = BASE_DIR + '/models/deploy.prototxt.txt'

# REGEX STRING
EMAIL_REGEX = r'(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)'
IDENTITY_DOCUMENT_REGEX = r"(00[1-9]{1}|0[1-8][0-9]|09[0-6])[0-9]{9}"
DOB_REGEX = r"([0-2][0-9]|(3)[0-1])(\/)(((0)[0-9])|((1)[0-2]))(\/)\d{4}"
CAPITALIZE_REGEX = r"[A-Z]"

# LABELS AND MOTION MODEL
LABELS = BASE_DIR + '/models/le.pickle'
MOTION = BASE_DIR + '/models/trained_model.h5'
# MOTION = BASE_DIR + '/models/vgg.h5'

LABELS_LIST = ['down' 'front' 'left' 'right' 'up']
PREDICT_QUEUE = 'predict_queue'

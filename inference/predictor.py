from __future__ import print_function

import json
import os
import sys
from urllib import response
import torch
import jsonpickle

import flask
import pandas as pd
from flask import request, jsonify
from PIL import Image
import json
import base64
from io import BytesIO

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")

# import some common libraries
import sys
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# Detic libraries
sys.path.insert(0, 'third_party/CenterNet2/projects/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test
from detic.modeling.text.text_encoder import build_text_encoder

cfg = get_cfg()
add_centernet_config(cfg)
add_detic_config(cfg)
cfg.merge_from_file("configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
cfg.MODEL.WEIGHTS = model_path + '/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True # For better visualization purpose. Set to False for all classes.
if not torch.cuda.is_available():
    cfg.MODEL.DEVICE='cpu'
predictor = DefaultPredictor(cfg)


'''
BUILDIN_CLASSIFIER = {
    'lvis': 'datasets/metadata/lvis_v1_clip_a+cname.npy',
    'objects365': 'datasets/metadata/o365_clip_a+cnamefix.npy',
    'openimages': 'datasets/metadata/oid_clip_a+cname.npy',
    'coco': 'datasets/metadata/coco_clip_a+cname.npy',
}

BUILDIN_METADATA_PATH = {
    'lvis': 'lvis_v1_val',
    'objects365': 'objects365_v2_val',
    'openimages': 'oid_val_expanded',
    'coco': 'coco_2017_val',
}
'''

def get_clip_embeddings(vocabulary, prompt='a '):
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    texts = [prompt + x for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb


class ImageService(object):

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        return predictor.model
    
    @classmethod
    def predict(cls, custom_vocabulary, input):
        """For the input, do the predictions and return them.
        Args:
            custom_vocabulary (string): string list with words to search into image
            input (image): The Image that will be analyzed"""
        
        metadata = MetadataCatalog.get("__unused")
        metadata.thing_classes = custom_vocabulary.split(',')
        classifier = get_clip_embeddings(metadata.thing_classes)
        num_classes = len(metadata.thing_classes)
        reset_cls_test(predictor.model, classifier, num_classes)
        
        outputs = predictor(input)

        MetadataCatalog.remove("__unused")
        return outputs


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ImageService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")



@app.route("/invocations", methods=["POST"])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as base64 str, convert
    it to binary image for internal use and then convert the predictions back to jsonpickle.
    """

    json_data = request.get_json() #Get the POSTed json
    dict_data = json.loads(json_data) #Convert json to dictionary

    img = dict_data["img"] #Take out base64# str
    img = base64.b64decode(img) #Convert image data converted to base64 to original binary data# bytes
    img = BytesIO(img) # _io.Converted to be handled by BytesIO pillow
    jpg_as_np = np.frombuffer(img.getbuffer(), dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, flags=1)
    
    custom_words = dict_data["labels"]
    print('Looking for "{}" on the image.'.format(custom_words), file=sys.stderr)    

    # Do the prediction
    predictions = ImageService.predict(custom_words, img)

    #print('predictions: ',predictions, file=sys.stderr)

    resp = jsonpickle.encode(predictions)

    return flask.Response(response=resp, status=200, mimetype="application/json")
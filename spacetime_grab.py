# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 17:13:03 2019

@author: ysirotin
"""

import cv2
import time
import numpy as np
import base64
import json
import zlib

def mat_to_json(mat):
    "encode numpy matrix as a json string"
    dtype = str(mat.dtype)
    shape = list(mat.shape)
    data = base64.b64encode(zlib.compress(mat.tobytes())).decode()
    dictdata = {"dtype":dtype,"shape":shape,"data":data}
    return json.dumps(dictdata)
    
def mat_from_json(jsondata):
    dictdata = json.loads(jsondata)
    data = zlib.decompress(base64.b64decode(dictdata['data']))
    dtype = dictdata["dtype"]
    shape = tuple(dictdata["shape"])
    return np.reshape(np.frombuffer(data,dtype=dtype),shape)
    
vid = cv2.VideoCapture(0)
w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

mat = np.zeros((h,w,30),dtype='uint16')
t = np.zeros((30),dtype='float')

for ii in range(30):
    ret, frame = vid.read()
    t[ii] = time.time()
    mat[:,:,ii] = np.sum(frame,axis=2)

vid.release()

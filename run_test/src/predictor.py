import os
import cv2
import time
import onnxruntime
import numpy as np


class Predictor(object):
    @classmethod
    def get_model(cls, model_path):
        cls.model = load_model(model_path)


    @classmethod
    def predict(cls, v):
        preprocessed = preprocess(v)
        start = time.time()
        output = cls.model.run(["output"], preprocessed)
        runtime = time.time() - start
        pred = postprocess_output(output)

        return pred, runtime


def load_model(model_path):
    ort_session = onnxruntime.InferenceSession(os.path.join(model_path, "model.onnx"), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    for h, w in [(324, 576), (372, 496), (375, 500), (375, 562), (378, 504), (408, 544), (486, 648), (510, 384), (616, 408), (752, 502)]:
        ort_inputs = {"input": np.zeros((1,3,h,w), dtype=np.float32)}
        _ = ort_session.run(["output"], ort_inputs) # warm up
        ort_inputs = {"input": np.zeros((1,3,w,h), dtype=np.float32)}
        _ = ort_session.run(["output"], ort_inputs) # warm up

    return ort_session


def preprocess(img):
    ort_inputs = {"input": np.array([cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose((2,0,1))], dtype=np.float32)/255}

    return ort_inputs


def postprocess_output(output):
    ndarray = cv2.cvtColor((output[0].transpose((0,2,3,1))[0]*255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    return ndarray

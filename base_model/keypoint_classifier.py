import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

class KeyPointClassifier(object):
    def __init__(self, model_path, num_threads=1,):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, landmark_list,):

        input_details_tensor_index = self.input_details[0]['index']

        self.interpreter.set_tensor(input_details_tensor_index, landmark_list)

        #
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        # 結果のインデックス
        result_index = np.argmax(np.squeeze(result))

        # 信頼度
        confidence = np.max(np.squeeze(result))

        # 結果のインデックスと信頼度を返す
        return result_index, confidence

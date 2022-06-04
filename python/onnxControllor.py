# import os, sys

# sys.path.append(os.getcwd())
import onnxruntime
import onnx


class ONNXModel():
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
        self.input_name = self.get_input_name()
        self.output_name = self.get_output_name()
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))

    def get_output_name(self):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        onnx_session = self.onnx_session
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        onnx_session = self.onnx_session
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, all_tensors):
        """
        input_feed={self.input_name: image_tensor}
        :param input_name:
        :param all_tensors:
        :return:
        """
        input_feed = {}
        assert len(self.input_name) == len(all_tensors),f"self.input_name={self.input_name} but all_tensors={all_tensors}"
        for name, tensor in zip(self.input_name, all_tensors):
            input_feed[name] = tensor
        return input_feed

    def forward(self, all_tensors, ret_name_id=None):
        '''
        image_tensor = image.transpose(2, 0, 1)
        image_tensor = image_tensor[np.newaxis, :]
        onnx_session.run([output_name], {input_name: x})
        :param image_tensor:
        :return:
        '''
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_tensor})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: image_tensor})
        input_feed = self.get_input_feed(all_tensors)
        res = self.onnx_session.run(self.output_name, input_feed=input_feed)
        if ret_name_id:
            return res[ret_name_id]
        else:
            return res

import numpy as np

if __name__ == "__main__":
    model = ONNXModel(
        onnx_path=r"./mymodel.onnx")
    obs = np.ones((1, 3, 15, 15)).astype(np.float32)
    from time import time
    t = time()
    for _ in range(1600):
        res = model.forward(all_tensors=[obs])
    print(time()-t)
    # print(res[0].shape)
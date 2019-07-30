import numpy as np
import torch
import torchvision.models as models
import os
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
gfile = tf.gfile

#
# IMAGENET_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_SD = (0.229, 0.224, 0.225)
#
# def normalize_by_imagenet(image_stack):
#     # Copy constant values multiple times to fill up a tensor of length SEQ_LENGTH * len(IMAGENET_MEAN)
#     for i in range(3):
#         image_stack[0, :, :, :] = (image_stack[0, i, :, :] - IMAGENET_MEAN) / IMAGENET_SD
#
#
#     return (image_stack - IMAGENET_MEAN) / IMAGENET_SD

# ''' Convert PyTorch pretrained model to ONNX'''
dummy_input = torch.randn(10, 3, 128, 416, device='cuda')
dummy_input_tf = np.random.uniform(size=(10, 3, 128, 416))
# resnet18 = models.resnet18(pretrained=True).cuda()
#
# input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
# output_names = [ "output1" ]
#
# torch.onnx.export(resnet18, dummy_input, "resnet_2.onnx", verbose=True, input_names=input_names, output_names=output_names)

# '''Load ONNX'''
# # Load the ONNX model
# model = onnx.load("resnet.onnx")
#
# # Check that the IR is well formed
# onnx.checker.check_model(model)
#
# # Print a human readable representation of the graph
# onnx.helper.printable_graph(model.graph)



#
# # # Set device to be used
#
#
# dummy_input = torch.from_numpy(X_test).float().to(device)
# dummy_output = resnet18(dummy_input)
# # print(dummy_output)
#
# # Export to ONNX format
# torch.onnx.export(resnet18, dummy_input, './models/resnet18.onnx', input_names=['input'], output_names=['output'])


# #
# # if not os.path.exists('./models/'):
# #     os.mkdir('./models/')
# #
# # torch.save(resnet18.state_dict(), './models/model_simple.pt')


# # Load ONNX model and convert to TensorFlow format
# model_onnx = onnx.load('./resnet_2.onnx')
#
# tf_rep = prepare(model_onnx)
#
# # Print out tensors and placeholders in model (helpful during inference in TensorFlow)
# print(tf_rep.tensor_dict)

# # Export model as .pb file
# tf_rep.export_graph('./models/resnet_2.pb')

# GRAPH_PB_PATH = './models/resnet18.pb'
# with tf.Session() as sess:
#    print("load graph")
#    with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
#        graph_def = tf.GraphDef()
#    graph_def.ParseFromString(f.read())
#    sess.graph.as_default()
#    tf.import_graph_def(graph_def, name='')
#    graph_nodes=[n for n in graph_def.node]
#    names = []
#    for t in graph_nodes:
#       names.append(t.name)
#    print(names)
#
# def load_pb(path_to_pb):
#     with tf.gfile.GFile(path_to_pb, 'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#     with tf.Graph().as_default() as graph:
#         tf.import_graph_def(graph_def, name='')
#         return graph
#
#
# tf_graph = load_pb('./models/resnet_2.pb')
# sess = tf.Session(graph=tf_graph)
#
# output_tensor = tf_graph.get_tensor_by_name('add_9:0')
# input_tensor = tf_graph.get_tensor_by_name('actual_input_1:0')
#
# output = sess.run(output_tensor, feed_dict={input_tensor: dummy_input_tf})
# print(output)


print('loading onnx model')
onnx_model = onnx.load('resnet_2.onnx')

print('prepare tf model')
tf_rep = prepare(onnx_model)
# print(tf_rep.predict_net)
print('-----')
print(tf_rep.tensor_dict)

out = tf_rep.run(dummy_input_tf)
# print(out)
#


with tf.Session() as sess:
    print("load graph")
    sess.graph.as_default()
    # tf.import_graph_def(tf_rep.graph.as_graph_def(), name='')
    # print(tf.trainable_variables())

#     print(sess.run(fc.bias))
#
    names = [n.name for n in tf.get_default_graph().as_graph_def().node]
    print(names)
    # input_tensor = sess.graph.get_tensor_by_name('actual_input_1:0')
    # print(sess.run(out, {'actual_input_1:0' : dummy_input_tf}))

    # print(input_tensor)
    # sess.run(input_tensor)

    # res = sess.run(out, {input_tensor: tf.convert_to_tensor(dummy_input_tf)})
    # print(res)

# tf_rep.export_graph('train/tf.pb')

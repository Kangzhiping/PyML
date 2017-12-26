# google 现成的模型分类
#$ cd tensorflow/models/image/imagenet/
#$ python classify_image.py --image_file ~/Desktop/bigcat.jpg

#训练
#$ python tensorflow/tensorflow/examples/image_retraining/retrain.py --bottleneck_dir bottleneck
#    --how_many_training_steps 4000 --model_dir model --output_graph output_graph.pb
#    --output_labels output_labels.txt --image_dir girl_types/

# 使用训练好的模型进行图片分类

import tensorflow as tf
import sys

# 命令行参数，传入要判断的图片路径
image_file = sys.argv[1]
# print(image_file)

# 读取图像
image = tf.gfile.FastGFile(image_file, 'rb').read()

# 加载图像分类标签
labels = []
for label in tf.gfile.GFile("output_labels.txt"):
    labels.append(label.rstrip())

# 加载Graph
with tf.gfile.FastGFile("output_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    predict = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image})

    # 根据分类概率进行排序
    top = predict[0].argsort()[-len(predict[0]):][::-1]
    for index in top:
        human_string = labels[index]
        score = predict[0][index]
        print(human_string, score)
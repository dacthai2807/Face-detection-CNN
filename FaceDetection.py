import tensorflow as tf
from cv2 import cv2, os
import matplotlib.pyplot as plt
import numpy as np
import random
from imutils.object_detection import non_max_suppression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

DET_SIZE = (200, 200)
IMAGE_SIZE = (28, 28)

SNAP_COUNT = 10 #Number of random snapshots per non-face image
SNAP_SIZE = 20 #Size of snapshots of non-faces

POS = [1, 0] # Vector output for faces
NEG = [0, 1] # Vector output for non-faces

path_pos = 'photo/Positive' # Positive dataset
path_neg = 'photo/Negative' # Negative dataset

def add_image(images, labels, image, label):
    image = cv2.resize(image, IMAGE_SIZE)
    image = np.reshape(image, IMAGE_SIZE[0] * IMAGE_SIZE[1])
    images.append(image)
    labels.append(label)

def add_snap(images, labels, image, label):
    image = cv2.resize(image, DET_SIZE)

    for i in range(SNAP_COUNT):
        fx = int(random.random() * (DET_SIZE[0] - SNAP_SIZE))
        fy = int(random.random() * (DET_SIZE[1] - SNAP_SIZE))
        snap = image[fx : fx + SNAP_SIZE, fy : fy + SNAP_SIZE]
        add_image(images, labels, snap, label)

def read_data(path, label):
    images = []
    labels = []
    dirs = os.listdir(path)

    for files in dirs:
        file_name = path + "/" + files
        image = cv2.imread(file_name, 0)
        add_image(images, labels, image, label)

        if label == NEG:
            add_snap(images, labels, image, label)
        else: 
            flip_image = cv2.flip(image, 1)
            add_image(images, labels, flip_image, label)
            add_snap(images, labels, image, NEG)
    return images, labels

images_pos, labels_pos = read_data(path_pos, POS)
images_neg, labels_neg = read_data(path_neg, NEG)

images = images_pos + images_neg
labels = labels_pos + labels_neg

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.1, random_state = 42)

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

# Training Parameters
learning_rate = 0.001
batch_size = 100
display_step = 10
num_steps = 500

# Network Parameters
num_classes = 2
num_input = IMAGE_SIZE[0] * IMAGE_SIZE[1]
dropout = 0.75 # Dropout, probability to keep units

# tf graph input
x_hold = tf.placeholder(tf.float32, [None, num_input])
y_hold = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

# Create some wrappers for simplicity
def conv2d(x, W, b, strides = 1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides = [1, strides, strides, 1], padding = 'SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k = 2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize = [1, k, k, 1], strides = [1, k, k, 1], padding = 'SAME')

# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape = [-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k = 2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k = 2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7 * 7 * 64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
    # 1024 inputs, 2 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Construct model
logits = conv_net(x_hold, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y_hold))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_hold, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables 
init = tf.global_variables_initializer()

def random_batch(x_train, y_train, batch_size):
    rnd_indices = np.random.randint(0, len(x_train), batch_size)
    x_batch = x_train[rnd_indices]
    y_batch = y_train[rnd_indices]
    return x_batch, y_batch

# Start training
sess = tf.Session() 
sess.run(init)

for step in range(1, num_steps + 1):
    batch_x, batch_y = random_batch(x_train, y_train, batch_size)
    # Run optimization op (backprop)
    sess.run(train_op, feed_dict = {x_hold: batch_x, y_hold: batch_y, keep_prob: 0.5})

    if step % display_step == 0 or step == 1:
        # Calculate batch loss and accuracy
        loss, acc = sess.run([loss_op, accuracy], feed_dict = {x_hold: batch_x, y_hold: batch_y, keep_prob: 1.0})
        print("Step " + str(step) + ", Minibatch Loss = " + \
                "{:.4f}".format(loss) + ", Training Accuracy = " + \
                "{:.3f}".format(acc))

print("Optimization Finished!")

# Test model
path = "photo/Test/test.jpg" # đường dẫn ảnh muốn test

img = cv2.imread(path, 0)

CONF_THRESH = 0.99 # Confidence threshold to mark a window as a face

X_STEP = 10 # Horizontal slide for the sliding window
Y_STEP = 10 # Vertical stride for the sliding window
WIN_MIN = 60 # Minimum sliding window size
WIN_MAX = 100 # Maximum sliding window size
WIN_STRIDE = 10 # Stride to increase the sliding window

img = cv2.resize(img, DET_SIZE)
detections = []

#Run sliding windows of different sizes
for bx in range(WIN_MIN, WIN_MAX, WIN_STRIDE):
    by = bx
    for x in range(0, img.shape[1] - bx, X_STEP):
        for y in range(0, img.shape[0] - by, Y_STEP):
            sub_img = cv2.resize(img[x : x + bx, y : y + by], IMAGE_SIZE)
            x_test = []
            x_test.append(np.reshape(sub_img, IMAGE_SIZE[0] * IMAGE_SIZE[1]))
            y_test = sess.run(prediction, feed_dict = {x_hold: x_test, keep_prob: 1.0}) [0]
            conf = y_test[0]

            if conf >= CONF_THRESH:
                detections.append((x, y, conf, bx, by))

clone = img.copy()

for (x_tl, y_tl, _, w, h) in detections:
    cv2.rectangle(clone, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 255, 0), thickness = 2)

rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
sc = [score for (x, y, score, w, h) in detections]
sc = np.array(sc)
pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.3)

for(xA, yA, xB, yB) in pick:
    cv2.rectangle(img, (xA, yA), (xB, yB), (0, 255, 0), 2)

plt.axis("off")
plt.imshow(cv2.cvtColor(clone, cv2.COLOR_BGR2RGB))
plt.title("Raw Detection before NMS")
plt.show()

plt.axis("off")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Final Detections after applying NMS")
plt.show()


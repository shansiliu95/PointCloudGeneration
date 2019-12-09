import tensorflow as tf
import pdb
import numpy as np

class SimpleNet():
    def __init__(self):
        self.cond_scale = dict()
        self.cond_shift = dict()
        self.cond_alpha = dict()
        self.enc_channels = [32, 32, 64 ,64, 128]
        self.dec_channels = [128, 64, 64, 32, 32]
    
    def point_sort(self, points):
        indices = tf.contrib.framework.argsort(points[:,0], axis=0)
        points_sorted = tf.gather(points, indices=indices, axis=0)
        return points_sorted

    def cond_instance_normalization_plus(self, data, output, cond_scale, cond_shift, cond_alpha, labels, name):
        means = tf.reduce_mean(data, axis=[1,2])
        eps = 1e-5
        m = tf.reduce_mean(means, axis=-1, keep_dims=True)
        v = tf.math.reduce_std(means, axis=-1, keepdims=True)
        means = (means-m) / (v + eps)
        instance_mean, instance_sigma = tf.nn.moments(data, axes=[1,2], keep_dims=True)
        h = (data - instance_mean) / tf.sqrt(instance_sigma + eps)
        if name not in cond_scale:
            cond_scale[name] = tf.get_variable(name + '_gamma', initializer=tf.random.normal(shape=[10, output], mean=1, stddev=0.02), dtype=tf.float32)
            cond_shift[name] = tf.get_variable(name + '_beta', initializer=tf.zeros(shape=[10, output], dtype=tf.float32))
            cond_alpha[name] = tf.get_variable(name + '_alpha', initializer=tf.random.normal(shape=[10, output], mean=1, stddev=0.02), dtype=tf.float32)
        gamma = tf.nn.embedding_lookup(cond_scale[name], labels)
        beta = tf.nn.embedding_lookup(cond_shift[name], labels)
        alpha = tf.nn.embedding_lookup(cond_alpha[name], labels)
        h = h + tf.expand_dims(tf.expand_dims(means, 1),1) * tf.expand_dims(tf.expand_dims(alpha, 1),1)
        h = tf.expand_dims(tf.expand_dims(gamma, 1),1) * h + tf.expand_dims(tf.expand_dims(beta, 1),1)
        return h

    def forward(self, x, labels):
        
        x = 2.0 * x - 1.0
        x = tf.map_fn(self.point_sort, elems=x, dtype=tf.float32, parallel_iterations=x.get_shape().as_list()[0])
        x = tf.expand_dims(x, axis=2)
        x = tf.layers.conv2d(x, 16, (5,1), padding='same')
        layers = []
        for i in range(len(self.enc_channels)):
            x = tf.layers.conv2d(x, self.enc_channels[i], (3,1),padding='same')
            x = self.cond_instance_normalization_plus(x, self.enc_channels[i], self.cond_scale, 
                                                        self.cond_shift, self.cond_alpha, labels, name='enc_'+str(i))
            x = tf.nn.relu(x)
            layers.append(x)
            x= tf.layers.max_pooling2d(x, pool_size=(2,1), strides=2)
        for i in range(len(self.dec_channels)):
            x = tf.layers.conv2d(x, self.dec_channels[i], (3,1), padding='same')
            x = self.cond_instance_normalization_plus(x, self.dec_channels[i], self.cond_scale, 
                                                        self.cond_shift, self.cond_alpha, labels, name='dec_'+str(i))
            x = tf.nn.relu(x)
            x = tf.keras.layers.UpSampling2D(size=(2, 1))(x)
            #x = tf.concat([x, layers[-i-1]], axis=-1)
        
        x = tf.layers.conv2d(x, 3, (1,1))
        x = tf.squeeze(x)
        return x

if __name__ == '__main__':
    points = tf.random_normal(shape=[3,512,3],seed=0)
    labels = tf.zeros(shape=[3],dtype=tf.int32)
    net = SimpleNet()
    output = net.forward(points, labels)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run([points,output]))
    pdb.set_trace()
        
    

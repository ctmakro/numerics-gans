import numpy as np
import tensorflow as tf
import canton as ct
from canton import *
import time

noise_shape = [16]
data_shape = [2]

class ResDense(Can):
    def __init__(self,nip):
        nmp = int(nip/4)
        d1,d2 = [
            Dense(nip, nmp, stddev=1.),
            Dense(nmp, nip, stddev=1.),
        ]

        self.incan([d1,d2])
        self.d1,self.d2 = d1,d2

    def __call__(self,i):
        ident = i
        i = self.d1(i)
        i = Act('selu')(i)
        i = self.d2(i)
        i = Act('selu')(i)

        out = (ident+i)*0.5
        return out

def dis():
    c = Can() # input shape: [batch, 2]

    def bd(): # batch discriminator
        c = Can()
        def call(k): # [batch, dims]
            diffk = k - tf.reverse(k,[0]) # subtract the first-dim reverse of self
            newk = tf.concat([k, diffk*.5], -1) # concat to expand last dim
            return newk # [batch, dims*2]
        c.set_function(call)
        return c

    def lay(i,o,usebd=False):
        c.add(Dense(i,o,stddev=1))
        if usebd:
            c.add(bd())
        # c.add(Drop(0.1))
        c.add(Act('selu'))

    lay(data_shape[0],16)
    # lay(16,16,usebd=True);lay(32,16)
    lay(16,16);lay(16,16)
    c.add(Dense(16,1)) # [batch, 1]
    c.chain()
    return c

def gen():
    c = Can() # input shape: [batch, 8]

    def lay(i,o):
        c.add(Dense(i,o,stddev=1))
        # c.add(Drop(0.1))
        c.add(Act('selu'))

    def rlay(i):
        c.add(ResDense(i))

    lay(noise_shape[0],32)
    # rlay(i)
    lay(32,32)
    lay(32,16)
    c.add(Dense(16,data_shape[0])) # output shape: [batch, 2]
    c.chain()
    return c

def dis():
    c = Can()
    c.add(Dense(data_shape[0], 16))
    c.add(Act('relu'));c.add(Dense(16, 16))
    c.add(Act('relu'));c.add(Dense(16, 16))
    c.add(Act('relu'));c.add(Dense(16, 16))
    c.add(Act('relu'));c.add(Dense(16, 1))
    c.chain()
    return c

def gen():
    c = Can()
    c.add(Dense(noise_shape[0], 16))
    c.add(Act('relu'));c.add(Dense(16, 16))
    c.add(Act('relu'));c.add(Dense(16, 16))
    c.add(Act('relu'));c.add(Dense(16, 16))
    c.add(Act('relu'));c.add(Dense(16, 2))
    c.chain()
    return c

d,g = dis(),gen()

def feed_gen(d,g):
    data = ph(data_shape)

    nshape = [tf.shape(data)[0]] + noise_shape
    # use whatever batchsize the data is in
    noise = tf.random_normal(shape=nshape)

    generated = g(noise)

    gscore = d(generated) # score for generated sample
    rscore = d(data) # score for real data

    # dloss = tf.reduce_mean((gscore-0)**2 + (rscore-1)**2)
    # gloss = tf.reduce_mean((gscore-1)**2)
    d_out_real = rscore
    d_out_fake = gscore

    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_out_real, labels=tf.ones_like(d_out_real)
    ))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_out_fake, labels=tf.zeros_like(d_out_fake)
    ))
    d_loss = d_loss_real + d_loss_fake

    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_out_fake, labels=tf.ones_like(d_out_fake)
    ))

    dloss,gloss = d_loss, g_loss
    # consensus optimization with RMSProp, as described in the paper "Numerics of GANs".
    from optimizers import ConsensusOptimizer as optim
    optimizer = optim(1e-4, alpha = 10.)

    train_step = optimizer.conciliate(dloss,gloss,d.get_weights(),g.get_weights())

    #
    # optimizer = Adam(3e-5)
    # optimizer = tf.train.MomentumOptimizer(1e-4,momentum=0.9)

    # update_wd = optimizer.minimize(dloss,var_list=d.get_weights())
    # update_wg = optimizer.minimize(gloss,var_list=g.get_weights())
    #
    # train_step = [update_wd, update_wg]

    losses = [dloss,gloss]

    def feed(minibatch):
        # actual GAN training function
        nonlocal train_step,losses,noise,data
        sess = get_session()
        res = sess.run([train_step,losses],feed_dict={
            data:minibatch,
        })

        loss_values = res[1]
        return loss_values #[dloss,gloss]

    def gen(noisebatch):
        nonlocal noise,generated
        sess = get_session()
        res = sess.run([generated],feed_dict={
            noise:noisebatch
        })
        return res[0] # generated

    return feed,gen

feed,gen = feed_gen(d,g)
get_session().run(gvi())

from mixture import sample
from plotter import interprocess_scatter_plotter as plotter

stationaryplot = plotter()
dynamicplot = plotter()

batch_size = 512

def r(ep):
    for i in range(ep):
        minibatch = sample(batch_size)
        loss = feed(minibatch)
        print('{}/{} dloss:{:4.2} gloss:{:4.2}'.format(i+1,ep,loss[0],loss[1]))

        if (i-1) % 50 == 0:
            show()

stationaryplot.scatter(sample(100))

def show():
    # sampled = sample(500)
    generated = gen(np.random.normal(size=[800]+noise_shape))

    dynamicplot.clearscatter()
    dynamicplot.scatter(generated)
    # plotscatter(sampled)
    # plotscatter(generated)

if __name__ == '__main__':
    show()

import copy
import random
import statistics

import numpy as np

from utils.Agent import *
import matplotlib.pyplot as plt

import matplotlib.animation as animation
from multiprocessing import Process, Pipe

import io
import PIL.Image, PIL.ImageDraw
import base64
import zipfile
import json
import requests
import matplotlib.pylab as pl
import glob
import IPython.display
import tensorflow as tf

from google.protobuf.json_format import MessageToDict
from tensorflow.python.framework import convert_to_constants
import tensorflow_addons as tfa
from tensorflow.keras.layers import Conv2D,Conv2DTranspose,Dense,Flatten,Reshape
from .FunctionalAutoencoder import prepare_data
from IPython.display import Image, HTML, clear_output ,display
import tqdm

import os
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['FFMPEG_BINARY'] = 'ffmpeg'
import moviepy.editor as mvp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

last_img = np.zeros((128, 96, 3))

CHANNEL_N = 8  # Number of CA state channels
TARGET_PADDING = 16  # Number of pixels used to pad the target image border
TARGET_SIZE = 40
BATCH_SIZE = 8
POOL_SIZE = 1024
CELL_FIRE_RATE = 0.5
EMOJI = '🦎😀💥👁🐠🦋🐞🕸🥨🎄'
TARGET_EMOJI = "🦎"  # @param {type:"string"}
DIVISION = 1#16
EXPERIMENT_TYPE = "Regenerating"  # @param ["Growing", "Persistent", "Regenerating"]Regenerating
EXPERIMENT_MAP = {"Growing": 0, "Persistent": 1, "Regenerating": 2}
EXPERIMENT_N = EXPERIMENT_MAP[EXPERIMENT_TYPE]

USE_PATTERN_POOL = [0, 1, 1][EXPERIMENT_N]
DAMAGE_N = False#[0, 0,1][EXPERIMENT_N]  # Number of patterns to damage in a batch
# last_img = np.zeros((12,8,3))
def runGraph(pipe):
    global last_img
    # Parameters
    x_len = 200  # Number of points to display
    y_range = [10, 40]  # Range of possible Y values to display

    # Create figure for plotting
    fig = plt.figure()
    ax = plt.subplot(111)
    xs = list(range(0, 200))
    ys = [0] * x_len

    # Create a blank line. We will update the line in animate
    line = ax.imshow(last_img)

    # This function is called periodically from FuncAnimation
    def animate(i, ys):
        # Read temperature (Celsius) from TMP102
        temp_c = np.random.random(1) * 40

        # Add y to list
        ys.append(temp_c)

        # Limit y list to set number of items
        ys = ys[-x_len:]

        # Update line with new Y values
        # print(pipe.recv())
        line.set_array(pipe.recv())

        return line,

    # Set up plot to call animate() function periodically

    ani = animation.FuncAnimation(fig,
                                  animate,
                                  fargs=(ys,),
                                  interval=50,
                                  blit=True)
    plt.show()

class VideoWriter:
  def __init__(self, filename, fps=30.0, **kw):
    self.writer = None
    self.params = dict(filename=filename, fps=fps, **kw)

  def add(self, img):
    img = np.asarray(img)
    if self.writer is None:
      h, w = img.shape[:2]
      self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
    if img.dtype in [np.float32, np.float64]:
      img = np.uint8(img.clip(0, 1)*255)
    if len(img.shape) == 2:
      img = np.repeat(img[..., None], 3, -1)
    self.writer.write_frame(img)

  def close(self):
    if self.writer:
      self.writer.close()

  def __enter__(self):
    return self

  def __exit__(self, *kw):
    self.close()


class SamplePool:
  def __init__(self, *, _parent=None, _parent_idx=None, **slots):
    self._parent = _parent
    self._parent_idx = _parent_idx
    self._slot_names = slots.keys()
    self._size = None
    for k, v in slots.items():
      if self._size is None:
        self._size = len(v)
      assert self._size == len(v)
      setattr(self, k, np.asarray(v))

  def sample(self, n):
    idx = np.random.choice(self._size, n, False)
    batch = {k: getattr(self, k)[idx] for k in self._slot_names}
    batch = SamplePool(**batch, _parent=self, _parent_idx=idx)
    return batch

  def commit(self):
    for k in self._slot_names:
      getattr(self._parent, k)[self._parent_idx] = getattr(self, k)
@tf.function
def make_circle_masks(n, h, w):
  x = tf.linspace(-1.0, 1.0, w)[None, None, :]
  y = tf.linspace(-1.0, 1.0, h)[None, :, None]
  center = tf.random.uniform([2, n, 1, 1], -0.5, 0.5)
  r = tf.random.uniform([n, 1, 1], 0.1, 0.4)
  x, y = (x-center[0])/r, (y-center[1])/r
  mask = tf.cast(x*x+y*y < 1.0, tf.float32)
  return mask


def plot_loss(loss_log):
  pl.figure(figsize=(10, 4))
  pl.title('Loss history (log10)')
  pl.plot(np.log10(loss_log), '.', alpha=0.1)
  pl.show()

#TO DO : Create several pools in order distant places in cellular automata to communicate
init = np.ones((10, 1))
global_pool = tf.Variable(init)
class CAModel(tf.keras.Model):

  def __init__(self, channel_n=CHANNEL_N, fire_rate=CELL_FIRE_RATE):
    super().__init__()
    self.channel_n = channel_n
    self.fire_rate = fire_rate

    self.dmodel = tf.keras.Sequential([
            Conv2D(48, 3),
        Conv2DTranspose(48, 3),
          Conv2D(128, 1, activation=tf.nn.relu),
          Conv2D(self.channel_n, 1, activation=None,
              kernel_initializer=tf.zeros_initializer),
    ])

    self.global_attention = tf.keras.Sequential([
        Flatten(),
        Dense(3* 3* 24* 3),
        Reshape((3, 3, 24, 3))
    ])

    self(tf.zeros([1, 64, 64, channel_n]))  # dummy call to build the model

  @tf.function
  def perceive(self, x, angle=0.0):
    #1,0,0 nothing special
    identify = np.float32([0, 1, 0])# origin 0,1,0 [1, 0, 1]-strange shape [0, 0, 1] strange shape [0, 0, 0] no shape just a spot

    identify = np.outer(identify, identify)
    #adding more dimmestion made the lines thicker
    dx = np.outer([1, 0, 1], [1, 0,1]) / 8.0  # Sobel filter np.outer([2, 3, 2], [-2, 0,2]) / 8.0
    dy = dx.T
    c, s = tf.cos(angle), tf.sin(angle)
    kernel = tf.stack([identify, c*dx-s*dy, s*dx+c*dy], -1)[:, :, None, :]
    kernel = tf.repeat(kernel, self.channel_n, 2)
    #y_1 = tf.keras.layers.Conv2D(48,3)(x)
    #y_1 = tf.keras.layers.Conv2DTranspose(48, 3)(y_1)
    #print('x', x.shape)
    y = tf.nn.depthwise_conv2d(x, kernel, [1, 1, 1, 1], 'SAME')
    y_sum = self.global_attention(y)
    #print('kernel', kernel.shape)
    #print('y_sum',y_sum[0].shape)
    y = tf.nn.depthwise_conv2d(y, y_sum[0], [1, 1, 1, 1], 'SAME')

    y = tf.math.l2_normalize(y)
    return y #+ tfa.image.rotate(y,90) does nothing special #1,1,1,3


  def get_living_mask(self,x):
        alpha = x[:, :, :, 3:4]
        return tf.nn.max_pool2d(alpha, 3, [1, 1, 1, 1], 'SAME') > 0.1

  @tf.function
  def call(self, x, fire_rate=None, angle=0.0, step_size=1.0):
      global global_pool

      pre_life_mask = self.get_living_mask(x)
      #print('x',x.shape)
      y = self.perceive(x, angle)

      dx = self.dmodel(y) * step_size
      if fire_rate is None:
          fire_rate = self.fire_rate
      update_mask = tf.random.uniform(tf.shape(x[:, :, :, :1])) <= fire_rate
      #removing the + sign lead to no generation
      x += dx * tf.cast(update_mask, tf.float32)

      #x += global_pool.read()*0.1
      #global_pool.write(x)
      post_life_mask = self.get_living_mask(x)
      life_mask = pre_life_mask & post_life_mask
      return x * tf.cast(life_mask, tf.float32)
"""
class CAModel(tf.keras.Model):

  def __init__(self, channel_n=CHANNEL_N, fire_rate=CELL_FIRE_RATE):
    super().__init__()
    self.channel_n = channel_n
    self.fire_rate = fire_rate

    self.dmodel = tf.keras.Sequential([
          Conv2D(128, 1, activation=tf.nn.relu),
          Conv2D(self.channel_n, 1),
    ])

    self(tf.zeros([1, 3, 3, channel_n]),tf.zeros([1, 3, 3, channel_n*3]))  # dummy call to build the model

  @tf.function
  def perceive(self, x, angle=0.0):
    identify = np.float32([0, 1, 0])#[1, 0, 1]-strange shape [0, 0, 1] strange shape [0, 0, 0] no shape just a spot

    identify = np.outer(identify, identify)
    dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
    dy = dx.T
    c, s = tf.cos(angle), tf.sin(angle)
    kernel = tf.stack([identify, c*dx-s*dy, s*dx+c*dy], -1)[:, :, None, :]
    kernel = tf.repeat(kernel, self.channel_n, 2)
    y = tf.nn.depthwise_conv2d(x, kernel, [1, 1, 1, 1], 'SAME')
    return y
  def get_living_mask(self,x):
        alpha = x[:, :, :, 3:4]
        return tf.nn.max_pool2d(alpha, 3, [1, 1, 1, 1], 'SAME') > 0.1

  #@tf.function
  def call(self, x,y_init, fire_rate=None, angle=0.0, step_size=1):
      '''
      pre_life_mask = self.get_living_mask(x)

      y = self.perceive(x, angle)
      #y += y_init
      dx = self.dmodel(y)
      if fire_rate is None:
          fire_rate = self.fire_rate
      update_mask = tf.random.uniform(tf.shape(x[:, :, :, :1])) <= fire_rate

      #this fixed the problem with expoling values and improved the quality so much
      #dx = tf.math.l2_normalize(dx)#* step_size

      #is this okay for the model?
      #no it is not.With this we have only noise
      #x = tf.math.l2_normalize(x)

      #will removing the + sign improve the results?
      x += dx * tf.cast(update_mask, tf.float32)

      post_life_mask = self.get_living_mask(x)
      life_mask = pre_life_mask & post_life_mask
      #this was removed because it wont generate any data at the end
      return x,y #* tf.cast(life_mask, tf.float32)
      '''
      pre_life_mask = self.get_living_mask(x)

      y = self.perceive(x, angle)

      dx = self.dmodel(y) * step_size
      if fire_rate is None:
          fire_rate = self.fire_rate
      update_mask = tf.random.uniform(tf.shape(x[:, :, :, :1])) <= fire_rate
      x += dx * tf.cast(update_mask, tf.float32)

      post_life_mask = self.get_living_mask(x)
      life_mask = pre_life_mask & post_life_mask
      return x * tf.cast(life_mask, tf.float32)
"""
class NeuralCellularAutomata_moded(Agent):
    def load_image(self,url, max_size=TARGET_SIZE):
        r = requests.get(url)
        img = PIL.Image.open(io.BytesIO(r.content))
        img.thumbnail((max_size, max_size), PIL.Image.ANTIALIAS)
        img = np.float32(img) / 255.0
        # premultiply RGB by Alpha
        img[..., :3] *= img[..., 3:]

        return img

    def load_emoji(self,emoji):
        code = hex(ord(emoji))[2:].lower()
        url = 'https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u%s.png?raw=true' % code
        return self.load_image(url)

    def export_model(self,ca, base_fn):
        ca.save_weights(base_fn)

        cf = ca.call.get_concrete_function(
            x=tf.TensorSpec([None, None, None, CHANNEL_N]),

            #y_int=tf.TensorSpec([None, None, None, 48]),
            fire_rate=tf.constant(0.5),
            angle=tf.constant(0.0),
            step_size=tf.constant(1.0))
        cf = convert_to_constants.convert_variables_to_constants_v2(cf)
        graph_def = cf.graph.as_graph_def()
        graph_json = MessageToDict(graph_def)
        graph_json['versions'] = dict(producer='1.14', minConsumer='1.14')
        model_json = {
            'format': 'graph-model',
            'modelTopology': graph_json,
            'weightsManifest': [],
        }
        with open(base_fn + '.json', 'w') as f:
            json.dump(model_json, f)

    def generate_pool_figures(self,pool, step_i):
        tiled_pool = self.tile2d(self.to_rgb(pool.x[:49]))
        fade = np.linspace(1.0, 0.0, 72)
        ones = np.ones(72)
        tiled_pool[:, :72] += (-tiled_pool[:, :72] + ones[None, :, None]) * fade[None, :, None]
        tiled_pool[:, -72:] += (-tiled_pool[:, -72:] + ones[None, :, None]) * fade[None, ::-1, None]
        tiled_pool[:72, :] += (-tiled_pool[:72, :] + ones[:, None, None]) * fade[:, None, None]
        tiled_pool[-72:, :] += (-tiled_pool[-72:, :] + ones[:, None, None]) * fade[::-1, None, None]
        self.imwrite('./data_sets/cellularAutomata/train_log/%04d_pool.jpg' % step_i, tiled_pool)

    def visualize_batch(self,x0, x, step_i):
        vis0 = np.hstack(self.to_rgb(x0).numpy())
        vis1 = np.hstack(self.to_rgb(x).numpy())
        vis = np.vstack([vis0, vis1])
        self.imwrite('./data_sets/cellularAutomata/train_log/batches_%04d.jpg' % step_i, vis)
        print('batch (before/after):')
        self.imshow(vis)

    def to_rgba(self,x):
        #if len(x.numpy().shape) == 3:
        #    return x[:,:,:4]
        #if len(x.numpy().shape) == 4:
        #    return x[:,:,:,:4]
        return x[..., :4]
        #return x[:,:,:]#x[..., :4]

    def to_alpha(self,x):
        return tf.clip_by_value(x[..., 3:4], 0.0, 1.0)

    def to_rgb(self,x):
        # assume rgb premultiplied by alpha
        rgb, a = x[..., :3], self.to_alpha(x)
        return 1.0 - a + rgb


    def make_seed(self,size, n=1):
        x = np.zeros([n, size, size, CHANNEL_N], np.float32)
        x[:, size // 2, size // 2, 3:] = 1.0
        return x

    def np2pil(self,a):
        if a.dtype in [np.float32, np.float64]:
            a = np.uint8(np.clip(a, 0, 1) * 255)
        return PIL.Image.fromarray(a)

    def imwrite(self,f, a, fmt=None):
        a = np.asarray(a)
        if isinstance(f, str):
            fmt = f.rsplit('.', 1)[-1].lower()
            if fmt == 'jpg':
                fmt = 'jpeg'
            f = open(f, 'wb')
        self.np2pil(a).save(f, fmt, quality=95)

    def imencode(self,a, fmt='jpeg'):
        a = np.asarray(a)
        if len(a.shape) == 3 and a.shape[-1] == 4:
            fmt = 'png'
        f = io.BytesIO()
        self.imwrite(f, a, fmt)
        return f.getvalue()

    def im2url(self,a, fmt='jpeg'):
        encoded = self.imencode(a, fmt)
        base64_byte_string = base64.b64encode(encoded).decode('ascii')
        return 'data:image/' + fmt.upper() + ';base64,' + base64_byte_string

    def imshow(self,a, fmt='jpeg'):
        display(Image(data=self.imencode(a, fmt)))

    def tile2d(self,a, w=None):
        a = np.asarray(a)
        if w is None:
            w = int(np.ceil(np.sqrt(len(a))))
        th, tw = a.shape[1:3]
        pad = (w - len(a)) % w
        a = np.pad(a, [(0, pad)] + [(0, 0)] * (a.ndim - 1), 'constant')
        h = len(a) // w
        a = a.reshape([h, w] + list(a.shape[1:]))
        a = np.rollaxis(a, 2, 1).reshape([th * h, tw * w] + list(a.shape[4:]))
        return a

    def zoom(self,img, scale=4):
        img = np.repeat(img, scale, 0)
        img = np.repeat(img, scale, 1)
        return img
    @tf.function
    def loss_f(self,x):

        return tf.reduce_mean(tf.square(self.to_rgba(x) - self.pad_target), [-2, -3, -1])

    #def loss_f_second(self,x,pad_target):
    #    return tf.reduce_mean(tf.square(x - pad_target), [-2, -3, -1])
    def __init__(self, inputs, outputs, data_schema_input, data_schema_output, class_num):
        self.model = None
        self.func_map = {}
        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        # @title Cellular Automata Parameters

        target_img = self.load_emoji(TARGET_EMOJI)

        p = TARGET_PADDING
        self.pad_target = tf.pad(target_img, [(p, p), (p, p), (0, 0)])
        # self.pad_target (72, 72, 4)
        print("self.pad_target", self.pad_target.shape)
        self.h, self.w = self.pad_target.shape[:2]
        self.seed = np.zeros([self.h, self.w, CHANNEL_N], np.float32)
        self.seed[self.h // 2, self.w // 2, 3:] = 1.0

        print("self.seed", self.seed.shape)

        self.ca = CAModel()

        self.loss_log = []

        lr = 2e-3
        lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            [2000], [lr, lr * 0.1])
        self.trainer = tf.keras.optimizers.Adam(lr_sched)

        self.loss0 = self.loss_f(self.seed).numpy()
        self.pool = SamplePool(x=np.repeat(self.seed[None, ...], POOL_SIZE, 0))

        self.imshow(self.zoom(self.to_rgb(target_img), 2), fmt='png')
        self.init_neural_network(inputs, outputs, data_schema_input, latent_dim=(1), class_num=class_num)  # 8*4*3
        self.calc_map_plot_counter = 0
        self.confusion_matrix = {}
        self.func_arr = [[], []]
        conn1, conn2 = Pipe()
        self.conn_send = conn2
        # p = Process(target=runGraph, args=(conn1,))
        # p.start()

    def train_step(self,x):
        iter_n = tf.random.uniform([], 64, 96, tf.int32)
        with tf.GradientTape() as g:
            for i in tf.range(iter_n):
              x = self.ca(x)
            loss = tf.reduce_mean(self.loss_f(x))

        grads = g.gradient(loss, self.ca.weights)
        #print('grads',grads)
        #exit(0)
        grads = [g / (tf.norm(g) + 1e-8) for g in grads]
        self.trainer.apply_gradients(zip(grads, self.ca.weights))
        return x, loss
    """
    #@tf.function
    def train_step(self,x,y_int):
              iter_n = tf.random.uniform([], 64, 96, tf.int32)
              x_s = x[:]

              x=tf.expand_dims(x, axis=0)
              with tf.GradientTape() as g:
                  for i in tf.range(iter_n):
                      #1.1 if I go above this value for step_size i will ecounter OOM
                      #step_size=3 this will produce pure noise it is too high
                      # step_size=1.5 same as above is the model correct?
                      #at even 1 I get noise something is wrong with the model
                      #at step_size = 10 I get a lot of noise but still I can see the image
                      # but nothing about the target image

                      x = self.ca_arr(x)#,y_int,step_size=10)[self.current_img_id]

                  #plt.imshow(self.target_img)
                  #plt.show()
                  #print('\nloss_f',self.loss_f(x,self.target_img))

                  loss = tf.reduce_mean(self.loss_f(x,self.target_img))#(x,self.target_img)
                  #print(self.loss_f(x,self.target_img))

              grads = g.gradient(loss, self.ca_arr.weights)#[self.current_img_id]
              print('loss', loss)
              print('grads',grads)
              #print('self.ca_arr[self.current_img_id].weights',self.ca_arr[self.current_img_id].weights)
              for local_grads in grads:
                  print(local_grads.numpy().max())
              plt.subplot(1, 2, 1)
              plt.imshow(np.sum(x[0], axis=2))

              plt.subplot(1, 2, 2)
              plt.imshow(self.target_img * 255)  # , cmap=reversed_color_map)
              plt.show()
              grads = [g / (tf.norm(g) + 1e-8) for g in grads]#1e-8
              self.trainer.apply_gradients(zip(grads, self.ca_arr.weights))#[self.current_img_id]


              return x, loss,x
         """

    '''
    #@tf.function
    def train_step(self,x):
        iter_n = tf.random.uniform([], 64, 96, tf.int32)

        x,y_int,loss = tf.ones((8, 72, 72, 16)),tf.ones((8, 72, 72, 16)),tf.ones((2))
        with tf.GradientTape() as g:
            for i in tf.range(iter_n):
                x = self.ca_1(x,tf.ones((8, 72, 72, 16)))
            loss = tf.reduce_mean(self.loss_f(x,self.target_img_arr[0]))
            #print('x',x[0].shape)
            #print('self.target_img_arr[0]', self.target_img_arr[0].shape)
        grads = g.gradient(loss, self.ca_1.weights)
        grads = [g / (tf.norm(g) + 1e-8) for g in grads]
        self.trainer.apply_gradients(zip(grads, self.ca_1.weights))
        return x, loss
    '''
    def init_neural_network(self, inputs, outputs, data_schema, latent_dim, class_num):
        pass

    '''
    def prepare_data(self, images, in_train=False):
        local_x_train_arr = []
        local_y_train_arr = []
        local_target_train_arr = []
        images_collection = {}
        for image in images:

            local_image = image.get_by_name('Image')

            local_id = image.get_by_name('Id')

            # plt.imshow(local_image)
            # plt.show()
            if local_id[:8] not in images_collection.keys():
                images_collection[local_id[:8]] = {'input': None, 'output': None}
            if 'input' in local_id:
                images_collection[local_id[:8]]['input'] = local_image
            elif 'output' in local_id:
                images_collection[local_id[:8]]['output'] = local_image
        for local_key in images_collection.keys():
            local_image_input = images_collection[local_key]['input']
            if images_collection[local_key]['output'] is None or images_collection[local_key]['input'] is None:
                continue
            if random.choice([True, False]):
                local_target_train_arr.append('rot')
                # print(images_collection[local_key]['output'])

                # print(images_collection[local_key]['input'])
                images_collection[local_key]['output'] = np.rot90(images_collection[local_key]['output'])
            else:
                local_target_train_arr.append('none')
            local_image_output = images_collection[local_key]['output']
            if type(local_image_output) == type(None) or type(local_image_input) == type(None):
                continue
            im = local_image_input.astype('float32')
            tiles_M = im.shape[0] // DIVISION
            tiles_N = im.shape[1] // DIVISION
            local_image_input = [im[x:x + tiles_M, y:y + tiles_N] for x in range(0, im.shape[0], tiles_M) for y in
                                 range(0, im.shape[1], tiles_N)]
            im = local_image_output.astype('float32')
            tiles_M = im.shape[0] // DIVISION
            tiles_N = im.shape[1] // DIVISION
            local_image_output = [im[x:x + tiles_M, y:y + tiles_N] for x in range(0, im.shape[0], tiles_M) for y in
                                  range(0, im.shape[1], tiles_N)]

            local_x_train_arr.append(local_image_input)  # self.contur_image(local_image)
            local_y_train_arr.append(local_image_output)

        return np.array(local_x_train_arr), np.array(local_y_train_arr)
    
    def prepare_data(self, images, in_train=False):
        local_x_train_arr = []
        local_y_train_arr = []
        local_y_original = []
        lens = []
        local_target_train_arr = []
        images_collection = {}
        for image in images:

            local_image = image.get_by_name('Image')
            

            local_image_mask = image.get_by_name('ImageMask')


            if local_image_mask is not None and local_image is not None:
                im = local_image.astype('float32')

                tiles_M = im.shape[0] // DIVISION
                tiles_N = im.shape[1] // DIVISION
                local_image_input = [im[x:x + tiles_M, y:y + tiles_N] for x in range(0, im.shape[0], tiles_M) for y in
                                     range(0, im.shape[1], tiles_N)]
                local_y_original.append(local_image_mask)
                im = local_image_mask.astype('float32')
                tiles_M = im.shape[0] // DIVISION
                tiles_N = im.shape[1] // DIVISION
                local_image_output = [im[x:x + tiles_M, y:y + tiles_N] for x in range(0, im.shape[0], tiles_M) for y in
                                      range(0, im.shape[1], tiles_N)]
                local_x_train_arr.append(local_image_input)
                local_y_train_arr.append(local_image_output)
                lens+=[len(x) for x in local_image_input]



        return np.array(local_x_train_arr), np.array(local_y_train_arr),None,np.array(local_y_original)
        '''


    def compute_loss(self, model, x, y, func_name='', is_plot=False):

        pass
    def save(self):
        ckpt_name = 'default_cpkt_name'
        re_result = re.search(r'.*\.(\w*)', str(self.__class__), re.S | re.U | re.I)
        if re_result:
            ckpt_name = re_result.group(1)
        try:
         for i ,element in enumerate(self.ca_arr):
            element.save_weights('./checkpoints/'+ckpt_name+str(i)+'/')
        except Exception as e:
            pass
    def load(self,force_train):
        ca_arr_new = []
        re_result = re.search(r'.*\.(\w*)', str(self.__class__), re.S | re.U | re.I)
        if re_result:
            ckpt_name = re_result.group(1)
        for i in range(100):
            if Path('./checkpoints/' + ckpt_name+str(i)).exists() and not force_train:
             self.ca_arr[i].load_weights('./checkpoints/' + ckpt_name+str(i)+'/')
             ca_arr_new.append(1)
        if len(ca_arr_new) >0:

            return True
        else:
            return False
    def train(self, images, force_train=False):
        global last_img
        local_x_train_arr, local_y_train_arr,local_y_original= prepare_data(images,in_train=True)
        for i in range(2500 + 1):
            for j in range(DIVISION**2):

                if USE_PATTERN_POOL:
                    batch = self.pool.sample(BATCH_SIZE)
                    x0 = batch.x
                    loss_rank = self.loss_f(x0).numpy().argsort()[::-1]
                    x0 = x0[loss_rank]
                    x0[:1] = self.seed
                    if local_y_train_arr[0][j].max() == 0:
                        continue

                    #local_y_train_arr[0][j] = np.random.randint(0,255,(64,64,3)).astype(np.float32)
                    self.pad_target = tf.convert_to_tensor( np.pad(local_y_train_arr[0][j]*255,((0,0),(0,0),(0,1)),mode='maximum') )
                    #self.pad_target = tf.convert_to_tensor(np.pad(self.pad_target,((0,(3)),(0,(3)),(0,0)),mode='constant'))
                    #self.pad_target = tf.convert_to_tensor(self.pad_target[:64,:64,:])
                    x1 = []
                    x1.append(local_x_train_arr[0][j])

                    #if np.array(local_x_train_arr[n][j]).shape != (64, 64, 16) :
                    #     continue
                    #x1.append(local_x_train_arr[n][j])
                    x1 = [np.pad(x1[0],((0,0),(0,0),(0,CHANNEL_N-3)),mode='edge')/255.0]*1
                    #print(np.array(x1).shape != (1,64,64,16))
                    print(np.array(x1).shape)
                    if np.array(x1).shape != (1,64,64,8):#CHANNEL_N+3):
                        continue

                    x0 = tf.convert_to_tensor(x1)#np.array([x1/255.0]*BATCH_SIZE))
                    if DAMAGE_N:
                        damage = 1.0 - make_circle_masks(DAMAGE_N, self.h, self.w).numpy()[..., None]
                        x0[-DAMAGE_N:] *= damage
                else:
                    x0 = np.repeat(self.seed[None, ...], BATCH_SIZE, 0)


                x, loss = self.train_step(x0)
                #if USE_PATTERN_POOL:
                #    batch.x[:] = x
                #    batch.commit()

                step_i = len(self.loss_log)
                self.loss_log.append(loss.numpy())

                if step_i % 10 == 0:
                    self.generate_pool_figures(self.pool, step_i)
                if step_i % 100 == 0:
                    clear_output()
                    self.visualize_batch(x0, x, step_i)
                    #plot_loss(self.loss_log)
                    self.export_model(self.ca, './data_sets/cellularAutomata/train_log/%04d' % step_i)

                print('\r step: %d, log10(loss): %.3f' % (len(self.loss_log), np.log10(loss)), end='')
        self.show()
    def show(self):
        # @title Training Progress (Checkpoints)

        models = []
        try:
         for i in [100, 500, 1000, 4000]:
            ca = CAModel()
            ca.load_weights('./data_sets/cellularAutomata/train_log/%04d' % i)
            models.append(ca)
        except Exception as e:
            pass
        out_fn = 'train_steps_damage_%d.mp4' % DAMAGE_N
        x = np.zeros([len(models), 64, 64, CHANNEL_N], np.float32)
        x[..., 36, 36, 3:] = 1.0

        with VideoWriter(out_fn) as vid:
            for i in tqdm.trange(5000):
                vis = np.hstack(self.to_rgb(x))
                vid.add(self.zoom(vis, 2))
                for ca, xk in zip(models, x):
                    x_output = self.ca(xk[None, ...])
                    #x_output,_ = self.ca_2(xk[None, ...],y_int)
                    xk[:] = x_output[0]
        mvp.ipython_display(out_fn)
        # @title Training Progress (Batches)
        frames = sorted(glob.glob('./data_sets/cellularAutomata/train_log/batches_*.jpg'))
        mvp.ImageSequenceClip(frames, fps=10.0).write_videofile('batches.mp4')
        mvp.ipython_display('batches.mp4')
        # @title Pool Contents
        frames = sorted(glob.glob('./data_sets/cellularAutomata/train_log/*_pool.jpg'))[:80]
        mvp.ImageSequenceClip(frames, fps=20.0).write_videofile('pool.mp4')
        mvp.ipython_display('pool.mp4')

        atlas = np.hstack([self.load_emoji(e) for e in EMOJI])
        self.imshow(atlas)

    def get_model(self,emoji='🦋', fire_rate=0.5, use_pool=1, damage_n=3, run=0,
                  prefix='models/', output='model'):
        path = prefix
        assert fire_rate in [0.5, 1.0]
        if fire_rate == 0.5:
            path += 'use_sample_pool_%d damage_n_%d ' % (use_pool, damage_n)
        elif fire_rate == 1.0:
            path += 'fire_rate_1.0 '
        code = hex(ord(emoji))[2:].upper()
        path += 'target_emoji_%s run_index_%d/08000' % (code, run)
        assert output in ['model', 'json']
        if output == 'model':
            ca = CAModel(channel_n=16, fire_rate=fire_rate)
            ca.load_weights(prefix)
            return ca
        elif output == 'json':
            return open(path + '.json', 'r').read()
    def teaser(self):
        # @title Teaser
        models = [self.get_model(emoji, run=1) for emoji in EMOJI]

        with VideoWriter('teaser.mp4') as vid:
            x = np.zeros([len(EMOJI), 64, 64, CHANNEL_N], np.float32)
            # grow
            for i in tqdm.trange(200):
                k = i // 20
                if i % 20 == 0 and k < len(EMOJI):
                    x[k, 32, 32, 3:] = 1.0
                vid.add(self.zoom(self.tile2d(self.to_rgb(x), 5), 2))
                for ca, xk in zip(models, x):
                    xk[:] = self.ca(xk[None, ...])[0]
            # damage
            mask = PIL.Image.new('L', (64 * 5, 64 * 2))
            draw = PIL.ImageDraw.Draw(mask)
            for i in tqdm.trange(400):
                cx, r = i * 3 - 20, 6
                y1, y2 = 32 + np.sin(i / 5 + np.pi) * 8, 32 + 64 + np.sin(i / 5) * 8
                draw.rectangle((0, 0, 64 * 5, 64 * 2), fill=0)
                draw.ellipse((cx - r, y1 - r, cx + r, y1 + r), fill=255)
                draw.ellipse((cx - r, y2 - r, cx + r, y2 + r), fill=255)
                x *= 1.0 - (np.float32(mask).reshape(2, 64, 5, 64)
                            .transpose([0, 2, 1, 3]).reshape(10, 64, 64, 1)) / 255.0
                if i < 200 or i % 2 == 0:
                    vid.add(self.zoom(self.tile2d(self.to_rgb(x), 5), 2))
                for ca, xk in zip(models, x):
                    xk[:] = self.ca(xk[None, ...])[0]
            # fade out
            last = self.zoom(self.tile2d(self.to_rgb(x), 5), 2)
            for t in np.linspace(0, 1, 30):
                vid.add(last * (1.0 - t) + t)

        mvp.ipython_display('teaser.mp4', loop=True)

    def unstable_patterns(self):
        font_fn = self.fm.findfont(self.fm.FontProperties())
        font = PIL.ImageFont.truetype(font_fn, 20)

        models = [self.get_model(ch, use_pool=0, damage_n=0) for ch in EMOJI]
        fn = 'unstable.mp4'
        with VideoWriter(fn) as vid:
            x = np.zeros([len(EMOJI), 64, 64, CHANNEL_N], np.float32)
            x[:, 32, 32, 3:] = 1.0
            # grow
            slider = PIL.Image.open("slider.png")
            for i in tqdm.trange(1000):
                if i < 200 or i % 5 == 0:
                    vis = self.zoom(self.tile2d(self.to_rgb(x), 5), 4).clip(0, 1)
                    vis_extended = np.concatenate((vis, np.ones((164, vis.shape[1], 3))), axis=0)
                    im = np.uint8(vis_extended * 255)
                    im = PIL.Image.fromarray(im)
                    im.paste(slider, box=(20, vis.shape[0] + 20))
                    draw = PIL.ImageDraw.Draw(im)
                    p_x = (14 + (610 / 1000) * i) * 2.0
                    draw.rectangle([p_x, vis.shape[0] + 20 + 55, p_x + 10, vis.shape[0] + 20 + 82], fill="#434343bd")
                    vid.add(np.uint8(im))
                for ca, xk in zip(models, x):
                    xk[:] = self.ca(xk[None, ...])[0]
            # fade out
            for t in np.linspace(0, 1, 30):
                vid.add(vis_extended * (1.0 - t) + t)

        mvp.ipython_display(fn, loop=True)
    def rotation(self):
        # @title Rotation
        row_size = 4
        models_of_interest = ["🦋", "🦎", "🐠", "😀"]
        num_images = 16
        imgs = []
        start_angle = np.random.randint(13, 76)

        for i in np.arange(num_images):
            ang = start_angle + i * np.random.randint(36, 111)
            ang = ang / 360.0 * 2 * np.pi
            if i % row_size == 0:
                ca = self.get_model(models_of_interest[i // row_size])
            x = np.zeros([1, 64, 64, CHANNEL_N], np.float32)
            x[:, 28, 28, 3:] = 1.0
            for i in range(500):
                ang = tf.constant(ang, tf.float32)
                x,y_int = self.ca(x, angle=ang)
            imgs.append(self.to_rgb(x)[0])
        # Assumes the result is a multiple of row_size
        assert len(imgs) % row_size == 0
        imgs = zip(*(iter(imgs),) * row_size)

        imgs_arr = np.concatenate([np.hstack(im_row) for im_row in imgs])
        vis = self.zoom(imgs_arr, 4)

        self.imshow(vis, fmt='png')
    def regeneration(self):
        models = [self.get_model(ch, damage_n=0) for ch in '😀🦋🦎']
        with VideoWriter('regen1.mp4') as vid:
            x = np.zeros([len(models), 5, 64, 64, CHANNEL_N], np.float32)
            cx, cy = 28, 28
            x[:, :, cy, cx, 3:] = 1.0
            for i in tqdm.trange(2000):
                if i == 200:
                    x[:, 0, cy:] = x[:, 1, :cy] = 0
                    x[:, 2, :, cx:] = x[:, 3, :, :cx] = 0
                    x[:, 4, cy - 8:cy + 8, cx - 8:cx + 8] = 0
                vis = self.to_rgb(x)
                vis = np.vstack([np.hstack(row) for row in vis])
                vis = self.zoom(vis, 2)
                if (i < 400 and i % 2 == 0) or i % 8 == 0:
                    vid.add(vis)
                if i == 200:
                    for _ in range(29):
                        vid.add(vis)
                for ca, row in zip(models, x):
                    row[:] = self.ca(row)

        mvp.ipython_display('regen1.mp4')

    def regeneration_dmg(self):
        # @title Regeneration (trained with damage)
        models = [self.get_model(ch, damage_n=3) for ch in '😀🦋🦎']
        with VideoWriter('regen2.mp4') as vid:
            x = np.zeros([len(models), 5, 64, 64, CHANNEL_N], np.float32)
            cx, cy = 28, 28
            x[:, :, cy, cx, 3:] = 1.0
            for i in tqdm.trange(2000):
                if i == 200:
                    x[:, 0, cy:] = x[:, 1, :cy] = 0
                    x[:, 2, :, cx:] = x[:, 3, :, :cx] = 0
                    x[:, 4, cy - 8:cy + 8, cx - 8:cx + 8] = 0
                vis = self.to_rgb(x)
                vis = np.vstack([np.hstack(row) for row in vis])
                vis = self.zoom(vis, 2)
                if (i < 400 and i % 2 == 0) or i % 8 == 0:
                    vid.add(vis)
                if i == 200:
                    for _ in range(29):
                        vid.add(vis)
                for ca, row in zip(models, x):
                    row[:] = self.ca(row)

        mvp.ipython_display('regen2.mp4')
    def planarian(self):
        ca = CAModel()
        ca.load_weights('./data_sets/cellularAutomata/planarian/train_log/8000')

        x = np.zeros([1, 64, 96, CHANNEL_N], np.float32)
        x[:, 32, 48, 3:] = 1.0
        with VideoWriter('planarian.mp4', 30.0) as vid:
            for i in range(400):
                vid.add(self.zoom(self.to_rgb(x[0])))
                x = self.ca(x, angle=np.pi / 2.0)
                if i == 150:
                    x = x.numpy()
                    for k in range(24):
                        x[:, :24] = np.roll(x[:, :24], 1, 2)
                        x[:, -24:] = np.roll(x[:, -24:], -1, 2)
                        vid.add(self.zoom(self.to_rgb(x[0])))
                    for k in range(20):
                        vid.add(self.zoom(self.to_rgb(x[0])))

        mvp.ipython_display('planarian.mp4')
    def interactive_demos(self):
        model = "CHECKPOINT"  # @param ['CHECKPOINT', '😀 1F600', '💥 1F4A5', '👁 1F441', '🦎 1F98E', '🐠 1F420', '🦋 1F98B', '🐞 1F41E', '🕸 1F578', '🥨 1F968', '🎄 1F384']
        model_type = '3 regenerating'  # @param ['1 naive', '2 persistent', '3 regenerating']

        # @markdown Shift-click to seed the pattern

        if model != 'CHECKPOINT':
            code = model.split(' ')[1]
            emoji = chr(int(code, 16))
            experiment_i = int(model_type.split()[0]) - 1
            use_pool = (0, 1, 1)[experiment_i]
            damage_n = (0, 0, 3)[experiment_i]
            model_str = self.get_model(emoji, use_pool=use_pool, damage_n=damage_n, output='json')
        else:
            last_checkpoint_fn = sorted(glob.glob('./data_sets/cellularAutomata/train_log/*.json'))[-1]
            model_str = open(last_checkpoint_fn).read()

        data_js = '''
          window.GRAPH_URL = URL.createObjectURL(new Blob([`%s`], {type: 'application/json'}));
        ''' % (model_str)

        display(self.IPython.display.Javascript(data_js))

        self.IPython.display.HTML('''
        <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.3.0/dist/tf.min.js"></script>

        <canvas id='canvas' style="border: 1px solid black; image-rendering: pixelated;"></canvas>

        <script>
          "use strict";

          const sleep = (ms)=>new Promise(resolve => setTimeout(resolve, ms));

          const parseConsts = model_graph=>{
            const dtypes = {'DT_INT32':['int32', 'intVal', Int32Array],
                            'DT_FLOAT':['float32', 'floatVal', Float32Array]};

            const consts = {};
            model_graph.modelTopology.node.filter(n=>n.op=='Const').forEach((node=>{
              const v = node.attr.value.tensor;
              const [dtype, field, arrayType] = dtypes[v.dtype];
              if (!v.tensorShape.dim) {
                consts[node.name] = [tf.scalar(v[field][0], dtype)];
              } else {
                // if there is a 0-length dimension, the exported graph json lacks "size"
                const shape = v.tensorShape.dim.map(d=>(!d.size) ? 0 : parseInt(d.size));
                let arr;
                if (v.tensorContent) {
                  const data = atob(v.tensorContent);
                  const buf = new Uint8Array(data.length);
                  for (var i=0; i<data.length; ++i) {
                    buf[i] = data.charCodeAt(i);
                  }
                  arr = new arrayType(buf.buffer);
                } else {
                  const size = shape.reduce((a, b)=>a*b);
                  arr = new arrayType(size);
                  if (size) {
                    arr.fill(v[field][0]);
                  }
                }
                consts[node.name] = [tf.tensor(arr, shape, dtype)];
              }
            }));
            return consts;
          }

          const run = async ()=>{
            const r = await fetch(GRAPH_URL);
            const consts = parseConsts(await r.json());

            const model = await tf.loadGraphModel(GRAPH_URL);
            Object.assign(model.weights, consts);

            let seed = new Array(16).fill(0).map((x, i)=>i<3?0:1);
            seed = tf.tensor(seed, [1, 1, 1, 16]);

            const D = 96;
            const initState = tf.tidy(()=>{
              const D2 = D/2;
              const a = seed.pad([[0, 0], [D2-1, D2], [D2-1, D2], [0,0]]);
              return a;
            });

            const state = tf.variable(initState);
            const [_, h, w, ch] = state.shape;

            const damage = (x, y, r)=>{
              tf.tidy(()=>{
                const rx = tf.range(0, w).sub(x).div(r).square().expandDims(0);
                const ry = tf.range(0, h).sub(y).div(r).square().expandDims(1);
                const mask = rx.add(ry).greater(1.0).expandDims(2);
                state.assign(state.mul(mask));
              });
            }

            const plantSeed = (x, y)=>{
              const x2 = w-x-seed.shape[2];
              const y2 = h-y-seed.shape[1];
              if (x<0 || x2<0 || y2<0 || y2<0)
                return;
              tf.tidy(()=>{
                const a = seed.pad([[0, 0], [y, y2], [x, x2], [0,0]]);
                state.assign(state.add(a));
              });
            }

            const scale = 4;

            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = w;
            canvas.height = h;
            canvas.style.width = `${w*scale}px`;
            canvas.style.height = `${h*scale}px`;

            canvas.onmousedown = e=>{
              const x = Math.floor(e.clientX/scale);
                const y = Math.floor(e.clientY/scale);
                if (e.buttons == 1) {
                  if (e.shiftKey) {
                    plantSeed(x, y);  
                  } else {
                    damage(x, y, 8);
                  }
                }
            }
            canvas.onmousemove = e=>{
              const x = Math.floor(e.clientX/scale);
              const y = Math.floor(e.clientY/scale);
              if (e.buttons == 1 && !e.shiftKey) {
                damage(x, y, 8);
              }
            }

            function step() {
              tf.tidy(()=>{
                state.assign(model.execute(
                    {x:state, fire_rate:tf.tensor(0.5),
                    angle:tf.tensor(0.0), step_size:tf.tensor(1.0)}, ['Identity']));
              });
            }

            function render() {
              step();

              const imageData = tf.tidy(()=>{
                const rgba = state.slice([0, 0, 0, 0], [-1, -1, -1, 4]);
                const a = state.slice([0, 0, 0, 3], [-1, -1, -1, 1]);
                const img = tf.tensor(1.0).sub(a).add(rgba).mul(255);
                const rgbaBytes = new Uint8ClampedArray(img.dataSync());
                return new ImageData(rgbaBytes, w, h);
              });
              ctx.putImageData(imageData, 0, 0);

              requestAnimationFrame(render);
            }
            render();
          }
          run();

        </script>
        ''')

    def pack_layer(self,weight, bias, outputType=np.uint8):
        in_ch, out_ch = weight.shape
        assert (in_ch % 4 == 0) and (out_ch % 4 == 0) and (bias.shape == (out_ch,))
        weight_scale, bias_scale = 1.0, 1.0
        if outputType == np.uint8:
            weight_scale = 2.0 * np.abs(weight).max()
            bias_scale = 2.0 * np.abs(bias).max()
            weight = np.round((weight / weight_scale + 0.5) * 255)
            bias = np.round((bias / bias_scale + 0.5) * 255)
        packed = np.vstack([weight, bias[None, ...]])
        packed = packed.reshape(in_ch + 1, out_ch // 4, 4)
        packed = outputType(packed)
        packed_b64 = base64.b64encode(packed.tobytes()).decode('ascii')
        return {'data_b64': packed_b64, 'in_ch': in_ch, 'out_ch': out_ch,
                'weight_scale': weight_scale, 'bias_scale': bias_scale,
                'type': outputType.__name__}

    def export_ca_to_webgl_demo(self,ca, outputType=np.uint8):
        # reorder the first layer inputs to meet webgl demo perception layout
        chn = ca.channel_n
        w1 = ca.weights[0][0, 0].numpy()
        w1 = w1.reshape(chn, 3, -1).transpose(1, 0, 2).reshape(3 * chn, -1)
        layers = [
            self.pack_layer(w1, ca.weights[1].numpy(), outputType),
            self.pack_layer(ca.weights[2][0, 0].numpy(), ca.weights[3].numpy(), outputType)
        ]
        return json.dumps(layers)
    def webGL_demo(self):
        with zipfile.ZipFile('webgl_models8.zip', 'w') as zf:
            for e in EMOJI:
                zf.writestr('ex1_%s.json' % e, self.export_ca_to_webgl_demo(self.get_model(e, use_pool=0, damage_n=0)))
                run = 1 if e in '😀🕸' else 0  # select runs that happen to quantize better
                zf.writestr('ex2_%s.json' % e, self.export_ca_to_webgl_demo(self.get_model(e, use_pool=1, damage_n=0, run=run)))
                run = 1 if e in '🦎' else 0  # select runs that happen to quantize better
                zf.writestr('ex3_%s.json' % e, self.export_ca_to_webgl_demo(self.get_model(e, use_pool=1, damage_n=3, run=run)))
    def predict(self, image):
        global last_img
        print('predict')
        x_arr_b,y_arr_b,len_mode, local_y_original = prepare_data(image)
        x_arr_b= x_arr_b[:1]

        y_arr_b= y_arr_b[:1]
        for x_sub_arr,y_sub_arr,y_orig  in zip(x_arr_b,y_arr_b,local_y_original):
            x_arr = x_sub_arr#np.ravel(x_sub_arr)
            y_arr = y_sub_arr#np.ravel(y_sub_arr)
            x_end_arr = []
            y_init_arr = [tf.zeros((1, 225, 225, 48))] * (DIVISION ** 2)
            y_init_arr_indx = 0
            y_int = tf.zeros((1, 225, 225, 16))
            for n,x_image,y_image in zip(range(len(x_arr)),x_arr,y_arr):
                #plt.imshow(x_image/255)
                #plt.show()
                if len(x_image) != len_mode or len(y_image) != len_mode:
                    continue

                self.target_img = y_image
                #x_image = np.ones(x_image.shape,dtype=np.float32)
                self.h, self.w = x_image.shape[:2]
                #self.current_img_id += 1
                #if self.current_img_id >= DIVISION**2:
                #    self.current_img_id = 0
                #plt.imshow(x_image/255.0)
                #print(x_image)
                #print(x_image.max())
                #plt.show()
                x_image = tf.pad(x_image, [[0, 0], [0, 0], [0, 13]], "CONSTANT")
                x = np.expand_dims(x_image/255.0 , axis=0)


                print(n,'/',len(x_arr))
                print('y_init_arr_indx',y_init_arr_indx,len(y_init_arr))
                #print('self.ca_arr',self.current_img_id,len(self.ca_arr))
                #for i in range(208 + 1):
                #    x,y_init = self.ca_arr[self.current_img_id](x,y_init_arr[y_init_arr_indx],step_size=10.0)
                    #print(np.expand_dims(np.sum(x[0],axis=2),axis=2).max())
                    #plt.imshow( np.expand_dims(np.sum(x[0],axis=2),axis=2))
                    #plt.show()
                #    plt_img = (np.array(x) / np.array(x).max()) * 255



                # x_end_arr.append((np.array(x) / np.array(x).max()))
                #y_init_arr_indx+=1
                #if y_init_arr_indx >= DIVISION ** 2:
                #    y_init_arr_indx = 0
                #y_init_arr[y_init_arr_indx] = y_init


            final_img = []

            try:
                for i in range(DIVISION):
                    final_img.append([])
                    for j in range(DIVISION):
                        local_img = np.sum(x_end_arr[i+j*DIVISION][0],axis=2)
                        final_img[i].append(local_img)
                    final_img[i]= np.concatenate(final_img[i])

                final_img = np.concatenate(final_img,axis=1)
                final_img = (final_img / final_img.max()) * 255
                print("final_img.shape",final_img.shape)
                plt.subplot(1, 2, 1)
                plt.imshow(final_img)


                print(np.array(y_orig).shape)
                print(y_orig.max())
                color_map = plt.cm.get_cmap()

                plt.subplot(1, 2, 2)
                plt.imshow(y_orig*255)#, cmap=reversed_color_map)
                plt.show()
            except Exception as e:
                print('e',e)
                pass
        return None

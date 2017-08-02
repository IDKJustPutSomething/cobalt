import tensorflow as tf
import numpy as np
from six.moves import xrange
import collections
import math
import os
import random
import zipfile

try:
  # pylint: disable=g-import-not-at-top
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

except ImportError:
    print('Please install sklearn, matplotlib, and scipy to show embeddings.')

class W2VModel:
    
    def __init__(self):
        #Vocabulary Settings
        self.vocabulary = None
        self.vocabulary_size = 50000
        self.embedding_size = 128
        
        #Batch Settings
        self.batch_size = 128
        self.num_skips = 2
        self.skip_window = 1
        
        #Validation Settings
        self.valid_size = 16               # Random set of words to evaluate similarity on.
        self.valid_window = 100            # Only pick dev samples in the head of the distribution.
        self.valid_examples = np.random.choice(self.valid_window, self.valid_size, replace=False)
        self.num_sampled = 64              # Number of negative examples to sample.
        
        #Training Settings
        self.device = '/cpu:0'
        self.learning_rate = 1
        
        #Misc Settings/Variables
        self.data_index = 0
        self.device = '/cpu:0'
    
    def read_data(self, path):
        with open(path) as f:
            vocab = f.read().lower()
            vocab = vocab.replace('.', '').replace(',', '').replace('\'s', '').replace('\'', '').replace('"', '')
            vocab = vocab.replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace('{', '').replace('}', '')
            self.vocabulary = vocab.split()
            print('Data size', len(self.vocabulary))
            
    def build_dataset(self):
        assert self.vocabulary is not None, 'No corpus has been loaded or failed to load.'
        self.count = [['UNK', -1]]
        self.count.extend(collections.Counter(self.vocabulary).most_common(self.vocabulary_size - 1))
        self.dictionary = dict()
        for word, _ in self.count:
            self.dictionary[word] = len(self.dictionary)
        
        self.data = list()
        unk_count = 0
        for word in self.vocabulary:
            if word in self.dictionary:
                index = self.dictionary[word]
            else:
                index = 0
                unk_count += 1
            self.data.append(index)
        self.count[0][1] = unk_count
        self.reversed_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))
        del self.vocabulary
        
    def generate_batch(self):
        assert self.batch_size % self.num_skips == 0, 'Error generating batch.'
        assert self.num_skips <= 2 * self.skip_window, 'Invalid num_skips to skip_window ratio.'
        self.batch = np.ndarray(shape=(self.batch_size), dtype=np.int32)
        self.labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        span = 2 * self.skip_window + 1
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data)
        for i in range(self.batch_size // self.num_skips):
            target = self.skip_window
            targets_to_avoid = [self.skip_window]
            for j in range(self.num_skips):
                while(target in targets_to_avoid):
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                self.batch[i * self.num_skips + j] = buffer[self.skip_window]
                self.labels[i * self.num_skips + j, 0] = buffer[target]
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data)
        # Backtrack a little bit to avoid skipping words in the end of a batch
        self.data_index = (self.data_index + len(self.data) - span) % len(self.data)
    
    def create_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
            valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)
            
            with tf.device(self.device):
                embeddings = tf.Variable(
                    tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
                embed = tf.nn.embedding_lookup(embeddings, self.train_inputs)
                
                nce_weights = tf.Variable(
                    tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                        stddev=1.0 / math.sqrt(self.embedding_size)))
                nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))
            
            
            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights,
                        biases=nce_biases,
                        labels=self.train_labels,
                        inputs=embed,
                        num_sampled=self.num_sampled,
                        num_classes=self.vocabulary_size))
            
            # Construct the SGD optimizer using a learning rate of 1.0.
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
                            
            # Compute the cosine similarity between minibatch examples and all embeddings.
            
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))

            self.normalized_embeddings = embeddings / norm
            self.valid_embeddings = tf.nn.embedding_lookup(
                self.normalized_embeddings, valid_dataset)
            self.similarity = tf.matmul(
                self.valid_embeddings, self.normalized_embeddings, transpose_b=True)
                
            # Add variable initializer.
            self.init = tf.global_variables_initializer()
            self.session = tf.Session(graph=self.graph)
            self.saver = tf.train.Saver(max_to_keep=4)
            
    def train(self, num_steps = 100001):
        self.session.run(self.init)
        print('Initialized W2V variables.')
            
        average_loss = 0
        for step in xrange(num_steps):
            self.generate_batch()
            feed_dict = {self.train_inputs: self.batch, self.train_labels: self.labels}
            
            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            _, loss_val = self.session.run([self.optimizer, self.loss], feed_dict=feed_dict)
            average_loss += loss_val
            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0
         
            # Note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 10000 == 0:
                if step is not 0:
                    self.save(step)
        self.final_embeddings = self.normalized_embeddings.eval(session=self.session)
        self.save(step)
    
    def plot_with_labels(self, filename='w2v.png'):
        
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        plot_only = 500
        low_dim_embs = tsne.fit_transform(self.final_embeddings[:plot_only, :])
        labels = [self.reversed_dictionary[i] for i in xrange(plot_only)]
        
        assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
        plt.figure(figsize=(18, 18))  # in inches
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(label,
                xy=(x, y),
                xytext=(5, 2),
                textcoords='offset points',
                ha='right',
                va='bottom')

        plt.savefig(filename)
        
    def new(self, pathToCorpus, steps):
        self.read_data(path=pathToCorpus)
        self.build_dataset()
        self.generate_batch()
        self.create_graph()
        self.train(steps)
        
    def save(self, step):
        print('Saving')
        try:
            self.saver.save(self.session, 'w2v/model', global_step=step)
            print('Saved')
        except ValueError:
            print('Failed To Save')

    def load(self, pathToCorpus):
        self.read_data(path=pathToCorpus)
        self.build_dataset()
        self.generate_batch()
        self.create_graph()
        name = tf.train.latest_checkpoint('w2v/')
        print(name)
        self.saver = tf.train.import_meta_graph(name + '.meta')
        self.saver.restore(self.session, name)
        self.final_embeddings = self.normalized_embeddings.eval(session=self.session)
        
    def get_nearest(self, word, number=1):
        i = self.dictionary.get(word)
        sim = self.similarity.eval(session=self.session)
        valid_word = self.reversed_dictionary[i]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in xrange(top_k):
            close_word = self.reversed_dictionary.get(nearest[k], 'UNK')
            log_str = '%s %s,' % (log_str, close_word)
        print(log_str)
        
        
        
        
        
        
        
                
                
                
                
                
                
                

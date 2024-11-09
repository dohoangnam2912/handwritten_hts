import os
import sys
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from dataloader_iam import Batch

tf.compat.v1.disable_eager_execution()

class DecoderType:
    """
    CTC decoder type
    """
    BestPath = 0
    BeamSearch = 1
    WordBeamSearch = 2

class Model:
    def __init__(self,
                char_list: List[str],
                decoder_type: str = DecoderType.BestPath,
                must_restore: bool = False,
                dump: bool = False) -> None:
        """
        Init model: Add CNN, RNN, CTC and initialize TF.
        """
        self.dump = dump
        self.char_list = char_list
        self.decoder_type = decoder_type
        self.must_restore = must_restore
        self.snap_ID = 0

        # Whether to use normalization over a batch or a population
        self.is_train = tf.compat.v1.placeholder(tf.bool, name='is_train')

        # Input image batch
        self.input_imgs = tf.compat.v1.placeholder(tf.float32, shape=(None, None, None))

        self.setup_cnn()
        self.setup_rnn()
        self.setup_ctc()

        # Set up optimizer to train NN
        self.batches_trained = 0
        self.update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.optimizer = tf.compat.v1.train.AdamOptimizer().minimize(self.loss)
        
        # Initialize TF
        self.sess, self.saver = self.setup_tf()
    
    def setup_cnn(self) -> None:
        """
        Create CNN layers.
        """
        cnn_in4d = tf.expand_dims(input=self.input_imgs, axis = 3)

        # List of parameters for the layers
        kernel_vals = [5, 5, 3, 3, 3]
        feature_vals = [1, 32, 64, 128, 128, 256]
        stride_vals = pool_vals = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2)]
        num_layers = len(stride_vals)

        # Creating layers
        pool = cnn_in4d # Input as first CNN layer
        for i in range(num_layers):
            kernel = tf.Variable(
                tf.random.truncated_normal([kernel_vals[i], kernel_vals[i], feature_vals[i + 1]], 
                                            stddev=0.1))
            conv = tf.nn.conv2d(input=pool, filters=kernel, padding='SAME', strides=(1, 1, 1, 1))
            conv_norm = tf.compat.v1.layers.batch_normalization(conv, training=self.is_train)
            relu = tf.nn.relu(conv_norm)
            pool = tf.nn.max_pool2d(input=relu, ksize=(1, pool_vals[i][0], pool_vals[i][1], 1),
                                    strides=(1, stride_vals[i][0], stride_vals[i][1], 1), padding='VALID')
            
        
        self.cnn_out_4d = pool

    def setup_rnn(self) -> None:
        """
        Create RNN layers.
        """
        rnn_in3d = tf.squeeze(self.cnn_out_4d, axis=[2])
        
        # Basic cells which is used to build RNN
        num_hidden = 256
        cells = [tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=num_hidden, state_is_tuple=True) for _ in range(2)] # 2 Layers

        # Stacking basic cells
        stacked = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

        # Bidirectional RNN
        # BxTxF -> BxTx2H
        (fw, bw), _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked,
                                                                inputs=rnn_in3d, dtype=rnn_in3d.dtype)
                                    
        # BxTxH + BxTxH -> BxTx2H -> BxTx1x2H
        concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)

        # Project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
        kernel = tf.Variable(tf.random.truncated_normal([1, 1, num_hidden * 2, len(self.char_list) + 1], 
                            stddev=0.1))
        self.rnn_out_3d = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'),
                            axis=[2])

    def setup_ctc(self) -> None:
        """
        Create CTC loss and decoder
        """
        # BxTxC -> TxBxC
        self.ctc_in_3d_tbc = tf.transpose(a=self.rnn_out_3d, perm=[1, 0, 2])
        # Ground truth text as sparse tensor
        self.gt_texts = tf.SparseTensor(tf.compat.v1.placeholder(tf.int64, shape=[None, 2]),
                                        tf.compat.v1.placeholder(tf.int32, [None]),
                                        tf.compat.v1.placeholder(tf.int64, [2])
                                        )
        
        # Calculating loss for batch
        self.seq_len = tf.compat.v1.placeholder(tf.int32, [None])
        self.loss = tf.reduce_mean(
            input_tensor=tf.compat.v1.nn.ctc_loss(labels=self.gt_texts,
                                                  inputs=self.ctc_in_3d_tbc,
                                                  sequence_length=self.seq_len,
                                                  ctc_merge_repeated=True)
        )

        # Calculate loss for each element to compute label probability
        self.saved_ctc_input = tf.compat.v1.placeholder(tf.float32,
                                                        shape=[None, None, len(self.char_list) + 1])
        self.loss_per_element = tf.compat.v1.nn.ctc_loss(labels=self.gt_texts, inputs=self.saved_ctc_input,
                                                          sequence_length=self.seq_len, ctc_merge_repeated=True)

        # Decoding selection
        if self.decoder_type == DecoderType.BestPath:
            self.decoder = tf.nn.ctc_greedy_decoder(inputs=self.ctc_in_3d_tbc, sequence_length=self.seq_len)
        elif self.decoder_type == DecoderType.BeamSearch:
            self.decoder = tf.nn.ctc_beam_search_decoder(inputs=self.ctc_in_3d_tbc, sequence_length=self.seq_len, beam_width=50)
        
        # Word beam search decoding
        else:
            chars = ''.join(self.char_list)
            word_chars = open('../model/wordCharList.txt').read().splitlines()[0]
            corpus = open('../data/corpus.txt').read()

            # Decode using the "Words" mode of word beam search
            from word_beam_search import WordBeamSearch
            self.decoder = WordBeamSearch(50, 'Words', 0.0, corpus.encode('utf8'), chars.encode('utf8'),
                                          word_chars.encode('utf8'))
            
            # The input to the decoder must have softmax already applied
            self.wbs_input = tf.nn.softmax(self.ctc_in_3d_tbc, axis=2)

    def setup_tf(self) -> Tuple[tf.compat.v1.Session, tf.compat.v1.train.Saver]:
        """
        Initialize TF
        """
        print('Python: ' + sys.version)
        print('Tensorflow: ' + tf.__version__)

        sess = tf.compat.v1.Session()

        saver = tf.compat.v1.train.Saver(max_to_keep=1) # Saver saves model to file
        model_dir = '../model/'
        latest_snapshot = tf.train.latest_checkpoint(model_dir) # Check if any saved model

        # If model must be restored for inference, there must be a snapshot (What is snapshot btw?)
        if self.must_restore and not latest_snapshot:
            raise Exception('No saved model found in: ' + model_dir)

        # Load saved model if available
        if latest_snapshot:
            print('Initialize with stored values from ' + latest_snapshot)
            saver.restore(sess, latest_snapshot)
        else:
            print('Initialize with new values')
            sess.run(tf.compat.v1.global_variables_initializer())
        
        return sess, saver

    def to_sparse(self, texts: List[str]) -> Tuple[List[List[int]], List[int], List[int]]:
        """
        Put ground truth texts into sparse tensor for ctc_loss
        """
        indices = []
        values = []
        shape = [len(texts), 0] # Last entry must be max(labelList[i])

        # First, we go over all texts
        for batchElement, text in enumerate(texts):
            # Convert to string of label (i.e. class-ids)
            label_str = [self.char_list.index(c) for c in text]
            # Sparse tensor must have size of max. label_string
            if len(label_str) > shape[1]:
                shape[1] = len(label_str)
            # Put each label into sparse tensor
            for i, label in enumerate(label_str):
                indices.appen([batchElement, i])
                values.append(label)

        return indices, values, shape

    def decoder_output_to_text(self, ctc_output: tuple, batch_size: int) -> List[str]:
        """
        Extract texts from output of CTC decoder.
        """

        # Word beam search: already contains labels strings
        if self.decoder_type == DecoderType.WordBeamSearch:
            label_strs = ctc_output
        
        # TF decoders: label strings are contained in sparse tensor
        else:
            # CTC returns tuple, first element is SparseTensor
            decoded = ctc_output[0][0]

            # Contains string of labels for each batch element
            label_strs = [[] for _ in range(batch_size)]

            # Go over all indices and save mapping: Batch -> Values
            for (idx, idx2d) in enumerate(decoded.indices):
                label = decoded.values[idx]
                batch_element = idx2d[0] # Index according to [b,t]
                label_strs[batch_element].append(label)
        
        # Map labels to chars for all batch elements
        return [''.join([self.char_list[c] for c in labelStr]) for labelStr in label_strs]

    
    def train_batch(self, batch: Batch) -> float:
        """
        Feed a batch into the NN to train it.
        """
        num_batch_elements = len(batch.imgs)
        max_text_len = batch.imgs[0].shape[0] // 4
        sparse = self.to_sparse(batch.gt_texts)
        eval_list = [self.optimizer, self.loss]
        feed_dict = {self.input_imgs: batch.imgs, self.gt_texts: sparse,
                     self.seq_len: [max_text_len] * num_batch_elements, self.is_train: True}
        _, loss_val = self.sess.run(eval_list, feed_dict)
        self.batches_trained += 1
        return loss_val

    @staticmethod
    def dump_nn_output(rnn_output: np.ndarray) -> None:
        """
        Dump the output of the NN to CSV files
        """
        dump_dir = '../dump/'
        if not os.path.isdir(dump_dir):
            os.mkdir(dump_dir)

        # Iterate over all batch elements and create a CSV file for each one
        max_t, max_b, max_c = rnn_output.shape
        for b in range(max_b):
            csv = ''
            for t in range(max_t):
                csv += ';'.join([str(rnn_output[t, b, c]) for c in range(max_c)]) + ';\n'
            fn = dump_dir + 'rnnOutput_' + str(b) + '.csv'
            with open(fn, 'w') as f:
                f.write(csv)
    
    def infer_batch(self, batch: Batch, calc_probability: bool=False, probability_of_gt: bool=False):
        """
        Feed a batch into the NN to recognize the texts
        """

        # Decode, optionally save RNN output
        num_batch_elements = len(batch.imgs)

        # Put tensors to be evaluated into list
        eval_list = []

        if self.decoder_type == DecoderType.WordBeamSearch:
            eval_list.append(self.wbs_input)
        else:
            eval_list.append(self.decoder)

        if self.dump or calc_probability:
            eval_list.append(self.ctc_in_3d_tbc)

        # Sequence length depends on input image size (model downsizes width by 4)
        max_text_len = batch.imgs[0].shape[0] // 4

        # Dictionary containing all tensor fed into the model
        feed_dict = {self.input_imgs: batch.imgs, self.seq_len: [max_text_len] * num_batch_elements, self.is_train: False}

        # Evaluate the model
        eval_res = self.sess.run(eval_list, feed_dict)

        # TF decoders: Decoding already done in TF graph
        if self.decoder_type != DecoderType.WordBeamSearch:
            decoded = eval_res[0]
        else:
            decoded = self.decoder.compute(eval_res[0])

        # Map labels to character string
        texts = self.decoder_output_to_text(decoded, num_batch_elements)

        # Feed RNN output and recognized text into CTC loss to compute labeling probability
        probs = None
        if calc_probability:
            sparse = self.to_sparse(batch.gt_texts) if probability_of_gt else self.to_sparse(texts)
            ctc_input = eval_res[1]
            eval_list = self.loss_per_element
            feed_dict = {
                self.saved_ctc_input: ctc_input, self.gt_texts: sparse,
                self.seq_len: [max_text_len] * num_batch_elements, self.is_train: False
            }
            loss_vals = self.sess.run(eval_list, feed_dict)

        # Dumping the output of the NN to CSV files
        if self.dump:
            self.dump_nn_output(eval_res[1])

        return texts, probs

    def save(self) -> None:
        """
        Save model to files
        """
        self.snap_ID += 1
        self.saver.save(self.sess, '../model/snapshot', global_step=self.snap_ID)
        

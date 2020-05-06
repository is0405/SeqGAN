import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM
import numpy as np

class TARGET_LSTM(object):
    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim, sequence_length, start_token, params):
        self.num_emb = num_emb
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.start_token = start_token
        self.start_token_vec = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
        self.params = params

        self.g_model = tf.keras.models.Sequential([
            Input((self.sequence_length,), dtype=tf.int32),
            Embedding(self.num_emb, self.emb_dim),
            LSTM(self.hidden_dim, return_sequences=True),
            Dense(self.num_emb, activation="softmax")
        ])
        weights = [
            # Embedding
            params[0],
            # LSTM
            np.c_[params[1], params[4], params[10], params[7]], # kernel (i, f, c, o)
            np.c_[params[2], params[5], params[11], params[8]], # recurrent_kernel
            np.r_[params[3], params[6], params[12], params[9]], # bias
            # Dense
            params[13],
            params[14]
        ]
        self.g_model.compile(loss="sparse_categorical_crossentropy")
        self.g_model.set_weights(weights)
        self.g_embeddings = self.g_model.trainable_weights[0]

    def target_loss(self, dataset):
        # dataset: each element has [self.batch_size, self.sequence_length]
        # outputs are 1 timestep ahead
        ds = dataset.map(lambda x: (tf.pad(x[:, 0:-1], ([0, 0], [1, 0]), "CONSTANT", self.start_token), x))
        loss = self.g_model.evaluate(ds, verbose=1)
        return loss
    
    @tf.function
    def generate_one_batch(self):
        # initial states
        h0 = c0 = tf.zeros([self.batch_size, self.hidden_dim])
        h0 = [h0, c0]

        gen_x = tf.TensorArray(dtype=tf.int32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)

        def _g_recurrence(i, x_t, h_tm1, gen_x):
            # o_t: batch x vocab, probability
            # h_t: hidden_memory_tuple
            o_t, h_t = self.g_model.layers[1].cell(x_t, h_tm1, training=False) # layers[1]: LSTM
            o_t = self.g_model.layers[2](o_t) # layers[2]: Dense
            log_prob = tf.math.log(o_t)
            next_token = tf.cast(tf.reshape(tf.random.categorical(log_prob, 1), [self.batch_size]), tf.int32)
            x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)  # batch x emb_dim
            gen_x = gen_x.write(i, next_token)  # indices, batch_size
            return i + 1, x_tp1, h_t, gen_x

        _, _, _, gen_x = tf.while_loop(
            cond=lambda i, _1, _2, _3: i < self.sequence_length,
            body=_g_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token_vec), h0, gen_x)
            )

        gen_x = gen_x.stack()  # seq_length x batch_size
        outputs = tf.transpose(gen_x, perm=[1, 0])  # batch_size x seq_length
        return outputs

    def generate_samples(self, generated_num, output_file):
        # Generate Samples
        with open(output_file, 'w') as fout:
            for _ in range(generated_num // self.batch_size):
                generated_samples = self.generate_one_batch().numpy()
                for poem in generated_samples:
                    print(' '.join([str(x) for x in poem]), file=fout)

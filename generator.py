import tensorflow as tf

class Generator(object):
    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim,
                 sequence_length, start_token,
                 learning_rate=0.01, reward_gamma=0.95):
        self.num_emb = num_emb
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.reward_gamma = reward_gamma
        self.g_params = []
        self.d_params = []
        self.temperature = 1.0
        self.grad_clip = 5.0

        self.expected_reward = tf.Variable(tf.zeros([self.sequence_length]))

        #with tf.variable_scope('generator'):
        self.g_embeddings = tf.Variable(self.init_matrix([self.num_emb, self.emb_dim]))
        self.g_params.append(self.g_embeddings)
        self.g_recurrent_unit = self.create_recurrent_unit(self.g_params)  # maps h_tm1 to h_t for generator
        self.g_output_unit = self.create_output_unit(self.g_params)  # maps h_t to o_t (output token logits)
        self.g_optimizer = self.create_optimizer(self.learning_rate)

    @tf.function
    def generate(self):
        # Initial states
        h0 = tf.zeros([self.batch_size, self.hidden_dim])
        h0 = tf.stack([h0, h0])

        gen_o = tf.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)
        gen_x = tf.TensorArray(dtype=tf.int32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)

        def _g_recurrence(i, x_t, h_tm1, gen_o, gen_x):
            h_t = self.g_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
            o_t = self.g_output_unit(h_t)  # batch x vocab , logits not prob
            log_prob = tf.math.log(tf.nn.softmax(o_t))
            next_token = tf.cast(tf.reshape(tf.random.categorical(log_prob, 1), [self.batch_size]), tf.int32)
            x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)  # batch x emb_dim
            gen_o = gen_o.write(i, tf.reduce_sum(tf.multiply(tf.one_hot(next_token, self.num_emb, 1.0, 0.0),
                                                             tf.nn.softmax(o_t)), 1))  # [batch_size] , prob
            gen_x = gen_x.write(i, next_token)  # indices, batch_size
            return i + 1, x_tp1, h_t, gen_o, gen_x

        _, _, _, self.gen_o, self.gen_x = tf.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
            body=_g_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token), h0, gen_o, gen_x))

        gen_x = self.gen_x.stack()  # seq_length x batch_size
        outputs = tf.transpose(gen_x, perm=[1, 0])  # batch_size x seq_length

        #outputs = sess.run(self.gen_x)
        return outputs

    @tf.function
    def pretrain_step(self, x):
        # x: [self.batch_size, self.sequence_length]

        # Initial states
        h0 = tf.zeros([self.batch_size, self.hidden_dim])
        h0 = tf.stack([h0, h0])

        with tf.GradientTape() as tape:
            # processed for batch
            with tf.device("/cpu:0"):
                processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, x), perm=[1, 0, 2])  # seq_length x batch_size x emb_dim

            # supervised pretraining for generator
            g_predictions = tf.TensorArray(
                dtype=tf.float32, size=self.sequence_length,
                dynamic_size=False, infer_shape=True)

            ta_emb_x = tf.TensorArray(
                dtype=tf.float32, size=self.sequence_length)
            ta_emb_x = ta_emb_x.unstack(processed_x)

            def _pretrain_recurrence(i, x_t, h_tm1, g_predictions):
                h_t = self.g_recurrent_unit(x_t, h_tm1)
                o_t = self.g_output_unit(h_t)
                g_predictions = g_predictions.write(i, tf.nn.softmax(o_t))  # batch x vocab_size
                x_tp1 = ta_emb_x.read(i)
                return i + 1, x_tp1, h_t, g_predictions

            _, _, _, g_predictions = tf.while_loop(
                cond=lambda i, _1, _2, _3: i < self.sequence_length,
                body=_pretrain_recurrence,
                loop_vars=(tf.constant(0, dtype=tf.int32),
                           tf.nn.embedding_lookup(self.g_embeddings, self.start_token),
                           h0, g_predictions))

            g_predictions = tf.transpose(g_predictions.stack(), perm=[1, 0, 2])  # batch_size x seq_length x vocab_size

            # pretraining loss
            pretrain_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(x, g_predictions))

        # training updates
        pretrain_grad, _ = tf.clip_by_global_norm(tape.gradient(pretrain_loss, self.g_params), self.grad_clip)
        self.g_optimizer.apply_gradients(zip(pretrain_grad, self.g_params))
        return pretrain_loss

        
    def init_matrix(self, shape):
        return tf.random.normal(shape, stddev=0.1)

    def init_vector(self, shape):
        return tf.zeros(shape)

    def create_recurrent_unit(self, params):
        # Weights and Bias for input and hidden tensor
        self.Wi = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Ui = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bi = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wf = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uf = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bf = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wog = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uog = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bog = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wc = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uc = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bc = tf.Variable(self.init_matrix([self.hidden_dim]))
        params.extend([
            self.Wi, self.Ui, self.bi,
            self.Wf, self.Uf, self.bf,
            self.Wog, self.Uog, self.bog,
            self.Wc, self.Uc, self.bc])

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(previous_hidden_state, self.Ui) + self.bi
            )

            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(previous_hidden_state, self.Uf) + self.bf
            )

            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(previous_hidden_state, self.Uog) + self.bog
            )

            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit

    def create_output_unit(self, params):
        self.Wo = tf.Variable(self.init_matrix([self.hidden_dim, self.num_emb]))
        self.bo = tf.Variable(self.init_matrix([self.num_emb]))
        params.extend([self.Wo, self.bo])

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit

    def create_optimizer(self, *args, **kwargs):
        return tf.keras.optimizers.Adam(*args, **kwargs)

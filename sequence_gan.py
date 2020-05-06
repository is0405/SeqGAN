#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
#tf.disable_v2_behavior()

import os
import random
from dataloader import dataset_for_generator, dataset_for_discriminator
from generator import Generator
from discriminator import Discriminator
from rollout import ROLLOUT
from target_lstm import TARGET_LSTM
import pickle

#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 32 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
SEQ_LENGTH = 20 # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 120 # supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 64

#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64

#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 200
positive_file = 'save/real_data.txt'
negative_file = 'save/generator_sample.txt'
eval_file = 'save/eval_file.txt'
generated_num = 10000

def target_loss(target_lstm, dataset):
    # target_loss means the oracle negative log-likelihood tested with the oracle model "target_lstm"
    # For more details, please see the Section 4 in https://arxiv.org/abs/1609.05473
    nll = []
    for batch in dataset:
        g_loss = target_lstm.pretrain_loss(batch)
        nll.append(g_loss)
    return np.mean(nll)

def pretrain_callback(epoch, logs):
    if epoch % 5 == 0:
        generator.generate_samples(generated_num, eval_file)
        likelihood_dataset = dataset_for_generator(eval_file, BATCH_SIZE)
        test_loss = target_loss(target_lstm, likelihood_dataset)
        print('pre-train epoch ', epoch, 'test_loss ', test_loss)
        # buffer = 'epoch:\t'+ str(epoch) + '\tnll:\t' + str(test_loss) + '\n'
        # log.write(buffer)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    assert START_TOKEN == 0

    vocab_size = 5000

    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)
    target_params = pickle.load(open('save/target_params.pkl', 'rb'), encoding="bytes")
    target_lstm = TARGET_LSTM(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, target_params) # The oracle model

    discriminator = Discriminator(sequence_length=20, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim, 
                                  filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, dropout_keep_prob=dis_dropout_keep_prob,
                                  l2_reg_lambda=dis_l2_reg_lambda)

    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #sess = tf.Session(config=config)
    #sess.run(tf.global_variables_initializer())

    # First, use the oracle model to provide the positive examples, which are sampled from the oracle data distribution
    if not os.path.exists(positive_file):
        target_lstm.generate_samples(generated_num, positive_file)
    gen_dataset = dataset_for_generator(positive_file, BATCH_SIZE)

    log = open('save/experiment-log.txt', 'w')
    #  pre-train generator
    if not os.path.exists("generator_pretrained.h5"):
        print('Start pre-training...')
        log.write('pre-training...\n')
        generator.pretrain(gen_dataset, PRE_EPOCH_NUM, generated_num // BATCH_SIZE, callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=pretrain_callback)])
        generator.g_model.save_weights("generator_pretrained.h5", save_format="h5")
    else:
        generator.g_model.load_weights("generator_pretrained.h5")

    if not os.path.exists("discriminator_pretrained.h5"):
        print('Start pre-training discriminator...')
        # Train 3 epoch on the generated data and do this for 50 times
        for _ in range(50):
            print("Dataset", _)
            generator.generate_samples(generated_num, negative_file)
            dis_dataset = dataset_for_discriminator(positive_file, negative_file, BATCH_SIZE)
            discriminator.train(dis_dataset, 3, (generated_num // BATCH_SIZE) * 2)
        discriminator.d_model.save_weights("discriminator_pretrained.h5", save_format="h5")
    else:
        discriminator.d_model.load_weights("discriminator_pretrained.h5")

    rollout = ROLLOUT(generator, 0.8)

    # TODO: Reset optimizer parameters
    print('#########################################################################')
    print('Start Adversarial Training...')
    log.write('adversarial training...\n')
    for total_batch in range(TOTAL_BATCH):
        print("Generator", total_batch)
        # Train the generator for one step
        for it in range(1):
            samples = generator.generate_one_batch()
            rewards = rollout.get_reward(samples, 16, discriminator)
            generator.train_step(samples, rewards)

        # Test
        if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
            generator.generate_samples(generated_num, eval_file)
            likelihood_dataset = dataset_for_generator(eval_file, BATCH_SIZE)
            test_loss = target_loss(target_lstm, likelihood_dataset)
            buffer = 'epoch:\t' + str(total_batch) + '\tnll:\t' + str(test_loss) + '\n'
            print('total_batch: ', total_batch, 'test_loss: ', test_loss)
            log.write(buffer)

        # Update roll-out parameters
        rollout.update_params()

        # Train the discriminator
        print("Discriminator", total_batch)
        for _ in range(5):
            generator.generate_samples(generated_num, negative_file)
            dis_dataset = dataset_for_discriminator(positive_file, negative_file, BATCH_SIZE)
            discriminator.train(dis_dataset, 3, (generated_num // BATCH_SIZE) * 2)
    log.close()


if __name__ == '__main__':
    main()

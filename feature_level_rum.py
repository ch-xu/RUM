import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import ConfigParser
import string, os, sys
import logging
import logging.config
import time

class data_generation(object):
    def __init__(self, type, logger):
        print 'init'
        #np.random.seed(0)
        self.data_type = type
        self.logger = logger
        self.train_file_path = '../data/' + self.data_type +'_train_filtered'
        self.test_file_path = '../data/' + self.data_type +'_test_filtered'


        self.train_users = []
        self.train_items = []
        self.train_ratings = []
        self.train_labels = []

        self.test_users = []
        self.test_candidate_items = []
        self.test_batch_users = []
        self.test_batch_items = []
        self.test_batch_ratings = []
        self.test_batch_real_users = []
        self.test_batch_real_items = []
        self.test_batch_real_ratings = []


        self.neg_number = 1
        self.train_batch_id = 0
        self.test_batch_id = 0
        self.user_number = 0
        self.item_number = 0
        self.records_number = 0

    def gen_train_data(self):
        self.data = pd.read_csv(self.train_file_path, names = ['user', 'items'], dtype='str')
        is_first_line = 1
        for line in self.data.values:
            if is_first_line == 1:
                self.user_number = int(line[0])+1
                self.item_number = int(line[1])+1
                self.user_purchased_item = dict()
                self.user_purchased_item_number = np.zeros(self.user_number, dtype='int32')
                is_first_line = 0
            else:
                user_id = int(line[0])
                items_id = [i for i in line[1].split('@')]
                size = len(items_id)
                self.user_purchased_item_number[user_id] = size
                self.user_purchased_item[user_id] = [int(itm.split(':')[0]) for itm in items_id]
                for item in items_id:
                    itm = item.split(':')
                    self.train_users.append(int(user_id))
                    self.train_items.append(int(itm[0]))
                    self.train_ratings.append(int(float(itm[1])))
                    self.train_labels.append(1)
                    self.records_number += 1
                    for i in range(self.neg_number):
                        self.train_users.append(int(user_id))
                        neg = self.gen_neg(int(user_id))
                        self.train_items.append(neg)
                        self.train_ratings.append(0)
                        self.train_labels.append(0)
                        self.records_number += 1

    def gen_test_data(self):
        self.data = pd.read_csv(self.test_file_path, header=None, dtype='str')
        items = []
        is_first_line = 1
        for line in self.data.values:
            if is_first_line == 1:
                self.user_number = int(line[0])+1
                self.item_number = int(line[1])+1
                self.user_item_ground_true = dict()
                self.user_item_ground_true_number = np.zeros(self.user_number, dtype='int32')
                is_first_line = 0
            else:
                user_id = int(line[0])
                items_id = [i for i in line[1].split('@')]
                size = len(items_id)
                self.user_item_ground_true_number[user_id] = size
                self.user_item_ground_true[user_id] = [int(itm.split(':')[0]) for itm in items_id]

                self.test_batch_real_users.append(int(user_id))
                self.test_batch_real_items.append([int(itm.split(':')[0]) for itm in items_id])
                items += [int(itm.split(':')[0]) for itm in items_id]

        self.test_candidate_items = list(range(self.item_number))
        self.test_users = list(set(self.test_batch_real_users))

        for u in self.test_users:
            for item_id in self.test_candidate_items:
                self.test_batch_users.append(u)
                self.test_batch_items.append(item_id)

    def gen_neg(self, user_id):
        neg_item = np.random.randint(self.item_number)
        while neg_item in self.user_purchased_item[user_id]:
            neg_item = np.random.randint(self.item_number)
        return neg_item

    def shuffle(self):
        self.logger.info('shuffle ...')
        self.index = np.array(range(len(self.train_items)))
        np.random.shuffle(self.index)

        self.train_users = list(np.array(self.train_users)[self.index])
        self.train_items = list(np.array(self.train_items)[self.index])
        self.train_labels = list(np.array(self.train_labels)[self.index])
        self.train_ratings = list(np.array(self.train_ratings)[self.index])

    def gen_train_batch_data(self, batch_size):
        l = len(self.train_users)

        if self.train_batch_id + batch_size >= l:

            batch_users = self.train_users[self.train_batch_id:] + self.train_users[:self.train_batch_id + batch_size - l]
            batch_items = self.train_items[self.train_batch_id:] + self.train_items[:self.train_batch_id + batch_size - l]
            batch_ratings = self.train_ratings[self.train_batch_id:] + self.train_ratings[:self.train_batch_id + batch_size - l]
            batch_labels = self.train_labels[self.train_batch_id:] + self.train_labels[:self.train_batch_id + batch_size - l]
            #self.shuffle()
            self.train_batch_id = self.train_batch_id + batch_size - l
        else:
            batch_users = self.train_users[self.train_batch_id:self.train_batch_id + batch_size]
            batch_items = self.train_items[self.train_batch_id:self.train_batch_id + batch_size]
            batch_ratings = self.train_ratings[self.train_batch_id: self.train_batch_id + batch_size]
            batch_labels = self.train_labels[self.train_batch_id: self.train_batch_id + batch_size]

            self.train_batch_id = self.train_batch_id + batch_size

        return batch_users, batch_items, batch_ratings, batch_labels

    def gen_test_batch_data(self, user_number):
        l = len(self.test_users)
        if self.test_batch_id == len(self.test_candidate_items) * l:
            self.test_batch_id = 0

        batch_size = len(self.test_candidate_items) * user_number

        test_batch_users = self.test_batch_users[self.test_batch_id:self.test_batch_id + batch_size]
        test_batch_items = self.test_batch_items[self.test_batch_id:self.test_batch_id + batch_size]
        self.test_batch_id = self.test_batch_id + batch_size

        return test_batch_users, test_batch_items

class rum():
    def __init__(self, data_type):
        print 'init ...'
        #tf.set_random_seed(0)
        self.input_data_type = data_type
        logging.config.fileConfig('logging.conf')
        self.logger = logging.getLogger()

        self.dg = data_generation(self.input_data_type, self.logger)
        self.dg.gen_train_data()
        self.dg.gen_test_data()

        self.train_user_purchsed_items_dict = self.dg.user_purchased_item
        self.test_user_purchsed_items_dict = self.dg.user_item_ground_true

        self.user_number = self.dg.user_number
        self.item_number = self.dg.item_number
        self.neg_number = self.dg.neg_number
        self.user_purchased_items = self.dg.user_purchased_item
        print self.user_number
        print self.item_number

        self.test_users = self.dg.test_users
        self.test_candidates = self.dg.test_candidate_items
        self.test_real_u = self.dg.test_batch_real_users
        self.test_real_i = self.dg.test_batch_real_items

        self.global_dimension = 50
        self.batch_size = 1
        self.memory_rows = 20
        self.recommend_new = 1
        self.K = 5
        self.results = []

        self.step = 0
        self.iteration = 10
        self.display_step = self.dg.records_number

        self.initializer = tf.random_uniform_initializer(minval=0, maxval=0.1)
        self.c_init = tf.constant_initializer(value=0)

        self.len = tf.placeholder(tf.int32, shape=[], name='len')
        self.user_id = tf.placeholder(tf.int32, shape=[None], name='user_id')
        self.item_id = tf.placeholder(tf.int32, shape=[None], name='item_id')
        self.label = tf.placeholder(tf.float32, shape=[None], name='label')
        self.index = tf.placeholder(tf.float32, shape=[None], name='index')
        self.rating = tf.placeholder(tf.float32, shape=[None], name='rating')

        self.user_embedding_matrix = tf.get_variable('user_embedding_matrix', initializer=self.initializer, shape=[self.user_number, self.global_dimension])
        self.item_embedding_matrix = tf.get_variable('item_embedding_matrix', initializer=self.initializer, shape=[self.item_number, self.global_dimension])
        self.rating_embedding_matrix = tf.get_variable('rating_embedding_matrix', initializer=self.initializer, shape=[5, self.global_dimension])

        self.user_bias_vector = tf.get_variable('user_bias_vector', shape=[self.user_number])
        self.item_bias_vector = tf.get_variable('item_bias_vector', shape=[self.item_number])
        self.global_bias = tf.constant([0.0])

        self.feature_key = tf.get_variable('feature_key', shape=[self.memory_rows, self.global_dimension])

        self.memory = tf.get_variable('memory', shape=[self.user_number, self.memory_rows, self.global_dimension])
        self.dropout_keep_prob = 1.0


    def clear_memory(self):
        print 'clear memory'
        zeros = tf.constant(np.zeros((self.user_number, self.memory_rows, self.global_dimension)), dtype='float32')
        self.memory = tf.scatter_update(self.memory, range(self.user_number), zeros)

    def read_memory(self, user_id, item_id):
        self.memory_batch_read = tf.nn.embedding_lookup(self.memory, user_id)
        batch_key = tf.expand_dims(self.feature_key, axis=0)
        current_item_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, item_id)
        self.w = tf.reduce_sum(tf.multiply(batch_key, tf.expand_dims(current_item_embedding,axis=1)),axis=2)
        self.weight = tf.nn.softmax(tf.expand_dims(self.w, axis=2))
        out = tf.reduce_sum(tf.multiply(self.memory_batch_read, self.weight), axis=1)
        return out

    def erase(self, i):
        out = tf.nn.sigmoid(i)
        return out

    def add(self, i):
        out = tf.nn.tanh(i)
        return out

    def write_memory(self,user_id, item_id, len):
        current_item_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, item_id)
        self.memory_batch_write = tf.nn.embedding_lookup(self.memory, user_id)
        ones = tf.ones([len, self.memory_rows, self.global_dimension], tf.float32)
        self.rating_embedding = tf.nn.embedding_lookup(self.rating_embedding_matrix, tf.subtract(tf.to_int32(self.rating), tf.constant(1)))
        e = tf.expand_dims(self.erase(current_item_embedding), axis=1)
        a = tf.expand_dims(self.add(current_item_embedding), axis=1)
        decay = tf.subtract(ones, tf.multiply(self.weight, e))
        increase = tf.multiply(self.weight, a)
        self.new_value = tf.add(tf.multiply(self.memory_batch_write, decay), increase)
        self.memory = tf.scatter_update(self.memory, user_id, self.new_value)

    def merge(self, u, m):
        merged = tf.add(u, tf.multiply(tf.constant(0.2), m))
        return merged

    def build_model(self):
        print 'building model ...'

        self.user_embedding = tf.nn.embedding_lookup(self.user_embedding_matrix, self.user_id)
        self.item_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, self.item_id)
        self.memory_out = self.read_memory(self.user_id, self.item_id)
        self.new_user_embedding = self.merge(self.user_embedding, self.memory_out)

        self.user_bias = tf.nn.embedding_lookup(self.user_bias_vector, self.user_id)
        self.item_bias = tf.nn.embedding_lookup(self.item_bias_vector, self.item_id)

        self.write_memory(self.user_id, self.item_id, self.len)

        # compute loss
        self.element_wise_mul = tf.multiply(self.new_user_embedding, self.item_embedding)
        self.element_wise_mul_drop = tf.nn.dropout(self.element_wise_mul, self.dropout_keep_prob)

        self.log_intention = tf.reduce_sum(self.element_wise_mul_drop, axis=1)
        #self.log_intention = tf.add(tf.add(tf.add(self.log_intention, self.user_bias),self.item_bias), self.global_bias)
        self.intention_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.log_intention, name='sigmoid'))
        #self.intention_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(targets=self.label, logits=self.log_intention, name='sigmoid'))

        self.regular_loss = tf.add(0.0 * tf.nn.l2_loss(self.user_embedding),
                                   0.0 * tf.nn.l2_loss(self.item_embedding))
        self.intention_loss = tf.add(self.regular_loss, self.intention_loss)


        l = len(self.test_candidates)
        self.test_pol_matrix = tf.reshape(self.log_intention, shape=[-1, l])
        self.top_value, self.top_index = tf.nn.top_k(self.test_pol_matrix, k=l, sorted=True)

    def run(self):
        print 'running ...'
        max = 0
        max_result = []
        with tf.Session() as self.sess:
            self.intention_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(
                self.intention_loss)

            init = tf.initialize_all_variables()
            self.sess.run(init)

            for iter in range(self.iteration):
                print 'new iteration begin ...'
                print 'iteration: ' + str(iter)
                print self.dg.records_number

                while self.step * self.batch_size <= self.dg.records_number:
                    user_batch, item_batch, rating_batch, label = self.dg.gen_train_batch_data(self.batch_size)

                    if label[0] == 1:
                        self.sess.run([self.intention_optimizer], feed_dict={self.user_id: user_batch, self.item_id: item_batch,
                                                             self.rating: rating_batch, self.label: label,
                                                                       self.len: self.batch_size})

                        self.sess.run([self.memory],
                                      feed_dict={self.user_id: user_batch, self.item_id: item_batch,
                                                 self.rating: rating_batch, self.label: label,
                                                 self.len: self.batch_size})

                    else:
                        self.sess.run(self.intention_optimizer,
                                      feed_dict={self.user_id: user_batch, self.item_id: item_batch,
                                                 self.rating: rating_batch, self.label: label,
                                                 self.len: self.batch_size})


                    self.step += 1
                    if self.step * self.batch_size % 1000 == 0:
                        print 'evel ...'
                        r = self.evaluate()
                        if r[3] > max:
                            max = r[3]
                            max_result = r

                #self.clear_memory()
                self.step = 0

            self.save()
            print max_result
            print("Optimization Finished!")

    def save(self):
        item_latent_factors = self.sess.run(self.item_embedding_matrix)
        t = pd.DataFrame(item_latent_factors)
        t.to_csv('item_latent_factors')
        feature_keys = self.sess.run(self.feature_key)
        t = pd.DataFrame(feature_keys)
        t.to_csv('feature_keys')

    def NDCG_k(self, recommend_list, purchased_list):
        user_number = len(recommend_list)
        u_ndgg = []
        for i in range(user_number):
            temp = 0
            Z_u = 0
            for j in range(len(recommend_list[i])):
                Z_u = Z_u + 1 / np.log2(j + 2)
                if recommend_list[i][j] in purchased_list[i]:
                    temp = temp + 1 / np.log2(j + 2)
            if Z_u == 0:
                temp = 0
            else:
                temp = temp / Z_u
            u_ndgg.append(temp)
        return u_ndgg

    def top_k(self, pre_top_k, true_top_k):
        user_number = len(pre_top_k)
        correct = []
        co_length = []
        re_length = []
        pu_length = []
        p = []
        r = []
        f = []
        hit = []
        for i in range(user_number):
            temp = []
            for j in pre_top_k[i]:
                if j in true_top_k[i]:
                    temp.append(j)
            if len(temp):
                hit.append(1)
            else:
                hit.append(0)
            co_length.append(len(temp))
            re_length.append(len(pre_top_k[i]))
            pu_length.append(len(true_top_k[i]))
            correct.append(temp)

        #print co_length

        for i in range(user_number):
            if re_length[i] == 0:
                p_t = 0.0
            else:
                p_t = co_length[i] / float(re_length[i])
            if pu_length[i] == 0:
                r_t = 0.0
            else:
                r_t = co_length[i] / float(pu_length[i])
            p.append(p_t)
            r.append(r_t)
            if p_t != 0 or r_t != 0:
                f.append(2.0 * p_t * r_t / (p_t + r_t))
            else:
                f.append(0.0)
        return p, r, f, hit

    def evaluate(self):
        user_number = 1
        all_p = []
        all_r = []
        all_f1 = []
        all_hit_ratio = []
        all_ndcg = []

        for i in range(len(self.test_users) / user_number):
            batch_users, batch_items = self.dg.gen_test_batch_data(user_number)
            top_k_value, top_k_index = self.sess.run(
                [self.top_value, self.top_index],
                feed_dict={self.user_id: batch_users,
                           self.item_id: batch_items})

            pre_top_k = []
            ground_truth = []

            user_index_begin = i * user_number
            user_index_end = (i + 1) * user_number

            for user_index in range(user_index_begin, user_index_end):
                index = [j for j in top_k_index[user_index - user_index_begin] if
                         self.test_candidates[j] not in self.train_user_purchsed_items_dict[
                             self.test_users[user_index]]]
                items = [self.test_candidates[j] for j in index]

                pre_top_k.append(list(items[:self.K]))
                ground_truth.append([k for k in self.test_user_purchsed_items_dict[self.test_users[user_index]] if
                                     k in self.test_candidates])

            p, r, f1, hit_ratio = self.top_k(pre_top_k, ground_truth)
            ndcg = self.NDCG_k(pre_top_k, ground_truth)

            all_p.append(np.array(p).mean())
            all_r.append(np.array(r).mean())
            all_f1.append(np.array(f1).mean())
            all_hit_ratio.append(np.array(hit_ratio).mean())
            all_ndcg.append(np.array(ndcg).mean())

        self.logger.info("Presicion@" + str(self.K) + "= " + "{:.6f}".format(np.array(all_p).mean()))
        self.logger.info("Recall@" + str(self.K) + "= " + "{:.6f}".format(np.array(all_r).mean()))
        self.logger.info("F1@" + str(self.K) + "= " + "{:.6f}".format(np.array(all_f1).mean()))
        self.logger.info("hit@" + str(self.K) + "= " + "{:.6f}".format(np.array(all_hit_ratio).mean()))
        self.logger.info("NDCG@" + str(self.K) + "= " + "{:.6f}".format(np.array(all_ndcg).mean()))

        return [np.array(all_p).mean(), np.array(all_r).mean(), np.array(all_f1).mean(), np.array(all_hit_ratio).mean(),
                np.array(all_ndcg).mean()]



if __name__ == '__main__':
    type = 'rating_instant_video_raw10_0.7'
    m = rum(type)
    m.build_model()
    m.run()




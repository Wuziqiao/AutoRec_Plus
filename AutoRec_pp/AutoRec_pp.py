import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import time
import numpy as np
import os
import math


class AutoRec_pp():
    def __init__(self, args,
                 num_users, num_items,
                 train_R, train_mask_R, test_R, test_mask_R, num_train_ratings, num_test_ratings,
                 user_train_set, item_train_set, user_test_set, item_test_set,
                 ):

        self.args = args

        self.num_users = num_users
        self.num_items = num_items

        self.train_R = train_R
        self.train_mask_R = train_mask_R
        self.test_R = test_R
        self.test_mask_R = test_mask_R
        self.num_train_ratings = num_train_ratings
        self.num_test_ratings = num_test_ratings

        self.user_train_set = user_train_set
        self.item_train_set = item_train_set
        self.user_test_set = user_test_set
        self.item_test_set = item_test_set

        self.hidden_neuron = args.hidden_neuron
        self.train_epoch = args.train_epoch
        self.batch_size = args.batch_size
        self.num_batch = int(math.ceil(self.num_items / float(self.batch_size)))

        self.base_lr = args.base_lr
        self.optimizer_method = args.optimizer_method
        self.display_step = args.display_step
        self.random_seed = args.random_seed

        self.global_step = tf.Variable(0, trainable=False)
        self.decay_epoch_step = args.decay_epoch_step
        self.decay_step = self.decay_epoch_step * self.num_batch
        self.lr = tf.train.exponential_decay(self.base_lr, self.global_step,
                                             self.decay_step, 0.8, staircase=True)
        self.lambda_value_1 = args.lambda_value_1
        self.lambda_value_2 = args.lambda_value_2

        self.train_cost_list = []
        self.test_cost_list = []
        self.test_rmse_list = []
        self.test_mae_list = []
        self.time_list = []
        self.grad_clip = args.grad_clip

        self.Encoder = None

    def run(self, sess, result_path, PB_mu, PB_bi, PB_bu):
        self.sess = sess
        self.result_path = result_path
        self.PB_bi = PB_bi
        self.PB_bu = PB_bu
        self.PB_mu = PB_mu

        self.prepare_model()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        total_time = 0
        for epoch_itr in range(self.train_epoch):
            start_time = time.time()
            self.train_model(epoch_itr)
            self.test_model(epoch_itr)
            total_time += (time.time() - start_time)
            self.time_list.append(total_time)
        self.make_records()
        return self.test_rmse_list, min(self.test_rmse_list), str(
            self.test_rmse_list.index(min(self.test_rmse_list))), min(self.test_mae_list), str(
            self.test_mae_list.index(min(self.test_mae_list)))

    def prepare_model(self):

        dataset = tf.data.Dataset.from_tensor_slices((self.train_R, self.train_mask_R, list(self.item_train_set),
                                                      self.PB_bi, self.PB_bu, self.test_R, self.test_mask_R))
        train_dataset = dataset.shuffle(buffer_size=1000)
        train_dataset = train_dataset.batch(self.batch_size)

        test_dataset = dataset.batch(self.batch_size)
        iter = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)
        self.train_init_op = iter.make_initializer(train_dataset)
        self.test_init_op = iter.make_initializer(test_dataset)

        input_R, input_mask_R, input_ids, input_PB_bi, input_PB_bu, input_test_r, input_test_r_mask = iter.get_next()
        input_R, input_mask_R, input_PB_bi, input_PB_bu, input_test_r, input_test_r_mask = tf.cast(input_R,
                                                                                                   tf.float32), tf.cast(
            input_mask_R,
            tf.float32), tf.cast(
            input_PB_bi, tf.float32), tf.cast(input_PB_bu, tf.float32), tf.cast(input_test_r, tf.float32), tf.cast(
            input_test_r_mask, tf.float32)
        V = tf.get_variable(name="V", initializer=tf.truncated_normal(shape=[self.num_users, self.hidden_neuron],
                                                                      mean=0, stddev=0.01), dtype=tf.float32)
        W = tf.get_variable(name="W", initializer=tf.truncated_normal(shape=[self.hidden_neuron, self.num_users],
                                                                      mean=0, stddev=0.01), dtype=tf.float32)
        mu = tf.get_variable(name="mu", initializer=tf.random_normal(shape=[self.hidden_neuron], stddev=0.01),
                             dtype=tf.float32)
        b = tf.get_variable(name="b", initializer=tf.random_normal(shape=[self.num_users], stddev=0.01),
                            dtype=tf.float32)

        # init TB
        TB_u_par = tf.get_variable(name="tb_u", initializer=tf.ones(shape=self.num_users), dtype=tf.float32)
        TB_i_par = tf.get_variable(name="tb_i", initializer=tf.ones(shape=self.num_items), dtype=tf.float32)

        # gather the id of the involved items
        TB_i = tf.gather(TB_i_par, input_ids)

        pre_Encoder = tf.matmul(input_R, V) + mu
        self.Encoder = tf.nn.sigmoid(pre_Encoder)
        pre_Decoder = tf.matmul(self.Encoder, W) + b
        pre_Decoder = tf.add(tf.add(tf.add(tf.add(tf.add(pre_Decoder, TB_u_par), tf.reshape(TB_i, (
            tf.shape(input_R)[0], 1))), input_PB_bi), input_PB_bu), self.PB_mu)
        self.Decoder = tf.identity(pre_Decoder)

        pre_numerator = tf.multiply((pre_Decoder - input_test_r), input_test_r_mask)
        self.numerator_mae = tf.reduce_sum(tf.abs(pre_numerator))
        self.numerator_rmse = tf.reduce_sum(tf.square(pre_numerator))

        pre_rec_cost = tf.multiply((input_R - self.Decoder), input_mask_R)
        rec_cost = tf.square(self.l2_norm(pre_rec_cost))
        pre_reg_cost_1 = tf.square(self.l2_norm(W)) + tf.square(self.l2_norm(V))
        pre_reg_cost_2 = tf.square(tf.reduce_sum(TB_i)) + tf.square(tf.reduce_sum(TB_u_par))
        reg_cost = self.lambda_value_1 * pre_reg_cost_1 + self.lambda_value_2 * pre_reg_cost_2

        self.cost = rec_cost + reg_cost

        if self.optimizer_method == "Adam":
            optimizer = tf.train.AdamOptimizer(self.lr)
        elif self.optimizer_method == "RMSProp":
            optimizer = tf.train.RMSPropOptimizer(self.lr)
        else:
            raise ValueError("Optimizer Key ERROR")

        if self.grad_clip:
            gvs = optimizer.compute_gradients(self.cost)
            capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
            self.optimizer = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)
        else:
            self.optimizer = optimizer.minimize(self.cost, global_step=self.global_step)

    def train_model(self, itr):
        start_time = time.time()
        self.sess.run(self.train_init_op)
        batch_cost = 0
        for i in range(self.num_batch):
            _, Cost = self.sess.run(
                [self.optimizer, self.cost])

            batch_cost = batch_cost + Cost
        self.train_cost_list.append(batch_cost)

        if (itr + 1) % self.display_step == 0:
            print("Training //", "Epoch %d //" % (itr), " Total cost = {:.2f}".format(batch_cost),
                  "Elapsed time : %d sec" % (time.time() - start_time))

    def test_model(self, itr):
        start_time = time.time()
        self.sess.run(self.test_init_op)
        numerator_rmse = 0
        numerator_mae = 0
        costs = 0
        for i in range(self.num_batch):
            Cost, num_rmse, num_mae = self.sess.run([self.cost, self.numerator_rmse, self.numerator_mae])
            costs += Cost
            numerator_rmse += num_rmse
            numerator_mae += num_mae
        RMSE = np.sqrt(numerator_rmse / float(self.num_test_ratings))
        MAE = numerator_mae / float(self.num_test_ratings)
        self.test_cost_list.append(costs)
        self.test_mae_list.append(MAE)
        self.test_rmse_list.append(RMSE)

        if (itr + 1) % self.display_step == 0:
            # save model

            print("Testing //", "Epoch %d //" % (itr), " Total cost = {:.2f}".format(costs),
                  " RMSE = {:.5f}".format(RMSE), " MAE = {:.5f}".format(MAE),
                  "Elapsed time : %d sec" % (time.time() - start_time))
            print("=" * 50)

    def make_records(self):
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        basic_info = self.result_path + "basic_info.txt"
        train_record = self.result_path + "train_record.txt"
        test_record = self.result_path + "test_record.txt"

        rmse_time = str(self.time_list[int(self.test_rmse_list.index(min(self.test_rmse_list)))])
        mae_time = str(self.time_list[int(self.test_mae_list.index(min(self.test_mae_list)))])

        print("rmse_time:" + rmse_time + ";mae_time:" + mae_time)
        with open(train_record, 'w') as f:
            f.write(str("Cost:"))
            f.write('\t')
            for itr in range(len(self.train_cost_list)):
                f.write(str(self.train_cost_list[itr]))
                f.write('\t')
            f.write('\n')

        with open(test_record, 'w') as g:
            g.write(str("Cost:"))
            g.write('\t')
            for itr in range(len(self.test_cost_list)):
                g.write(str(self.test_cost_list[itr]))
                g.write('\t')
            g.write('\n')

            g.write(str("RMSE:"))
            for itr in range(len(self.test_rmse_list)):
                g.write(str(self.test_rmse_list[itr]))
                g.write('\t')
            g.write('\n')

            g.write(str("MAE:"))
            for itr in range(len(self.test_mae_list)):
                g.write(str(self.test_mae_list[itr]))
                g.write('\t')
            g.write('\n')

            g.write(str("Best_RMSE:"))
            g.write(str(min(self.test_rmse_list)))
            g.write('\n')

            g.write(str("Best_RMSE_epoch:"))
            g.write(str(self.test_rmse_list.index(min(self.test_rmse_list))))
            g.write('\n')

            g.write(str("Best_RMSE_time:"))
            g.write(rmse_time)
            g.write('\n')

            g.write(str("Best_MAE:"))
            g.write(str(min(self.test_mae_list)))
            g.write('\n')

            g.write(str("Best_MAE_epoch:"))
            g.write(str(self.test_mae_list.index(min(self.test_mae_list))))
            g.write('\n')

            g.write(str("Best_MAE_time:"))
            g.write(mae_time)
            g.write('\n')

        with open(basic_info, 'w') as h:
            h.write(str(self.args))

    def l2_norm(self, tensor):
        return tf.sqrt(tf.reduce_sum(tf.square(tensor)))

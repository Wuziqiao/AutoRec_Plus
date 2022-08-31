from decimal import Decimal
import tensorflow as tf
from data_preprocessor import *
from AutoRec_pp import AutoRec_pp
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import time
import argparse

current_time = time.time()

parser = argparse.ArgumentParser(description='I-AutoRec ')
parser.add_argument('--hidden_neuron', type=int, default=500)
parser.add_argument('--lambda_value_1', type=float, default=1)
parser.add_argument('--lambda_value_2', type=float, default=1)

parser.add_argument('--train_epoch', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=1500)

parser.add_argument('--optimizer_method', choices=['Adam', 'RMSProp'], default='Adam')
parser.add_argument('--grad_clip', type=bool, default=False)
parser.add_argument('--base_lr', type=float, default=1e-3)
parser.add_argument('--decay_epoch_step', type=int, default=50, help="decay the learning rate for each n epochs")

parser.add_argument('--random_seed', type=int, default=1000)
parser.add_argument('--display_step', type=int, default=1)

args = parser.parse_args()
tf.set_random_seed(args.random_seed)
np.random.seed(args.random_seed)

data_name = 'douban'
num_users = 3000
num_items = 3000
num_total_ratings = 136891
train_ratio = 0.9
path = "data/%s" % data_name + "/"

train_R, train_mask_R, test_R, test_mask_R, num_train_ratings, num_test_ratings, \
user_train_set, item_train_set, user_test_set, item_test_set, known_entry_set, num_Ru, num_Ri \
    = read_rating(path, num_users, num_items, num_total_ratings, 1, 0, train_ratio)


# PB
PB_mu = train_R.sum() / num_train_ratings
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

PB_bi = (((train_R - PB_mu) * train_mask_R).sum(axis=0) / (
        train_R.sum(axis=0) / np.count_nonzero(train_R, axis=0) + num_Ri)).squeeze()

# replace inf and nan with 0
PB_bi[PB_bi == np.inf] = 0
PB_bi[PB_bi == -np.inf] = 0
PB_bi = np.nan_to_num(PB_bi)
PB_bi = np.repeat(np.expand_dims(PB_bi, axis=0), train_R.shape[0], axis=0)
PB_bu = (((train_R - PB_mu - PB_bi) * train_mask_R).sum(axis=1) / (
        train_R.sum(axis=1) / np.count_nonzero(train_R, axis=1) + num_Ru)).squeeze()
PB_bu[PB_bu == np.inf] = 0
PB_bu[PB_bu == -np.inf] = 0
PB_bu = np.nan_to_num(PB_bu)
PB_bu = np.repeat(np.expand_dims(PB_bu, axis=1), train_R.shape[1], axis=1)

result_path = 'results/' + data_name + '/' + str(args.optimizer_method) + '_' + str(
    args.base_lr) + "/"

# train
with tf.Session(config=config) as sess:
    Autorec_pp = AutoRec_pp(args, num_users, num_items, train_R.T, train_mask_R.T, test_R.T,
                            test_mask_R.T,
                            num_train_ratings, num_test_ratings, user_train_set, item_train_set, user_test_set,
                            item_test_set)
    # run
    test_result, best_RMSE_result, epoch_RMSE_index, best_MAE_result, epoch_MAE_index = Autorec_pp.run(sess,
                                                                                                       result_path,
                                                                                                       PB_mu,
                                                                                                       PB_bi.T,
                                                                                                       PB_bu.T)
    print("=========best_RMSE_result:" + str(
        Decimal(best_RMSE_result).quantize(Decimal('0.000'))) + "best_MAE_result:" + str(
        Decimal(best_MAE_result).quantize(Decimal('0.000'))) + "============")
    sess.close()
tf.reset_default_graph()
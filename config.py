
import argparse

def get_args():

  parser = argparse.ArgumentParser()
  parser.add_argument('--random_seed', type=int, default=123, help='random seed')

  parser.add_argument('--train_dir', type=str, default='data/pre/train.csv')
  parser.add_argument('--log_dir', type=str, default='save/logs')
  parser.add_argument('--save_dir', type=str, default='save/saves')
  parser.add_argument('--vocab_dir', type=str, default='data/pre/vocab.txt', help='vocab dir')
  parser.add_argument('--nb_classes', type=int, default=7, help='class numbers')
  parser.add_argument('--model_type', type=str, default="cnn")

  parser.add_argument('--sentence_length', type=int, default=40, help='The length of input x')
  parser.add_argument('--vocab_size', type=int, default=10000, help='data vocab size')
  parser.add_argument('--embed_size', type=int, default=128, help='dims of word embedding')
  parser.add_argument('--filter_sizes', type=list, default=[3, 4, 5], help='')
  parser.add_argument('--num_filters', type=int, default=128, help='num of filters')
  parser.add_argument('--hidden_size', type=int, default=128, help='rnn hidden size')
  parser.add_argument('--keep_prob', type=float, default=1, help='keep prob in dropout')

  parser.add_argument('--batch_size', type=int, default=32, help='Example numbers every batch')
  parser.add_argument('--max_size', type=int, default=1000, help='max numbers every batch')
  parser.add_argument('--max_step', type=int, default=50000, help='max train step')
  parser.add_argument('--nb_epochs', type=int, default=8, help='Number of epoch')   #rnn 7.8
  parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
  parser.add_argument('--max_grad_norm', type=float, default=10.0, help='Max norm of gradient')

  return parser.parse_args()
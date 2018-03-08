
import argparse

def get_args():

  parser = argparse.ArgumentParser()
  parser.add_argument('--random_seed', type=int, default=1023, help='random seed')
  #data
  parser.add_argument('--train_dir', type=str, default='pre/train.csv', help="train data")
  parser.add_argument('--valid_dir', type=str, default='pre/valid.csv', help="valid data")
  parser.add_argument('--test_dir', type=str, default='../data/pre/test.csv', help="test data")
  parser.add_argument('--sub_dir', type=str, default='../data/pre/submission.csv', help="submission save")
  parser.add_argument('--vocab_dir', type=str, default='../data/pre/vocab.txt', help='vocab dir')
  parser.add_argument('--wordvec_dir', type=str, default='../data/glove/crawl-300d-2M.vec', help='fasttext dir')
  parser.add_argument('--save_dir', type=str, default='save/saves')

  # model
  parser.add_argument('--model_type', type=str, default="cnn", help='cnn,rnn,attention')
  parser.add_argument('--nb_classes', type=int, default=6, help='class numbers')
  parser.add_argument('--max_len', type=int, default=200, help='The length of input x')
  parser.add_argument('--vocab_size', type=int, default=20001, help='data vocab size')
  parser.add_argument('--embed_size', type=int, default=300, help='dims of word embedding')
  #cnn
  parser.add_argument('--filter_sizes', type=list, default=[1, 2, 3, 4, 5], help='')
  parser.add_argument('--num_filters', type=int, default=128, help='num of filters')
  parser.add_argument('--keep_prob_cnn', type=float, default=0.5, help='keep prob in cnn')
  parser.add_argument('--max_size_cnn', type=int, default=1000, help='max numbers every batch of cnn')
  parser.add_argument('--max_step_cnn', type=int, default=10000, help='max cnn train step')

  #rnn
  parser.add_argument('--cell_type', type=str, default="gru", help='lstm or gru')
  parser.add_argument('--hidden_size', type=int, default=64, help='rnn hidden size')
  parser.add_argument('--rnn_layers', type=int, default=2, help='rnn layers')
  parser.add_argument('--keep_prob_rnn', type=float, default=0.5, help='keep prob in dropout')
  parser.add_argument('--max_size_rnn', type=int, default=100, help='max numbers every batch of rnn')
  parser.add_argument('--max_step_rnn', type=int, default=16000, help='max rnn train step')

  parser.add_argument('--batch_size', type=int, default=128, help='example numbers every batch')
  parser.add_argument('--learning_rate', type=float, default=0.0005, help='initial learning rate')
  parser.add_argument('--max_grad_norm', type=float, default=5.0, help='max norm of gradient')
  parser.add_argument('--nb_epochs', type=int, default=10, help='number of epoch')
  parser.add_argument('--nfolds', type=int, default=10, help='cv') 

  return parser.parse_args()
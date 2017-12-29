
import os
import utils
import time
import config
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def run_eval(args, dev_model, dev_sess, model_dir):
  with dev_model.graph.as_default():
    loaded_dev_model, global_step = helper.create_or_load_model(
        dev_model.model, model_dir, dev_sess, "dev")

  _run_eval(args, loaded_infer_model, global_step, infer_sess, infer_model.iterator)

def _sample_generate(args, model, global_step, sess, iterator):
  if iterator is not None:
    sess.run(iterator.initializer)

  sample_id, sample_words = model.infer(sess)
  if args.beam_width > 0:
    sample_words = sample_words[0]

  for sample in sample_words: # many sent
    utils.print_out("gen sentence: %s" % (" ".join(sample)))

def main(args):

  data_dir = args.train_dir
  vocab_dir = args.vocab_dir
  save_dir = args.save_dir
  log_dir = args.log_dir
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)

  train_model = helper.build_train_model(args)
  dev_model = helper.build_eval_model(args)

  config_proto = utils.get_config_proto()
  train_sess = tf.Session(config=config_proto, graph=train_model.graph)
  dev_sess = tf.Session(config=config_proto, graph=dev_model.graph)

  with train_model.graph.as_default():
    loaded_train_model, global_step = helper.create_or_load_model(
        train_model.model, args.save_dir, train_sess, name="train")

    sess.run(model.iterator.initializer)
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

    loss = 0.
    total_reviews = 0
    epoch = 1
    epoch_start_time = time.time()
    step_start_time = epoch_start_time

    print "Epoch %d start " % (epoch)
    print "- " * 50

    for step in range(args.max_step):
      try:
        _, loss_t, global_step, batch_size, summaries = model.train(sess, args.keep_prob)

        loss += loss_t * batch_size
        total_reviews += batch_size
        summary_writer.add_summary(summaries, global_step)

        if global_step % 100 == 0:
          print "epoch %d, step %d, loss %f, time %.2fs" % \
            (epoch, global_step, loss_t, time.time() - step_start_time)
          step_start_time = time.time()

      except tf.errors.OutOfRangeError:
        print "epoch %d finish, time %.2fs" % (epoch, time.time() - epoch_start_time)
        print "- " * 50
        sess.run(model.iterator.initializer)
        epoch_time = time.time() - epoch_start_time
        epoch_start_time = time.time()
        print "%.2f seconds in this epoch" % (epoch_time)
        print "train loss %f total comments %d" % (loss / total_reviews, total_reviews)
        loaded_train_model.saver.save(train_sess,
                      os.path.join(args.save_dir, "model.ckpt"), global_step=global_step)
        run_eval(args, dev_model, dev_sess, args.save_dir)

        total_reviews = 0
        loss = 0.
        epoch += 1
        print "Epoch %d start " % (epoch)
        print "- " * 50
        continue

if __name__ == '__main__':
  args = config.get_args()
  main(args)
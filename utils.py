
def read_row(csv_row):
    record_defaults = [[0], [0.], [""], [0], [0], [0], [0], [0], [0], [0]]
    row = tf.decode_csv(csv_row, record_defaults=record_defaults)
    return row[2], row[3:]

def input_pipeline(filenames, batch_size):
    dataset = (tf.data.TextLineDataset(filenames)
               .skip(1)
               .map(lambda line: read_row(line))
               #.shuffle(buffer_size=10)
               .batch(batch_size))
    return dataset.make_initializable_iterator()

iterator = input_pipeline(['pre/dev.csv'], 1)
features, labels = iterator.get_next()

nof_examples = 10
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    sess.run(iterator.initializer)
    while nof_examples > 0:
        nof_examples -= 1
        try:
            data_features, data_labels = sess.run([features, labels])
            print(data_features)
            print(data_labels)
        except tf.errors.OutOfRangeError:
            pass
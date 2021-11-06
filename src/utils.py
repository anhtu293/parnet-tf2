import tensorflow as tf
import tensorflow_datasets as tfds


def load_dataset(dataset,
                 num_classes,
                 augmentation=False,
                 train=True,
                 batch_size=8):
    # normalize
    def normalize(img, label):
        img = tf.cast(img, tf.float32) / 255.
        label = tf.one_hot(label, num_classes)
        return img, label

    # data augmentation
    def augment(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_crop(img, (20, 20, 3))
        img = tf.image.resize(img, (32, 32, 3))
        return img, label

    if train:
        if dataset == 'imagenet':
            split = ['train', 'val']
        else:
            split = ['train[:90%]', 'train[90%:]']
        train, val = tfds.load(dataset, split=split,
                               shuffle_files=True,
                               as_supervised=True)
        train = train.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
        val = val.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
        if augmentation:
            train = train.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
            val = val.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        train = train.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val = val.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return train, val
    else:
        split =['test']
        ds = tfds.load(dataset, split=split,
                       shuffle_files=True,
                       as_supervised=True)
        ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
        ds.prefetch(batch_size * 2).batch(batch_size)
        return ds

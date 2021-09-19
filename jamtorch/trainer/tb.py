import numpy as np
import scipy.misc
# FIXME: bug for tf1,tf2
try:
    from StringIO import StringIO as BytesIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x

import tensorflow

if tensorflow.__version__ >= "1.14.0":
    import tensorflow.compat.v1 as tf
    create_fn = tensorflow.summary.create_file_writer
else:
    import tensorflow as tf
    create_fn = tf.summary.FileWriter


class TBLogger(object):
    # Adapted from:
    # https://raw.githubusercontent.com/SherlockLiao/pytorch-beginner/

    def __init__(self, log_dir):
        self.writer = create_fn(log_dir)

    def scalar_summary(self, tag, value, step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        img_summaries = []
        for i, img in enumerate(images):
            # BUG: remove toimage
            # Write the image to a string
            s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(
                encoded_image_string=s.getvalue(),
                height=img.shape[0],
                width=img.shape[1],
            )
            # Create a Summary value
            img_summaries.append(
                tf.Summary.Value(tag="%s/%d" % (tag, i), image=img_sum)
            )

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)

    def flush(self):
        self.writer.flush()

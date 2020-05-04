import tensorflow as tf

#Protocol Buffer
def dict_to_example(data_dict, coonfig):
    feature_dict = {}
    for feature, value in data_dict.items():
        feature_config = config[feature]
        shape = feature_config['shape']
        if shape == () or shape == []:
            value = [value]
        value_type = feature_config['type']
        if value_type == 'int':
            feature_dict[feature] = tf.train.Feature(int64_list=tf.train.Int64List(value))
        elif value_type == 'float':
            feature_dict[feature] = tf.train.Feature(float_list=tf.train.FloatList(value))
        elif value_type == 'string' or value_type == 'bytes':
            if value_type == 'string':
                value = [s.encode() for s in value]
            feature_dict[feature] = tf.train.Feature(byte_list=tf.train.ByteList(value))
    features = tf.train.Features(feature=feature_dict)
    return tf.train.Example(features=features)

def create_example_spec(config):
    example_spec = {}
    for fname, fconfig in config.items():
        if fconfig['type'] == 'int':
            tf_type = tf.int64
        elif fconfig['type'] == 'float':
            tf_type = tf.float32
        else:
            tf_type = tf.string
        shape = fconfig['shape']
        if shape is None:
            feature = tf.VarLenFeature(tf_type)
        else:
            default_value = fconfig['default_value']
            feature = tf.FixedLenFeature(shape, tf_type, default_value)
        example_spec[fname] = feature
    return example_spec

#Parsing
def parse_example(example_bytes, example_spec, output_features=None):
    parsed_features = tf.parse_single_example(example_bytes, example_spec)
    if output_features is not None:
        parsed_features = {k: parsed_features[k] for k in output_features}
    return parsed_features

#Mapping to Dataset
def dataset_from_examples(filenames, config, output_features=None):
    example_spec = create_example_spec(config)
    dataset = tf.data.TFRecordsDataset(filenames)
    def wrapper(example):
        return parse_example(example, example_spec, output_features)
    return dataset.map(lambda x: wrapper(x))

def create_feature_columns(config, example_spec, output_features=None):
    if output_features is None:
        output_features = config.keys()
    feature_columns = []
    for feature_name in output_features:
        dtype = example_spec[feature_name].dtype
        feature_config = config[feature_name]
        if 'vocab_list' in feature_config:
            vocab_col = tf.feature_column.categorical_column_with_vocabulary_list(feature_name,
                    feature_config['vocab_list'],
                    dtype=dtype)
            feature_col = tf.feature_column.indicator_column(vocab_col)
        elif 'vocab_file' in feature_config:
            vocab_col = tf.feature_column.categorical_column_with_vocaublary_file(feature_name,
                    feature_config['vocab_file'],
                    dtype=dtype)
            feature_col = tf.feature_column.indicator_column(vocab_col)
        else:
            feature_col = tf.feature_column.numeric_column(feature_name,
                    shape=feature_config['shape'],
                    dtype=dtype)
        feature_columns.append(feature_col)
    return feature_columns



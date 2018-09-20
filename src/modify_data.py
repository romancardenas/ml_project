def to_binary(data, columns):
    out_data = data.copy()
    for column in columns:
        out_data[column] = (data[column] != 0).astype(int)
    return out_data


def one_to_K(data, columns):
    pass


def normalize(data, columns=None):
    if columns is None:
        mean = data.mean(axis=1)
    pass

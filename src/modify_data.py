def to_binary(data, columns):
    out_data = data.copy()
    out_data[:, columns] = (data[:, columns] != 0).astype(int)
    return out_data


def one_to_K(data, columns):
    pass


def normalize(data, columns=None):
    out_data = data.copy()
    aux = data[:, columns]
    out_data_mean = aux.mean(axis=0)
    out_data_std = aux.std(axis=0)
    out_data[:, columns] = (aux - out_data_mean) / out_data_std
    return out_data, out_data_mean, out_data_std

import pickle as pkl

def save(obj, path, protocol = pkl.DEFAULT_PROTOCOL, verbose = False):
    if verbose:
        print("Saving file on path: {}".format(str(path)))
    try:
        file = open(str(path), 'wb')
        pkl.dump(obj, file, protocol = pkl.DEFAULT_PROTOCOL)
        file.close()

        if verbose:
            print('Saved.')

    except pkl.PickleError:
        print('An Error occured during saving.')


def load(path, verbose = False):
    if verbose:
        print("Loading file from path: {}".format(str(path)))

    try:
        file = open(str(path), 'rb')
        obj = pkl.load(file)
        file.close()

        if verbose:
            print('Loaded.')
        
        return obj

    except pkl.PickleError:
        print('An Error occured during saving.')

    
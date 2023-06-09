import sys

sys.path.append('../loader')
from .unaligned_data_loader import*
from .svhn import*
from .mnist import*
from .usps import*
from .gtsrb import*
from .synth_traffic import*

from .amazon import*
from .webcam import*

def return_dataset(data, scale=False, amazon=False, all_use='no'):
    if data == 'svhn':
        train_image, train_label, \
        test_image, test_label = load_svhn()
    if data == 'mnist':
        train_image, train_label, \
        test_image, test_label = load_mnist()#scale=scale, usps=usps, all_use=all_use)
        print(train_image.shape)
    if data == 'usps':
        train_image, train_label, \
        test_image, test_label = load_usps()
    if data == 'synth':
        train_image, train_label, \
        test_image, test_label = load_syntraffic()
    if data == 'gtsrb':
        train_image, train_label, \
        test_image, test_label = load_gtsrb()
    if data == 'amazon':
        train_image, train_label, \
        test_image, test_label = load_amazon(all_use=all_use)
    if data == 'webcam':
        train_image, train_label, \
        test_image, test_label = load_webcam(scale=scale,amazon=amazon, all_use=all_use)


    return train_image, train_label, test_image, test_label


def dataset_read(source, target, batch_size, scale=False, all_use='no'):
    S = {}
    S_test = {}
    T = {}
    T_test = {}
    amazon = False
    if source == 'amazon' or target == 'amazon':
        amazon = True

    train_source, s_label_train, test_source, s_label_test = return_dataset(source, scale=scale,amazon=amazon, all_use=all_use)
    train_target, t_label_train, test_target, t_label_test = return_dataset(target, scale=scale, amazon=amazon,all_use=all_use)


    S['imgs'] = train_source
    S['labels'] = s_label_train
    T['imgs'] = train_target
    T['labels'] = t_label_train

    # input target samples for both 
    S_test['imgs'] = test_target
    S_test['labels'] = t_label_test
    T_test['imgs'] = test_target
    T_test['labels'] = t_label_test
    scale = 40 if source == 'synth' else 28 if source == 'usps' or target == 'usps'else 28 if source == 'amazon' or target == 'amazon' else 32
    train_loader = UnalignedDataLoader()
    train_loader.initialize(S, T, batch_size, batch_size, scale=scale)
    dataset = train_loader.load_data()
    test_loader = UnalignedDataLoader()
    test_loader.initialize(S_test, T_test, batch_size, batch_size, scale=scale)
    dataset_test = test_loader.load_data()
    return dataset, dataset_test       


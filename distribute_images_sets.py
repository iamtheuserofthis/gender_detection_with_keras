import os, shutil


def generate_indices(train_size, test_size, validation_size):
    return (0, train_size), \
           (train_size, train_size + test_size), \
           (train_size + test_size, train_size + test_size + validation_size)


if __name__ == '__main__':
    path = input('Enter the path root dir:')
    ROOT_PATH = path
    DATASET_PATH = os.path.join(ROOT_PATH, 'dataset2')

    print('files at location: ', os.listdir(DATASET_PATH))
    path_dsets = os.listdir(DATASET_PATH)
    class_locs = {}
    class_count = {}
    for i, dataset in enumerate(path_dsets):
        temp_name = os.path.join(DATASET_PATH, dataset)
        print(dataset)
        if not os.path.isdir(temp_name):
            raise NotADirectoryError('%s: Not A Directory' % temp_name)

        dir_size = len(os.listdir(temp_name))
        print('Dir:%s Size:%s' % (temp_name, dir_size))
        class_locs[dataset] = temp_name
        class_count[dataset] = dir_size

    print(class_locs, class_count)

    train_size, test_size, validation_size = list(
        map(lambda x: int(x), input('train_size,test_size,validation_size: ').split(','))
    )

    total_set_size = sum([train_size, test_size, validation_size])
    if total_set_size > min(list(class_count.values())):
        raise ValueError(
            'Total Size: %s is greater than total_images: %s' % (total_set_size, min(list(class_count.values()))))

    print('Remaining Images: %s' % (min(list(class_count.values())) - total_set_size))
    TRAIN_DIR = os.path.join(ROOT_PATH, 'Train')
    #os.makedirs(TRAIN_DIR)
    TEST_DIR = os.path.join(ROOT_PATH, 'Test')
    #os.makedirs(TEST_DIR)
    VALIDATION_DIR = os.path.join(ROOT_PATH, 'Validation')
    #os.makedirs(VALIDATION_DIR)
    (train_start, train_end), (test_start, test_end), (valid_start, valid_end) = generate_indices(train_size, test_size,
                                                                                                  validation_size)
    source_list = {k: os.listdir(v) for k, v in class_locs.items()}
    for datadir, (start_index, end_index) in zip([TRAIN_DIR, TEST_DIR, VALIDATION_DIR],
                                                 [(train_start, train_end), (test_start, test_end),
                                                  (valid_start, valid_end)]):
        for class_name in class_locs:
            target_dir_name = os.path.join(datadir, class_name)
            #os.makedirs(target_dir_name)
            source_files_loc = os.listdir(class_locs[class_name])
            source_imgs_path = os.path.join(DATASET_PATH,class_name)
            for img_name in source_list[class_name][start_index:end_index]:
                shutil.copy(os.path.join(source_imgs_path,img_name),target_dir_name)

def evalCorruptDataset(logger, fn_test_corrupt, args_test_corrupt):
    """
    The wrapper helps to repeat the original testing function on all corrupted test sets.
    It also helps to compute metrics.
    :param model: model
    :param fn_test_corrupt: original evaluation function, returns a dict of metrics, e.g., {'acc': 0.93}
    :param args_test_corrupt: a dict of arguments to fn_test_corrupt, e.g., {'test_loader': loader}
    :return:
    """
    corruptions = [
        'clean',
        'scale',
        'jitter',
        'rotate',
        'dropout_global',
        'dropout_local',
        'add_global',
        'add_local',
    ]
    DGCNN_OA = {
        'clean': 0.926,
        'scale': 0.906,
        'jitter': 0.684,
        'rotate': 0.785,
        'dropout_global': 0.752,
        'dropout_local': 0.793,
        'add_global': 0.705,
        'add_local': 0.725
    }
    DGCNN_OA_OBJ = {
        'clean': 0.397,
        'scale': 0.370,
        'jitter': 0.217,
        'rotate': 0.332,
        'dropout_global': 0.273,
        'dropout_local': 0.261,
        'add_global': 0.213,
        'add_local': 0.244
    }
    DGCNN_OA_scan = {
        'clean': 0.858,
        'scale': 0.578,
        'jitter': 0.456,
        'rotate': 0.733,
        'dropout_global': 0.622,
        'dropout_local': 0.697,
        'add_global': 0.540,
        'add_local': 0.773
    }

    if args_test_corrupt.dataset == 'Objaverse-C':
        OA = DGCNN_OA_OBJ
    elif args_test_corrupt.dataset == 'ModelNet-C':
        OA = DGCNN_OA
    elif args_test_corrupt.dataset == 'ScanObjectNN-C':
        OA = DGCNN_OA_scan
 
    OA_clean = None
    perf_all = {'OA': [], 'CE': [], 'RCE': []}

    for corruption_type in corruptions:
        perf_corrupt = {'OA': []}
        for level in range(5):
            if corruption_type == 'clean':
                split = "clean"
            else:
                split = corruption_type + '_' + str(level)
            test_perf = fn_test_corrupt(split=split)
            if not isinstance(test_perf, dict):
                test_perf = {'acc': test_perf}
            perf_corrupt['OA'].append(test_perf['acc'])
            test_perf['corruption'] = corruption_type
            if corruption_type != 'clean':
                test_perf['level'] = level
            logger.info(test_perf)

            if corruption_type == 'clean':
                OA_clean = round(test_perf['acc'], 3)
                break

        for k in perf_corrupt:
            perf_corrupt[k] = sum(perf_corrupt[k]) / len(perf_corrupt[k])
            perf_corrupt[k] = round(perf_corrupt[k], 3)

        if corruption_type != 'clean':

            perf_corrupt['CE'] = (1 - perf_corrupt['OA']) / (1 - OA[corruption_type])
            perf_corrupt['RCE'] = (OA_clean - perf_corrupt['OA']) / (OA['clean'] - OA[corruption_type])
            for k in perf_all:
                perf_corrupt[k] = round(perf_corrupt[k], 3)
                perf_all[k].append(perf_corrupt[k])
        perf_corrupt['corruption'] = corruption_type
        perf_corrupt['level'] = 'Overall'
        logger.info(perf_corrupt)

    for k in perf_all:
        perf_all[k] = sum(perf_all[k]) / len(perf_all[k])
        perf_all[k] = round(perf_all[k], 3)
    perf_all['mCE'] = perf_all.pop('CE')
    perf_all['RmCE'] = perf_all.pop('RCE')
    perf_all['mOA'] = perf_all.pop('OA')
    logger.info(perf_all)

def evalModelNet40C(logger, fn_test_corrupt):
    """
    Evaluate model on ModelNet40-C with various corruptions and levels.
    :param model: the model to evaluate
    :param fn_test_corrupt: a function like test_one_epoch(split=..., model=..., ...)
    :param args_test_corrupt: a dict of other args needed by fn_test_corrupt
    :param io: optional logger with io.cprint()
    """
    corruptions = [
        'original', 'background', 'cutout', 'density', 'density_inc',
        'distortion', 'distortion_rbf', 'distortion_rbf_inv', 'gaussian',
        'impulse', 'lidar', 'occlusion', 'rotation', 'shear', 'uniform', 'upsampling'
    ]

    perf_all = {}  # corruption_name -> avg_error
    clean_error = None

    for corruption in corruptions:
        errors = []
        for level in range(1, 6):
            if corruption == 'original':
                split = 'original'
            else:
                split = f'{corruption}_{level}'

            test_perf = fn_test_corrupt(split=split)
            if not isinstance(test_perf, dict):
                test_perf = {'acc': test_perf}
            acc = test_perf['acc']
            error = 1.0 - acc
            errors.append(error)

            test_perf['error'] = round(error, 3)
            test_perf['corruption'] = corruption
            test_perf['level'] = level

            logger.info(test_perf)

            if corruption == 'original':
                clean_error = round(error, 3)
                break  

        avg_error = round(sum(errors) / len(errors), 3)
        perf_all[corruption] = avg_error

        if corruption != 'original':
            result_line = {
                'corruption': corruption,
                'level': 'Overall',
                'avg_error': avg_error
            }
            logger.info(result_line)

    corruption_errors = [v for k, v in perf_all.items() if k != 'original']
    avg_corruption_error = round(sum(corruption_errors) / len(corruption_errors), 3)

    summary = {
        'clean_error': clean_error,
        'avg_corruption_error': avg_corruption_error
    }
    logger.info(summary)

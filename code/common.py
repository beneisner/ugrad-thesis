import pandas
import PIL.Image
from cStringIO import StringIO
import IPython.display
import numpy as np

import trace.common as com
import trace.sampling as samp
import trace.train as train
import trace.train.hooks as hooks
import trace.evaluation as eva

def showarray(a, fmt='png'):
    a = np.uint8(a)
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    IPython.display.display(IPython.display.Image(data=f.getvalue()))

def print_and_save_metrics(model_name, df, p_error, r_full, r_merge, r_split):
    print(model_name)
    print('Pixel Error: %.6f' % p_error)
    print('Rand - Full: %.6f' % r_full)
    print('Rand - Merge: %.6f' % r_merge)
    print('Rand - Split: %.6f' % r_split)
    df.loc[model_name] = [p_error, r_full, r_merge, r_split]
    
def get_metrics(pipeline, classifier, inputs, labels, targets):
    preds = classifier.predict(inputs, pipeline.inference_params, mirror_inputs=True)
    binary_preds = np.round(preds)
    pixel_error = np.mean(np.absolute(binary_preds - targets))
    scores = eva.rand_error_from_prediction(labels[0, :, :, :, 0],
                                            preds[0],
                                            pred_type=pipeline.model_arch.output_mode)
    return preds, pixel_error, scores['Rand F-Score Full'], scores['Rand F-Score Merge'], scores['Rand F-Score Split']

def load_classifier(pipeline, run_name):
    train_params = pipeline.training_params

    # Create model
    arch = pipeline.model_arch
    model_const = pipeline.model_constructor
    model = model_const(arch)

    # Determine the input size to be sampled from the dataset
    sample_shape = np.asarray(train_params.patch_shape) + np.asarray(arch.fov_shape) - 1

    # Create the dataset sampler
    dataset = pipeline.dataset_constructor(pipeline.data_path)
    dset_sampler = samp.EMDatasetSampler(dataset, sample_shape=sample_shape, batch_size=train_params.batch_size,
                                         augmentation_config=pipeline.augmentation_config,
                                         label_output_type=arch.output_mode)

    # Define results folder
    ckpt_folder = pipeline.data_path + 'results/' + model.model_name + '/run-' + run_name + '/'

    # Create and restore the classifier
    classifier = train.Learner(model, ckpt_folder)
    classifier.restore()
    
    return classifier, dset_sampler

def run_full_training(pipeline, run_name):
    arch = pipeline.model_arch
    train_params = pipeline.training_params
    
    model_const = pipeline.model_constructor
    model = model_const(arch)
    
    # Determine the input size to be sampled from the dataset
    sample_shape = np.asarray(train_params.patch_shape) + np.asarray(arch.fov_shape) - 1

    # Construct the dataset sampler
    dataset = pipeline.dataset_constructor(pipeline.data_path)
    dset_sampler = samp.EMDatasetSampler(dataset,
                                         sample_shape=sample_shape,
                                         batch_size=train_params.batch_size,
                                         augmentation_config=pipeline.augmentation_config,
                                         label_output_type=arch.output_mode)

    ckpt_folder = pipeline.data_path + 'results/' + model.model_name + '/run-' + run_name + '/'

    classifier = train.Learner(model, ckpt_folder)

    hooks_list = [
        hooks.LossHook(50, model),
        hooks.ModelSaverHook(500, ckpt_folder),
        hooks.ValidationHook(100, dset_sampler, model, pipeline.data_path, arch.output_mode, pipeline.inference_params),
        hooks.ImageVisualizationHook(2000, model),
        # hooks.HistogramHook(100, model),
        # hooks.LayerVisualizationHook(500, model),
    ]

    # Train the model
    print('Training for %d iterations' % train_params.n_iterations)
    classifier.train(train_params, dset_sampler, hooks_list)
    
    return classifier, dset_sampler
import os
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.msdatasets.audio.asr_dataset import ASRDataset
import argparse


def modelscope_finetune(params):
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir, exist_ok=True)
    # dataset split ["train", "validation"]
    ds_dict = ASRDataset.load(params.data_path, namespace='speech_asr')
    kwargs = dict(
        model=params.model,
        data_dir=ds_dict,
        dataset_type=params.dataset_type,
        work_dir=params.output_dir,
        batch_bins=params.batch_bins,
        max_epoch=params.max_epoch,
        lr=params.lr)
    trainer = build_trainer(Trainers.speech_asr_trainer, default_args=kwargs)
    trainer.train()


if __name__ == '__main__':
    from funasr.utils.modelscope_param import modelscope_args

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model', type=str, default="damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        help='Name of the model to be downloaded.')
    parser.add_argument('output_dir', type=str, default='./checkpoint', help='The output dir for the checkpoint')
    parser.add_argument('data_path', type=str, default='speech_asr_aishell1_trainsets',
                        help='The data path, which can be a local path or a data id in datahub')
    parser.add_argument('dataset_type', type=str, default='small', help='dataset size, can be `small`(<=1000 rows) '
                                                                        'or `large`(>1000 rows)')
    parser.add_argument('batch_bins', type=int, default=2000, help='Batch size, if dataset_type is small,'
                                                                   'the unit of batch_bins is '
                                                                   'the feature frame numbers of fbank, else '
                                                                   'the unit of batch_bins is mille second')
    parser.add_argument('max_epoch', type=int, default=50, help='The max epoch number')
    parser.add_argument('lr', type=float, default=0.00005, help='The learning rate')
    args, _ = parser.parse_known_args()

    params = modelscope_args(model=args.model)
    params.output_dir = args.output_dir
    params.data_path = args.data_path
    params.dataset_type = args.dataset_type
    params.batch_bins = args.batch_bins
    params.max_epoch = args.max_epoch
    params.lr = args.lr

    modelscope_finetune(params)

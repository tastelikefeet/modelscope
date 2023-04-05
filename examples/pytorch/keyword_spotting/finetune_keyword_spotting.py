# coding = utf-8

import os
from modelscope.utils.hub import read_config
from modelscope.utils.hub import snapshot_download
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer

def main():
    # s1
    work_dir = './test_kws_training'

    # s2
    model_id = 'damo/speech_charctc_kws_phone-xiaoyun'
    model_dir = snapshot_download(model_id)
    configs = read_config(model_id)
    config_file = os.path.join(work_dir, 'config.json')
    configs.dump(config_file)

    # s3
    kwargs = dict(
        model=model_id,
        work_dir=work_dir,
        cfg_file=config_file,
    )
    trainer = build_trainer(
        Trainers.speech_kws_fsmn_char_ctc_nearfield, default_args=kwargs)

    # s4
    train_scp = './example_kws/train_wav.scp'
    cv_scp = './example_kws/cv_wav.scp'
    trans_file = './example_kws/merge_trans.txt'
    kwargs = dict(
        train_data=train_scp,
        cv_data=cv_scp,
        trans_data=trans_file
    )
    trainer.train(**kwargs)

    # s5
    keywords = '小云小云'
    test_dir = os.path.join(work_dir, 'test_dir')
    test_scp = './example_kws/test_wav.scp'
    trans_file = './example_kws/test_trans.txt'
    rank = int(os.environ['RANK'])
    if rank == 0:
        kwargs = dict(
            test_dir=test_dir,
            test_data=test_scp,
            trans_data=trans_file,
            gpu=0,
            keywords=keywords,
            batch_size=256,
            )
        trainer.evaluate(None, None, **kwargs)

if __name__ == '__main__':
    main()

import os.path
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', '..', '..','src')
if src_path not in sys.path:
    sys.path.append(src_path)

from easyeditor import EditTrainer, MENDTrainingHparams, ZsreDataset

training_hparams = MENDTrainingHparams.from_hparams('../../../src/hparams/TRAINING/MEND/llama2-7b-hf-chat-kenji.yaml')


train_ds = ZsreDataset('../../../data/train_data/data/zsre/zsre_mend_train_10000.json', config=training_hparams)
eval_ds = ZsreDataset('../../../data/train_data/data/zsre/zsre_mend_eval.json', config=training_hparams)
trainer = EditTrainer(
    config=training_hparams,
    train_set=train_ds,
    val_set=eval_ds
)
trainer.run()
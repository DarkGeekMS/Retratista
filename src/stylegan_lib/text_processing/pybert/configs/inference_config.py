
from pathlib import Path
import os
PYBERT_BASE_DIR = Path('text_processing/pybert')
BASE_DIR = Path('text_processing')

config = {
    'checkpoint_dir': PYBERT_BASE_DIR / "output/checkpoints",
    'bert_vocab_path': PYBERT_BASE_DIR / 'pretrain/bert/base-uncased/bert_vocab.txt',
    'max_attributes_path': BASE_DIR / 'attributes_max.pkl'
}

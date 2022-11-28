import os
import argparse
import torch

here = os.path.dirname(os.path.abspath(__file__))



default_pretrained_model_path = os.path.join(here, './pretrained_models/bert-base-english')
default_train_file = os.path.join(here, './datas/train_data')
default_test_file = os.path.join(here, './datas/test_data')
default_output_dir = os.path.join(here, './saved_models')
default_log_dir = os.path.join(default_output_dir, 'runs')
default_target_file = os.path.join(here, './datas/label.vocab')
default_model_file = os.path.join(default_output_dir, 'model.bin')
default_checkpoint_file = os.path.join(default_output_dir, 'checkpoint.json')

parser = argparse.ArgumentParser()

parser.add_argument("--pretrained_model_path", type=str, default=default_pretrained_model_path)
parser.add_argument("--model_name", type=str, default="bert-base-english")
parser.add_argument("--train_file", type=str, default=default_train_file)
parser.add_argument("--test_file", type=str, default=default_test_file)
parser.add_argument("--output_dir", type=str, default=default_output_dir)
parser.add_argument("--log_dir", type=str, default=default_log_dir)
parser.add_argument("--target_file", type=str, default=default_target_file)
parser.add_argument("--model_file", type=str, default=default_model_file)
parser.add_argument("--checkpoint_file", type=str, default=default_checkpoint_file)

# model
parser.add_argument('--embedding_dim', type=int, default=768, required=False, help='embedding_dim')
parser.add_argument('--hidden_size', type=int, default=300)
parser.add_argument('--dropout', type=float, default=0.5, required=False, help='dropout')

parser.add_argument('--device', type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--max_len", type=int, default=128)
parser.add_argument("--vocab_size", type=str,default=1500000)
parser.add_argument("--train_batch_size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--learning_rate_decay", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=0)

hparams = parser.parse_args()
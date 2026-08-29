import sys 
with open('train_roberta.py', 'r', encoding='utf-8') as f: lines = f.readlines() 
lines.insert(12, 'from data_utils import get_unified_splits\n') 
split_start = -1 
split_end = -1 

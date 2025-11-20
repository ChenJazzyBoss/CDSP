import pandas as pd
import torch
from tqdm import tqdm
from llm2vec import LLM2Vec
import os

# 对应直接生成大模型的嵌入

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# l2v参数
local_model_path = "/home/admin1/code/BitBlitz/PAD/llm2vec/Meta-Llama-3-8B-Instruct"
peft_model_name_or_path = "/home/admin1/code/BitBlitz/PAD/llm2vec/Llama/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"

# 存储变量以便于修改
dataset_name = "Prime_Pantry"
meta_data = "metadata_Prime_Pantry.csv"

l2v = LLM2Vec.from_pretrained(
    local_model_path,
    peft_model_name_or_path,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)
d = pd.read_csv(f'./dataset/{dataset_name}/{meta_data}')

item_word_embs = [torch.zeros(4096)]
for i in tqdm(range(8)):
    strlist = d.iloc[1000 * i:1000 * (i + 1), 1].tolist()
    item_feature = l2v.encode(strlist)
    item_word_embs.extend(item_feature)

strlist = d.iloc[8000:, 1].tolist()
item_feature = l2v.encode(strlist)
item_word_embs.extend(item_feature)

a = torch.stack(tensors=item_word_embs, dim=0)
# torch.save(a '/dataset/Amazon_Clothing_Shoes_and_Jewelry_llm2vec.pt')
torch.save(a, f'./dataset/{dataset_name}/{dataset_name}_llm2vec.pt')

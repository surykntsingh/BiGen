import json
import spacy
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize

import torch

def read_json_file(json_path):
    with open(json_path) as f:
        d = json.load(f)
    return d

def get_texts(json_path):
    reports =  read_json_file(json_path)
    text_list = [r['report'] for r in reports['train']]

    nlp = spacy.load("en_core_web_sm")
    sentences = []
    for t in text_list:
        sents = [sent.text for sent in nlp(t).sents]
        sentences.extend(sents)

    return sentences

def batch_list(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def get_text_embeddings(texts, model_cfg, checkpoint_path, batch_size=64):
    model, preprocess = create_model_from_pretrained(model_cfg, checkpoint_path)
    tokenizer = get_tokenizer()  # load tokenizer
    all_embeddings = []
    for batch in batch_list(texts, batch_size):
        text_tokens = tokenize(texts=batch, tokenizer=tokenizer)  # tokenize the text
        with torch.inference_mode():
            text_embs = model.encode_text(text_tokens)
            all_embeddings.append(text_embs.cpu())

    final_embeddings = torch.cat(all_embeddings, dim=0)
    print(final_embeddings.shape)
    return final_embeddings

def save_text_embeddings(text_embs, save_path):
    print(f'saving text embeddings at: {save_path}/memory_short.pt')
    torch.save(text_embs, f'{save_path}/memory_short.pt')

def save_sent_texts(texts, save_path):
    print(f'saving texts at: {save_path}/memory_short.pt')
    with open(f'{save_path}/short_texts.json', 'w') as f:
        json.dump(texts, f, indent=4)


if __name__=="__main__":
    json_path = '/lustre/nvwulf/projects/PrasannaGroup-nvwulf/surya/datasets/TCGA_processed/brca_v2/tcga_brca_reports_splits.json'
    save_path = '/lustre/nvwulf/projects/PrasannaGroup-nvwulf/surya/datasets/TCGA_processed/conch'

    model_cfg = 'conch_ViT-B-16'
    checkpoint_path = '../../CONCH/checkpoints/conch/pytorch_model.bin'

    texts = get_texts(json_path)
    print(f'num texts: {len(texts)}')
    text_embs =  get_text_embeddings(texts, model_cfg, checkpoint_path)
    save_text_embeddings(text_embs, save_path)
    save_sent_texts(texts, save_path)










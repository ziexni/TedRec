import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from recbole.data.dataset import SequentialDataset


class TedRecDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)
        self.plm_size = config['plm_size']
        self.plm_suffix = config['plm_suffix']
        plm_embedding_weight = self.load_plm_embedding()
        self.plm_embedding = self.weight2emb(plm_embedding_weight)
    
    def load_plm_embedding(self):
        """
        원본: .feat1CLS 파일에서 로드
        수정: .npy 파일에서 로드 (우리 데이터)
        """
        feat_path = osp.join(self.config['data_path'], f'{self.dataset_name}.{self.plm_suffix}.npy')
        loaded_feat = np.load(feat_path)  # (num_items, plm_size)
        
        # RecBole item_id mapping (0 = padding)
        mapped_feat = np.zeros((self.item_num, self.plm_size))
        for i, token in enumerate(self.field2id_token['item_id']):
            if token == '[PAD]': 
                continue
            item_id = int(token)
            if item_id < len(loaded_feat):
                mapped_feat[i] = loaded_feat[item_id]
        
        return mapped_feat
    
    def weight2emb(self, weight):
        plm_embedding = nn.Embedding(self.item_num, self.plm_size, padding_idx=0)
        plm_embedding.weight.requires_grad = False
        plm_embedding.weight.data.copy_(torch.from_numpy(weight))
        return plm_embedding
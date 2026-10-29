import pandas as pd
import faiss
import torch
from open_clip import create_model_from_pretrained, get_tokenizer

class SearchSigLIP():

    def __init__(self, index_path, metadata_path):

        # 1. Initialise Index
        print(f'Loading index from PATH={index_path}')
        self.index_path = index_path
        self.init_index()
        print('[DONE]')

        # 2. Initialise Metadata
        print(f'Loading metadata from PATH={metadata_path}')
        self.metadata_path = metadata_path
        self.metadata_df = pd.read_parquet(self.metadata_path)
        print('[DONE]')

        # 3. Initialise Text Encoder
        self.init_model()

    def init_index(self):
        self.cpu_index = faiss.read_index(self.index_path)
        # GPU when the Space has one (L4 grant); otherwise the same index on CPU: slower search, same results
        self.device = 'cuda' if torch.cuda.is_available() and faiss.get_num_gpus() > 0 else 'cpu'
        if self.device == 'cuda':
            res = faiss.StandardGpuResources()
            cloner_options = faiss.GpuClonerOptions()
            cloner_options.useFloat16LookupTables = True
            self.index = faiss.index_cpu_to_gpu(res, 0, self.cpu_index, cloner_options)
            self.index.nprobe = 32  # Higher = more accurate, slower
        else:
            self.index = self.cpu_index
            faiss.ParameterSpace().set_index_parameter(self.index, 'nprobe', 32)  # same as on GPU
        print(f'Text search on {self.device}')

    def init_model(self):
        self.model, self.preprocess = create_model_from_pretrained('hf-hub:timm/ViT-SO400M-14-SigLIP-384')
        self.model.eval()
        self.tokenizer = get_tokenizer('hf-hub:timm/ViT-SO400M-14-SigLIP')

    def encode_text(self, text, device=None):
        device = device or self.device
        self.model.to(device)
        with torch.no_grad():
            text = self.tokenizer([text], context_length=self.model.context_length)
            return self.model.encode_text(text.to(device))

    def search_with_grid(self, query_vec, k=5):
        # Prepare query
        if isinstance(query_vec, torch.Tensor):
            query_vec = query_vec.cpu().squeeze().numpy()
            
        query_vec = query_vec.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_vec)
        
        # Search
        distances, indices = self.index.search(query_vec, k)
        
        # Flatten results
        ids = indices[0]
        scores = distances[0]
        
        results = []
        
        # Batch lookup in pandas (Faster than looping)
        # We ignore -1 (which happens if k > total vectors, unlikely here)
        valid_mask = ids != -1
        valid_ids = ids[valid_mask]
        valid_scores = scores[valid_mask]
        
        if len(valid_ids) > 0:
            # MAGIC LINE: Direct lookup by integer index
            matches = self.metadata_df.iloc[valid_ids].copy()
            matches['score'] = valid_scores
            
            # Convert to list of dicts for easy usage
            results = matches.to_dict(orient='records')
            
        return results

    def faiss(self, text, k=1): # k - number of neighbours

        # 1. Compute query
        q = self.encode_text(text)

        # 2. Find Hits
        results = self.search_with_grid(q, k=k)

        return results
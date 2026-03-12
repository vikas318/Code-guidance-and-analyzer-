import os
import json
import warnings
import torch
from sentence_transformers import SentenceTransformer, util
warnings.filterwarnings("ignore")

class SmartSearchEngine:
    def __init__(self):
        print("[SearchEngine] Booting up semantic indexer...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.data_store = {"sandbox": [], "dsa": []}
        self.corpus_embeddings = None
        self.flat_file_data = []

    def index_directory(self, directory_path, category):
        print(f"[SearchEngine] Indexing '{category}' files from '{directory_path}'...")
        file_list = []
        if not os.path.exists(directory_path):
            print(f"[SearchEngine] Warning: Directory '{directory_path}' not found.")
        else:
            for root, _, files in os.walk(directory_path):
                for file in files:
                    filepath = os.path.join(root, file)
                    try:
                        if category == "sandbox" and file.endswith(".py"):
                            with open(filepath, 'r', encoding='utf-8') as f:
                                content = f.read().strip()
                                if content:
                                    file_list.append({
                                        "path": file, 
                                        "content": content, 
                                        "snippet": content[:150] + "...",
                                        "category": category
                                    })
                        elif category == "dsa" and file.endswith(".json"):
                            with open(filepath, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                full_content = f"{data.get('title', '')} {data.get('description', '')} {data.get('actual_solution', '')}"
                                file_list.append({
                                    "path": data.get("title", file),
                                    "content": full_content, 
                                    "snippet": data.get("description", "")[:150] + "...",
                                    "category": category
                                })
                    except Exception as e:
                        pass
        self.data_store[category] = file_list
        self._rebuild_index()

    def _rebuild_index(self):
        self.flat_file_data = self.data_store["sandbox"] + self.data_store["dsa"]
        if not self.flat_file_data:
            self.corpus_embeddings = None
            return
        
        corpus_texts = [item["content"] for item in self.flat_file_data]
        self.corpus_embeddings = self.encoder.encode(corpus_texts, convert_to_tensor=True)

    def search(self, query, threshold=25.0, top_k=5):
        if self.corpus_embeddings is None or len(self.flat_file_data) == 0:
            return {"sandbox": [], "dsa": []}
        query_embedding = self.encoder.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
        sorted_indices = torch.argsort(cos_scores, descending=True)
        results = {"sandbox": [], "dsa": []}
        for idx in sorted_indices:
            score = cos_scores[idx].item() * 100
            if score < threshold:
                continue 
            item = self.flat_file_data[idx]
            cat = item["category"]
            if len(results[cat]) < top_k:
                results[cat].append({
                    "file_path": item["path"],
                    "match_percentage": round(score, 1),
                    "content_snippet": item["snippet"]
                })
        return results

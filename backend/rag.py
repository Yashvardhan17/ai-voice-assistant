from pathlib import Path
from functools import lru_cache
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
@lru_cache(maxsize=1)
def index():
    import faiss
    from sentence_transformers import SentenceTransformer
    texts=[];sources=[]
    for file in sorted((ROOT/'data'/'knowledge').glob('*.txt')):
        for paragraph in file.read_text(encoding='utf8').split('\n\n'):
            if paragraph.strip():texts.append(paragraph.strip());sources.append(file.name)
    encoder=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    vectors=np.asarray(encoder.encode(texts,normalize_embeddings=True),dtype='float32')
    db=faiss.IndexFlatIP(vectors.shape[1]);db.add(vectors)
    return encoder,db,texts,sources

def retrieve(query:str,k:int=2):
    encoder,db,texts,sources=index()
    vector=np.asarray(encoder.encode([query],normalize_embeddings=True),dtype='float32')
    scores,ids=db.search(vector,min(k,len(texts)))
    return [{'source':sources[i],'text':texts[i],'score':round(float(score),3)} for score,i in zip(scores[0],ids[0]) if i>=0 and score>=0.30]

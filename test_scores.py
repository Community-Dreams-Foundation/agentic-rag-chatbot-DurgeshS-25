from app.embed import load_index
from collections import Counter

index, meta = load_index()
print(dict(Counter(c['filename'] for c in meta)))


Some examples of using Milvus/Langchain/HuggingFace to do cool things.

#### embedding.py
A simple example of using Milvus and the sentence transformer `all-MiniLM-L6-v2` to do vector embeddings.

#### Config env:
```
$ python -m venv env
$ pip install -r embedding_reqs.txt
```

#### Run
```
$ python embedding.py Odyssey.txt
```

#### multi_query.py
Use Langchain and Milvus to do multi-query RAG.

#### Config env:
```
$ python -m venv env
$ pip install -r multi_query_reqs.txt
```

#### Run
```
$ python embedding.py Odyssey.txt
```

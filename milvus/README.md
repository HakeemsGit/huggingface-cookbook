Some examples of using Milvus/Langchain/HuggingFace to do cool things.

#### aws_pipeline.py
Process product reviews and images from S3 using OpenAI (text) and CLIP (image) embeddings. Store in Milvus for vector search.

#### Config env:
```
$ python -m venv env
$ pip install -r aws_pipeline_reqs.txt
```

Required environment variables:
```
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
OPENAI_API_KEY=your_key
```

S3 bucket structure:
```
product-dataset/
  p_1/
    review.txt
    image.png
  p_2/
    review.txt
    image.png
  ...
```

#### Run:
```
$ python aws_pipeline.py
```

#### embedding.py
A simple example of using Milvus and the sentence transformer `all-MiniLM-L6-v2` to do vector embeddings.

#### Config env:
```
$ python -m venv env
$ pip install -r embedding_reqs.txt
```

#### Run
```
$ python embedding.py books/Odyssey.txt
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
$ python embedding.py books/Odyssey.txt
```


#### insertion_comparison.py
Compare the speed of single vs. bulk insertions for embeddings.

#### Config env:
```
$ python -m venv env
$ pip install -r insertion_comparison_reqs.txt
```

#### Run
```
$ python insertion_comparison.py books/Odyssey.txt
```

Some examples of using llama to do cool things

#### llm_judge.py
A simple example of training llamma to performa as a judge.

#### Config env:
```
$ mkdir -p models
$ wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf -O models/llama-2-7b-chat.gguf
$ python -m venv env
$ pip install -r llm_judge_reqs.txt
```

#### Run
```
$ python llm_judge.py
```

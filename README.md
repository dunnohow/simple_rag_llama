# Simple RAG system using pdf file as back data
- model repository : https://huggingface.co/YorkieOH10/Meta-Llama-3.1-8B-Instruct-Q8_0-GGUF/tree/main
- llama2 paper url : https://arxiv.org/abs/2307.09288

### sample request format

```commandline
# Sample question formats
curl -X POST http://127.0.0.1:5000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "what is llama"}'
     
curl -X POST http://127.0.0.1:5000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "how many parameters llama2 model have"}'
     
# Getting responses
curl -X GET http://127.0.0.1:5000/responses 
```
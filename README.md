# Compute the arithmetic intensity of phi-1 model

This is a case study of how to compute the arithmetic intensity of phi-1 model for training and inference

## Model architecture

```
PhiForCausalLM(
  (model): PhiModel(
    (embed_tokens): Embedding(51200, 2048)
    (embed_dropout): Dropout(p=0.0, inplace=False)
    (layers): ModuleList(
      (0-23): 24 x PhiDecoderLayer(
        (self_attn): PhiAttention(
          (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
          (k_proj): Linear(in_features=2048, out_features=2048, bias=True)
          (v_proj): Linear(in_features=2048, out_features=2048, bias=True)
          (dense): Linear(in_features=2048, out_features=2048, bias=True)
          (rotary_emb): PhiRotaryEmbedding()
        )
        (mlp): PhiMLP(
          (activation_fn): NewGELUActivation()
          (fc1): Linear(in_features=2048, out_features=8192, bias=True)
          (fc2): Linear(in_features=8192, out_features=2048, bias=True)
        )
        (input_layernorm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
        (resid_dropout): Dropout(p=0.0, inplace=False)
      )
    )
    (final_layernorm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=2048, out_features=51200, bias=True)
)
```


## Model config

```
{
  "_name_or_path": "microsoft/phi-1",
  "architectures": [
    "PhiForCausalLM"
  ],
  "auto_map": {
    "AutoConfig": "configuration_phi.PhiConfig",
    "AutoModelForCausalLM": "modeling_phi.PhiForCausalLM"
  },
  "attention_dropout": 0.0,
  "bos_token_id": null,
  "embd_pdrop": 0.0,
  "eos_token_id": null,
  "hidden_act": "gelu_new",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 8192,
  "layer_norm_eps": 1e-05,
  "max_position_embeddings": 2048,
  "model_type": "phi",
  "num_attention_heads": 32,
  "num_hidden_layers": 24,
  "num_key_value_heads": null,
  "partial_rotary_factor": 0.5,
  "qk_layernorm": false,
  "resid_pdrop": 0.0,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "float32",
  "transformers_version": "4.37.0",
  "use_cache": true,
  "vocab_size": 51200
}
```

##  Table of the model setting

|Symbol| Definition| Shape| value| FLOPS | IO | FLOPS:IO|
|------------|-----------|-------|-----|----------|---------|-----|
|**Scalars**|
|**Input Shape**|
|b| Batch size| 1|
|s| Sequence lenghth | 1|    |   |
|M| Size of SRAM|1|
|**Model Hyper-parameter**|
|n| Number of attention heads| 1| 32
|d|Hidden state size of one head| 1|64
|h|Hidden state size(h = n*d)|1|2048
|**Parameters**|
|W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub>| Projection for Q, K, V| (h, h)|(2048, 2048)
|W<sub>O</sub>| Projection for self-attention ouput| (h, h)|(2048, 2048)
|W1| First layer in the FFN|(h, 4h)|(2048, 8192)
|W2| Second layer in the FFN|(4h, h)|(8192, 2048)



## Table of self-attention

|Symbol| Definition| Shape| value| FLOPS | IO | FLOPS:IO|
|------------|-----------|-------|-----|----------|---------|-----|
|**Input**|
|X| Input for self attention| (b,s,h)|(b,s,2048)|
|**Self-Attention**|
|Q<sup>^^</sup>, K<sup>^^</sup>, V<sup>^^</sup>|XW<sub>Q</sub>, XW<sub>K</sub>, XW<sub>V</sub>| (b, s, h)| (b, s, 2048)|3*2bsh<sup>2</sup>|3(2bsh + h<sup>2</sup>)|
|Q<sup>^</sup>, K<sup>^</sup>, V<sup>^</sup>|Reshape Q<sup>^^</sup>, K<sup>^^</sup>, V<sup>^^</sup>|(b,s,n,d)|
|Q, K, V|Transpose Q<sup>^</sup>, K<sup>^</sup>, V<sup>^</sup>|(b,n,s,d)|
|K<sup>T</sup>|Transpose K|(b,n,d,s)|
|P|Softmax(QK<sup>T</sup>/sqrt(d))|(b,n,s,s)| (b,32,s,s)|3nbs<sup>2</sup>|2bsnd+bs<sup>2</sup>n|
|A<sup>^^</sup>|PV|(b,n,s,d)|(b,32,s,64)|2bs<sup>2</sup>nd| 2bsnd+bs<sup>2</sup>n|
|A<sup>^</sup>|Transpose A<sup>^^</sup>|(b,s,n,d)|
|A|Reshape A<sup>^</sup>|(b,s,h)|
|Y|AW<sub>O</sub>|(b,s,h)|(b,s,2048)| 2bsh<sup>2</sup>|2bsh+h<sup>2</sup>|
|**MLP**|
|Z|GELU(YW<sub>1</sub>)W<sub>2</sub>|(b,s,h)|(b,s,2048)|64bsh<sup>2</sup>|4bsh+16h<sup>2</sup>|
|**Input**|
|x| Input for self attention| (b,1,h)|
|K<sub>s</sub>, V<sub>S</sub>|Key/Value cache from past| (b,n,s,d)|
|**Self-Attention**|
|q<sup>^^</sup>, k<sup>^^</sup>, v<sup>^^</sup>|xW<sub>Q</sub>, xW<sub>K</sub>, xW<sub>V</sub>| (b, 1, h)| (b, 1, 2048)|3*2bh<sup>2</sup>|3(2bh + h<sup>2</sup>)|
|q<sup>^</sup>, k<sup>^</sup>, v<sup>^</sup>|Reshape q<sup>^^</sup>, k<sup>^^</sup>, v<sup>^^</sup>|(b,1,n,d)|
|q, k, v|Transpose q<sup>^</sup>, k<sup>^</sup>, v<sup>^</sup>|(b,n,1,d)|
|K, V|concat(K<sub>s</sub>,k), concat(V<sub>s</sub>, v)|(b,n,s+1,d)|
|K<sup>T</sup>|Transpose K|(b,n,d,s+1)|
|p|Softmax(qK<sup>T</sup>/sqrt(d))|(b,n,1,s+1)| (b,32,1,s+1)|3bsnd|bsn+bsnd+bnd|
|a<sup>^^</sup>|pV|(b,n,1,d)|(b,32,1,64)|2bsnd| bsn+bsnd+bnd|
|a<sup>^</sup>|Transpose A<sup>^^</sup>|(b,1,n,d)|(b,1,32,64)|
|a|Reshape A<sup>^</sup>|(b,1,h)|
|y|AW<sub>O</sub>|(b,1,h)|(b,1,2048)| 2bh<sup>2</sup>|2bh+h<sup>2</sup>|
|**MLP**|
|z|GELU(yW<sub>1</sub>)W<sub>2</sub>|(b,1,h)|(b,1,2048)|64bh<sup>2</sup>|4bh+16h<sup>2</sup>|

## With FlashAttention
Based on this implementation https://huggingface.co/microsoft/phi-1/blob/main/modeling_phi.py. Flash_attention is used after Q, K, V transpose.

|Symbol| Definition| Shape| value| FLOPS | IO | FLOPS:IO|
|------------|-----------|-------|-----|----------|---------|-----|
|**Input**|
|X| Input for self attention| (b,s,h)|(b,s,2048)|
|**Flash-Attention**|
|Q<sup>^^</sup>, K<sup>^^</sup>, V<sup>^^</sup>|XW<sub>Q</sub>, XW<sub>K</sub>, XW<sub>V</sub>| (b, s, h)| (b, s, 2048)|3*2bsh<sup>2</sup>|3(2bsh + h<sup>2</sup>)|
|Q<sup>^</sup>, K<sup>^</sup>, V<sup>^</sup>|Reshape Q<sup>^^</sup>, K<sup>^^</sup>, V<sup>^^</sup>|(b,s,n,d)|
|Q, K, V|Transpose Q<sup>^</sup>, K<sup>^</sup>, V<sup>^</sup>|(b,n,s,d)|
|K<sup>T</sup>|Transpose K|(b,n,d,s)|
|A=flash_atttion(Q,K,V)|flash attention|(b, s, h)|(b, s, 2048)|3nbs<sup>2</sup> + 2bs<sup>2</sup>nd|s<sup>2</sup>d<sup>2</sup>/M
|Y|AW<sub>O</sub>|(b,s,h)|(b,s,2048)| 2bsh<sup>2</sup>|2bsh+h<sup>2</sup>|
|**MLP**|
|Z|GELU(YW<sub>1</sub>)W<sub>2</sub>|(b,s,h)|(b,s,2048)|64bsh<sup>2</sup>|4bsh+16h<sup>2</sup>|
|**Input**|
|x| Input for self attention| (b,1,h)|
|K<sub>s</sub>, V<sub>S</sub>|Key/Value cache from past| (b,n,s,d)|
|**Flash-Attention**|
|q<sup>^^</sup>, k<sup>^^</sup>, v<sup>^^</sup>|xW<sub>Q</sub>, xW<sub>K</sub>, xW<sub>V</sub>| (b, 1, h)| (b, 1, 2048)|3*2bh<sup>2</sup>|3(2bh + h<sup>2</sup>)|
|q<sup>^</sup>, k<sup>^</sup>, v<sup>^</sup>|Reshape q<sup>^^</sup>, k<sup>^^</sup>, v<sup>^^</sup>|(b,1,n,d)|
|q, k, v|Transpose q<sup>^</sup>, k<sup>^</sup>, v<sup>^</sup>|(b,n,1,d)|
|K, V|concat(K<sub>s</sub>,k), concat(V<sub>s</sub>, v)|(b,n,s+1,d)|
|K<sup>T</sup>|Transpose K|(b,n,d,s+1)|
|a=flash_atttion(Q,K,V)|flash attention|(b, s, h)|(b, s, 2048)|5bsnd|s<sup>2</sup>d<sup>2</sup>/M
|y|aW<sub>O</sub>|(b,1,h)|(b,1,2048)| 2bh<sup>2</sup>|2bh+h<sup>2</sup>|
|**MLP**|
|z|GELU(yW<sub>1</sub>)W<sub>2</sub>|(b,1,h)|(b,1,2048)|64bh<sup>2</sup>|4bh+16h<sup>2</sup>|



## Result analysis
Please note that the tables above is only considering the forward passing. For the backward passing, the flash attention will recompute the S and P. The FLOPS may be higher than self-attention. However, if we denote all these numbers in Big O notation. The result will be consistent. 

If we compare these two tables. we know that all steps other than attention part are exactlly the same. So the 


## Reference
- https://arxiv.org/pdf/1706.03762.pdf
- https://arxiv.org/pdf/2205.14135.pdf
- https://kipp.ly/transformer-inference-arithmetic/#capacity
- https://www.adamcasson.com/posts/transformer-flops
- https://le.qun.ch/en/blog/2023/05/13/transformer-batching/







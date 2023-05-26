# mpttune: 4-Bit Finetuning of MPTs on a Consumer GPU

**mpttune** allows finetuning MPTs (e.g., mpt-7b-storywriter-4bit) on as little as one consumer-grade A100 40GB. 

Its features tiny and easy-to-use codebase.

One benefit of being able to finetune larger LLMs on one GPU is the ability to easily leverage data parallelism for large models.

Underneath the hood, **mpttune** implements the LoRA algorithm over an LLM compressed using the GPTQ algorithm, which requires implementing a backward pass for the quantized LLM.

**mpttune** can generate a 600-token epilogue when fed 9000 tokens from a book on A100 40GB for ~ 30 seconds using triton backend

```
$ tail ... $book

“She still retained her beauty. She was more than common tall, of
majestic presence, she had an exquisitely-modelled neck and bust, and
her hand was the delight of the sculptor. Her smile was distinguished
by its sweetness and her voice was rich and low. Her lofty brow, and
clear, thoughtful gaze  

----------------------------------------------------------------------


etained her beauty. She was more than common tall, of
majestic presence, she had an exquisitely-modelled neck and bust, and
her hand was the delight of the sculptor. Her smile was distinguished
by its sweetness and her voice was rich and low. Her lofty brow, and
clear, thoughtful gaze . 


EPILOGUE


$ mpttune generate --interactive --model mpt-7b-storywriter-4bit --weights mpt-7b-storywriter-4bit-128g.safetensors --max_new_tokens=600 --use_cache --do_sample --prompt "$book"

EPILOGUE
The Project Gutenberg eBook of A forgotten Prince of Wales, by Henry Curties  This eBook is for the use of anyone anywhere in the United States and most other parts of the world at no cost and with almost no restrictions whatsoever. You may copy it, give it away or re-use it under the terms of the Project Gutenberg License included with this eBook or online at www.gutenberg.org. If you are not located in the United States, you will have to check the laws of the country where you are located before using this eBook.  Title: A forgotten Prince of Wales  Author: Henry Curties  Release Date: May 19, 2023 [eBook #70795]  Language: English  Produced by: MWS and the Online Distributed Proofreading Team at              https://www.pgdp.net (This file was produced from images              generously made available by The Internet Archive/Canadian              Libraries)  *** START OF THE PROJECT GUTENBERG EBOOK A FORGOTTEN PRINCE OF WALES ***                                    A FORGOTTEN                             PRINCE OF WALES.                              [Illustration]                              [Illustration:             _National Portrait Gallery._      _Emery Walker._           FREDERICK, PRINCE OF WALES, AND HIS SISTERS AT KEW.]                                   A FORGOTTEN                             PRINCE OF WALES                                      BY                          CAPTAIN HENRY CURTIES                      Author of “When England Slept,”                                etc., etc.                                    LONDON                           EVERETT & CO., LTD.                      42 ESSEX STREET, STRAND, W.C.                             Dedicated by permission                                    to                    His Grace the Duke of Argyll, K.G.                                    CONTENTS                                                                       PAGE  CHAPTER I.   Which Seizes upon the Prince as he comes into the World               1   CHAPTER II.   The Falling in of a Great Legacy                                     12   CHAPTER III.   The Prince at the Age of Nine                                        18   CHAPTER IV.   In which England gets a new King and Queen                           25   CHAPTER V.   A Double Event which did not come off                                41   CHAPTER VI.   The Prince and the London of 1728                                    50   CHAPTER VII.   Peter Wentworth’s Letters on the Prince’s Life                       60   CHAPTER VIII.   The Prince’s Embarrassments                                          73   CHAPTER IX.   The Duchess of Marlborough Throws for a Big Stake                    83   CHAPTER X.   The Beautiful Vanilla                                                92   CHAPTER XI.   The Prince Asserts Himself                                          104   CHAPTER XII.   A Child Bride                                                       121   CHAPTER XIII.   The Nuptials                                                        141   CHAPTER XIV.   Lady Archibald                                                      147   CHAPTER XV.   A Rope Ladder and Some Storms                                       153   CHAPTER XVI.   Parliament and the Prince’s Income                                  178   CHAPTER XVII.   A New Favourite and a Settlement                                    198   CHAPTER XVIII.   A Most Extraordinary Event                                          203   CHAPTER XIX.   Which Contains a Great Deal of Fussing and Fuming and a little  Poetry                                                              221   CHAPTER XX.   The Prince is Cast Forth with His Family                            247   CHAPTER XXI.   The Death of the Queen                                              261   CHAPTER XXII.   The Year of Mourning                                                282   CHAPTER XXIII.   A Husband and a Lover                                               294   CHAPTER XXIV.   The Reconciliation               "In London," says the Duke of Somerset, "I do not know why the Duke of Kent, who has a large share of our fortunes, has not had the honour of being elected King of England, but there is a precedent for it, which I think we will be better served with a new edition of the history of the British Empire, and the Prince of Wales, who is the eldest brother of the Duke of Kent, has been admitted, and my eldest brother, George, Duke of Kent, has also been elected, and his brother the Duke of Cornwall, Prince of Wales, has been appointed," said the Duke of York, and he was right.

The question is very simple, but we have seen how many centuries ago I came into this realm, and so it is a little bit, and I am sure that I don't like to say this, but I am going to say it anyway.

As a boy I am, and I think that I have a right to be, but I think I am also not going to be, because I'm a man and that is the best thing about me.

I don't know how many years ago I was born, but it was one of those moments, and I have no idea why I was born. But, sir, the truth is that I am the last of the princes who have been born in England, and so I can be a prince and I'm not going to say it, because it is a very serious matter, because I don't think that it is a big thing for someone to be a prince, but I have the same rights as the other princes before me, and I am a prince that, if you think, can be very popular and a prince that has not been born a prince, I'm not sure that it is a very good thing.

When I was born, I was very important, and I have been told that I am the last of the princes that I am, but, as I said, I am also the last of the Prince of Waleses who has been born, and I'm not going to say what else it is, but I am proud to be able, if I have a son, to go forward, and I'm going to be a king.

I am not a prince of the British Empire, but I do not think it is something that is important for my family, and I am not going to say that it is, I just do not think it is a good thing for a boy to be a prince.

I'm not going to say that I think that I am still the only person that I am not going to say this, and I have been in the same position as a prince, and I'm going to be a prince, and that is an issue.

There is no doubt that, if you think, my mother and my brothers, that I am going to



Took 30.842 s
```

This example is based on the model: OccamRazor/mpt-7b-storywriter-4bit-128g.

Here is a [Google Colab](https://colab.research.google.com/drive/1JoSObRbuehRHWh7Q12Qy-7kFPRVj25yz?usp=sharing). 
You will need a A100 40GB to read a context length of 9000 tokens.

## Installation

### Setup

```
pip install -r requirements.txt 
python setup.py install         
```

The default backend is triton which is the fastest. For cuda support install also the CUDA kernels:

```
python setup_cuda.py install         
```


## Running mpttune

The above process installs a `mpttune` command in your environment.

### Download Models

First, start by downloading the weights of a MPT model:
```
$ wget https://huggingface.co/OccamRazor/mpt-7b-storywriter-4bit-128g/resolve/main/model.safetensors
```

### Generate Text

You can generate text directly from the command line. This generates text from the base model:
```
$ mpttune generate \
    --interactive \
    --model mpt-7b-storywriter-4bit \
    --weights model.safetensors \
    --max_new_tokens=600 \
    --use_cache \
    --do_sample \
    --prompt "The first person on the moon is "
```

### Finetune A Base Model

You may also finetune a base model yourself. First, you need to download a dataset:
```
$ wget https://github.com/gururise/AlpacaDataCleaned/raw/main/alpaca_data_cleaned.json
```

You can finetune any model of the MPT family:

<details>
<summary>MPT-7B</summary>
<br>

    $ mpttune finetune \
        --model=mpt-7b \
        --weights=mosaicml/mpt-7b \
        --dataset=./alpaca_data_cleaned.json \
        --data_type=alpaca \
        --lora_out_dir=./mpt-7b-alpaca/ \
        --mbatch_size=1 \
        --batch_size=2 \
        --epochs=3 \
        --lr=3e-4 \
        --cutoff_len=256 \
        --lora_r=8 \
        --lora_alpha=16 \
        --lora_dropout=0.05 \
        --warmup_steps=5 \
        --save_steps=50 \
        --save_total_limit=3 \
        --logging_steps=5 \
        --target_modules='["Wqkv"]'

    The above commands will download the model and use LoRA to finetune the quantized model. The final adapters and the checkpoints will be saved in `mpt-7b-alpaca` and available for generation as follows:

    $ mpttune generate \
        --interactive \
        --model mpt-7b \
        --weights mosaicml/mpt-7b \
        --lora_apply_dir mpt-7b-alpaca \
        --max_new_tokens 50 \
        --use_cache \
        --do_sample \
        --instruction "How to prepare pasta?"

</details>


<details>
<summary>MPT-7B-INSTRUCT</summary>
<br>

    $ mpttune finetune \
        --model=mpt-7b-instruct \
        --weights=mosaicml/mpt-7b-instruct \
        --dataset=./alpaca_data_cleaned.json \
        --data_type=alpaca \
        --lora_out_dir=./mpt-7b-instruct-alpaca/ \
        --mbatch_size=1 \
        --batch_size=2 \
        --epochs=3 \
        --lr=3e-4 \
        --cutoff_len=256 \
        --lora_r=8 \
        --lora_alpha=16 \
        --lora_dropout=0.05 \
        --warmup_steps=5 \
        --save_steps=50 \
        --save_total_limit=3 \
        --logging_steps=5 \
        --target_modules='["Wqkv"]'

    The above commands will download the model and use LoRA to finetune the quantized model. The final adapters and the checkpoints will be saved in `mpt-7b-instruct-alpaca` and available for generation as follows:

    $ mpttune generate \
        --interactive \
        --model mpt-7b-instruct \
        --weights mosaicml/mpt-7b-instruct \
        --lora_apply_dir mpt-7b-instruct-alpaca \
        --max_new_tokens 50 \
        --use_cache \
        --do_sample \
        --instruction "How to prepare pasta?"

</details>


<details>
<summary>MPT-7B-CHAT</summary>
<br>

    $ mpttune finetune \
        --model=mpt-7b-chat \
        --weights=mosaicml/mpt-7b-chat \
        --dataset=./alpaca_data_cleaned.json \
        --data_type=alpaca \
        --lora_out_dir=./mpt-7b-chat-alpaca/ \
        --mbatch_size=1 \
        --batch_size=2 \
        --epochs=3 \
        --lr=3e-4 \
        --cutoff_len=256 \
        --lora_r=8 \
        --lora_alpha=16 \
        --lora_dropout=0.05 \
        --warmup_steps=5 \
        --save_steps=50 \
        --save_total_limit=3 \
        --logging_steps=5 \
        --target_modules='["Wqkv"]'

    The above commands will download the model and use LoRA to finetune the quantized model. The final adapters and the checkpoints will be saved in `mpt-7b-chat-alpaca` and available for generation as follows:

    $ mpttune generate \
        --interactive \
        --model mpt-7b-chat \
        --weights mosaicml/mpt-7b-chat\
        --lora_apply_dir mpt-7b-chat-alpaca \
        --max_new_tokens 50 \
        --use_cache \
        --do_sample \
        --instruction "How to prepare pasta?"

</details>


<details>
<summary>MPT-7B-STORYWRITER</summary>
<br>

    $ mpttune finetune \
        --model=mpt-7b-storywriter \
        --weights=mosaicml/mpt-7b-storywriter \
        --dataset=./alpaca_data_cleaned.json \
        --data_type=alpaca \
        --lora_out_dir=./mpt-7b-storywriter-alpaca/ \
        --mbatch_size=1 \
        --batch_size=2 \
        --epochs=3 \
        --lr=3e-4 \
        --cutoff_len=256 \
        --lora_r=8 \
        --lora_alpha=16 \
        --lora_dropout=0.05 \
        --warmup_steps=5 \
        --save_steps=50 \
        --save_total_limit=3 \
        --logging_steps=5 \
        --target_modules='["Wqkv"]'

    The above commands will download the model and use LoRA to finetune the quantized model. The final adapters and the checkpoints will be saved in `mpt-7b-storywriter-alpaca` and available for generation as follows:

    $ mpttune generate \
        --interactive \
        --model mpt-7b-storywriter \
        --weights mosaicml/mpt-7b-storywriter \
        --lora_apply_dir mpt-7b-storywriter-alpaca \
        --max_new_tokens 50 \
        --use_cache \
        --do_sample \
        --instruction "How to prepare pasta?"

</details>


<details>
<summary>MPT-7B-STORYWRITER-4BIT-128G</summary>
<br>

    $ wget https://huggingface.co/OccamRazor/mpt-7b-storywriter-4bit-128g/resolve/main/model.safetensors
    
    $ mpttune finetune \
        --model=mpt-7b-storywriter-4bit \
        --weights=./model.safetensors \
        --dataset=./alpaca_data_cleaned.json \
        --data_type=alpaca \
        --lora_out_dir=./mpt-7b-storywriter-4bit-alpaca/ \
        --mbatch_size=1 \
        --batch_size=2 \
        --epochs=3 \
        --lr=3e-4 \
        --cutoff_len=256 \
        --lora_r=8 \
        --lora_alpha=16 \
        --lora_dropout=0.05 \
        --warmup_steps=5 \
        --save_steps=50 \
        --save_total_limit=3 \
        --logging_steps=5 \
        --target_modules='["Wqkv"]'

    The above commands will download the model and use LoRA to finetune the quantized model. The final adapters and the checkpoints will be saved in `mpt-7b-storywriter-4bit-alpaca` and available for generation as follows:

    $ mpttune generate \
        --interactive \
        --model mpt-7b-storywriter-4bit \
        --weights model.safetensors \
        --lora_apply_dir mpt-7b-storywriter-4bit-alpaca \
        --max_new_tokens=50 \
        --use_cache \
        --do_sample \
        --instruction "How to prepare pasta?"

</details>









## Todos

Work that stills needs to be done:
* Add triton flash attention as the only one that supports attention bias (alibi)


## Acknowledgements

**mpttune** is based on the following projects:
* The GPTQ algorithm and codebase by the [IST-DASLAB](https://github.com/IST-DASLab/gptq) with modifications by [@qwopqwop200](https://github.com/qwopqwop200/)
* The `alpaca_lora_4bit` repo by [johnsmith0031](https://github.com/johnsmith0031)
* The PEFT repo and its implementation of LoRA
* The LLAMA, OPT, and BLOOM models by META FAIR and the BigScience consortium
* The `llmtune` repo by [kuleshov-group](https://github.com/kuleshov-group/llmtune)


## Consultations
Need a custom solution? Let me know: `r.m.mihaylov@gmail.com`

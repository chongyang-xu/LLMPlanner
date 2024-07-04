from llm_planner.service.llm_service import SingleLLM

from dataset import load_coqa_story

import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

REPEAT = 3

model_list = [
    #'/DS/dsg-ml/nobackup/cxu/weights/Mistral-7B-v0.3/',
    '/DS/dsg-ml/nobackup/cxu/weights/Meta-Llama-3-8B/'
]

q_list = load_coqa_story(n_query=16)

model = model_list[0]

handle = SingleLLM(None, model_path=model, dtype='half')

n_in = 256
t_list = handle.token(q_list, seq_token_len=n_in)
#logger.info(handle.detoken(t_list))
for n_out in [1, 4, 16, 64, 256, 1024]:
    try:
        ml = model.split('/')[-2]
        tag = f"{ml}_out_{n_out}"
        gen_texts = handle.test_tids(t_list,
                                     max_tokens=n_out,
                                     tag=f"{tag}_warmup")
        for r in range(REPEAT):
            gtag = f"{tag}_r{r}"
            gen_texts = handle.test_tids(t_list, max_tokens=n_out, tag=gtag)
    except Exception as e:
        #logger.info(gen_texts)
        print(e)

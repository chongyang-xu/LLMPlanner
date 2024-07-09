from typing import Dict, Any

from llm_planner.service.vllm_serve import VLLMServe

from llm_planner.data.coqa_repurpose import load_coqa_story

REPEAT = 3

model = '/DS/dsg-ml/nobackup/cxu/weights/Meta-Llama-3-8B/'

policy_para: Dict[str, Any] = {"model_path": model}
handle = VLLMServe(None, policy_param_=policy_para)
handle.init_service()

n_in = 256
q_list = load_coqa_story(n_query=16)
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

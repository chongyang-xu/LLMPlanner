from typing import List, Dict, Any

from llm_planner.util import timing
from llm_planner.query import Query, InstructQuery, Instruct
from .service import SingleLLMFinetune

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer

from peft import get_peft_model
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


class HFFullParameterFinetune(SingleLLMFinetune):

    def __init__(self, p_selector, policy_param_: Dict[str, Any]):
        super().__init__()
        self.p_selector = p_selector
        self.init_done = False
        ##########################
        #
        # setting parameters
        #
        ##########################
        self.model_str = policy_param_.get(
            'model', "/DS/dsg-ml/nobackup/cxu/weights/gpt2/")

    def init_service(self, inst_param: Dict[str, Any]):
        if self.init_done:
            return
        self.model_str = inst_param.get('model', self.model_str)
        self.output_path = inst_param.get('output_path', None)
        self.dataset = inst_param.get('dataset', None)
        assert self.output_path is not None, "output_path is required"
        assert self.dataset is not None, "dataset is required"

        output_dir = inst_param.get('output_dir', self.output_path)
        per_device_train_batch_size = inst_param.get(
            'per_device_train_batch_size', 1)
        per_device_eval_batch_size = inst_param.get(
            'per_device_eval_batch_size', 1)

        eval_strategy = inst_param.get('eval_strategy', 'steps')
        logging_strategy = inst_param.get('logging_strategy', 'steps')
        save_strategy = inst_param.get('save_strategy', 'steps')

        max_steps = inst_param.get('max_steps', 10)
        save_steps = inst_param.get('save_steps', max_steps)
        logging_steps = inst_param.get('logging_strategy', 5)

        optim = inst_param.get('optim', 'adamw_torch')
        gradient_accumulation_steps = inst_param.get(
            'gradient_accumulation_steps', 1)
        learning_rate = inst_param.get('learning_rate', 5e-5)
        lr_scheduler_type = inst_param.get('lr_scheduler_type', 'cosine')
        warmup_ratio = inst_param.get('warmup_ratio', 0.0)

        max_seq_length = inst_param.get('max_seq_length', 512)

        # dump parameters
        print('-' * 50)
        print(self.__class__.__name__)
        print(f"model      : {self.model_str}")
        print('-' * 50)

        config = {
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "contiguous_gradients": True,
                "overlap_comm": True
            },
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
        }
        self.sft_config = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_checkpointing=True,
            deepspeed=config,
            eval_strategy=eval_strategy,
            logging_strategy=logging_strategy,
            logging_steps=logging_steps,
            optim=optim,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_steps=max_steps,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            save_strategy=save_strategy,
            save_steps=save_steps,
            max_seq_length=max_seq_length)

        self.base_model = AutoModelForCausalLM.from_pretrained(self.model_str)
        self.base_model.gradient_checkpointing_enable()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_str,
                                                       trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.base_model.resize_token_embeddings(len(self.tokenizer))

        self.trainer = SFTTrainer(model=self.base_model,
                                  train_dataset=self.dataset.train(),
                                  eval_dataset=self.dataset.val(),
                                  peft_config=None,
                                  formatting_func=None,
                                  tokenizer=self.tokenizer,
                                  args=self.sft_config)

    @timing
    def work_on(self, q_list: List[Query]):
        assert len(q_list) == 1
        q = q_list[0]
        assert isinstance(q, InstructQuery)
        assert q.instruct == Instruct.FINETUNE_FULL

        self.init_service(q.instruct_param)

        self.trainer.train()
        self.trainer.save_model(f"{self.output_path}/")

        q_list[0].response = "Finished"

        return q_list


class HFLoRAFinetune(SingleLLMFinetune):

    def __init__(self, p_selector, policy_param_: Dict[str, Any]):
        super().__init__()
        self.p_selector = p_selector
        self.init_done = False
        ##########################
        #
        # setting parameters
        #
        ##########################
        self.model_str = policy_param_.get(
            'model', "/DS/dsg-ml/nobackup/cxu/weights/gpt2/")

    def init_service(self, inst_param: Dict[str, Any]):
        if self.init_done:
            return
        self.model_str = inst_param.get('model', self.model_str)
        self.lora_output_path = inst_param.get('lora_output_path', None)
        self.dataset = inst_param.get('dataset', None)
        assert self.lora_output_path is not None, "lora_output_path is required"
        assert self.dataset is not None, "dataset is required"

        lora_rank = inst_param.get('lora_rank', 4)
        lora_alpha = inst_param.get('lora_alpha', 16)
        target_modules = inst_param.get('target_modules', ["k_proj", "v_proj"])
        lora_dropout = inst_param.get('lora_dropout', 0.1)
        lora_bias = inst_param.get('lora_bias', "none")

        output_dir = inst_param.get('output_dir', self.lora_output_path)
        per_device_train_batch_size = inst_param.get(
            'per_device_train_batch_size', 4)
        per_device_eval_batch_size = inst_param.get(
            'per_device_eval_batch_size', 4)

        eval_strategy = inst_param.get('eval_strategy', 'steps')
        logging_strategy = inst_param.get('logging_strategy', 'steps')
        save_strategy = inst_param.get('save_strategy', 'steps')

        max_steps = inst_param.get('max_steps', 10)
        save_steps = inst_param.get('save_steps', max_steps)
        logging_steps = inst_param.get('logging_strategy', 5)

        optim = inst_param.get('optim', 'adamw_torch')
        gradient_accumulation_steps = inst_param.get(
            'gradient_accumulation_steps', 4)
        learning_rate = inst_param.get('learning_rate', 5e-5)
        lr_scheduler_type = inst_param.get('lr_scheduler_type', 'cosine')
        warmup_ratio = inst_param.get('warmup_ratio', 0.0)

        max_seq_length = inst_param.get('max_seq_length', 512)

        # dump parameters
        print('-' * 50)
        print(self.__class__.__name__)
        print(f"model      : {self.model}")
        print(f"lora_output_path: {self.lora_output_path}")
        print(f"lora_rank       : {lora_rank}")
        print(f"lora_alpha      : {lora_alpha}")
        print(f"target_modules  : {target_modules}")
        print(f"lora_dropout    : {lora_dropout}")
        print(f"lora_bias       : {lora_bias}")
        print('-' * 50)

        self.base_model = AutoModelForCausalLM.from_pretrained(self.model_str)
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.gradient_checkpointing_enable()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_str,
                                                       trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.base_model.resize_token_embeddings(len(self.tokenizer))

        self.lora_config = LoraConfig(r=lora_rank,
                                      lora_alpha=lora_alpha,
                                      target_modules=target_modules,
                                      lora_dropout=lora_dropout,
                                      bias=lora_bias,
                                      task_type="CAUSAL_LM")
        self.peft_model = get_peft_model(self.base_model, self.lora_config)

        self.sft_config = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_checkpointing=True,
            eval_strategy=eval_strategy,
            logging_strategy=logging_strategy,
            logging_steps=logging_steps,
            optim=optim,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_steps=max_steps,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            save_strategy=save_strategy,
            save_steps=save_steps,
            max_seq_length=max_seq_length)

        self.trainer = SFTTrainer(model=self.peft_model,
                                  train_dataset=self.dataset.train(),
                                  eval_dataset=self.dataset.val(),
                                  peft_config=self.lora_config,
                                  formatting_func=None,
                                  tokenizer=self.tokenizer,
                                  args=self.sft_config)

        #for name, module in trainer.model.named_modules():
        #    if "norm" in name:
        #        module = module.to(torch.float32)

    @timing
    def work_on(self, q_list: List[Query]):
        assert len(q_list) == 1
        q = q_list[0]
        assert isinstance(q, InstructQuery)
        assert q.instruct == Instruct.FINETUNE_LORA

        self.init_service(q.instruct_param)

        self.trainer.train()
        self.trainer.save_model(f"{self.lora_output_path}/")

        q_list[0].response = "Finished"

        return q_list

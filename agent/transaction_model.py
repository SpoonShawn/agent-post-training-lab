"""Hugging Face model backend for isolated transaction episodes."""
import time

from agent.transaction_runtime import ContextBudgetExceeded


class TransactionAgent:
    def __init__(self, model_path, adapter_path=None, *, max_length=8192, max_new_tokens=512):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(model_path,dtype=torch.bfloat16,
                                                       device_map="auto",trust_remote_code=False)
        if adapter_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model,adapter_path)
        self.model.eval()
        self.max_length,self.max_new_tokens = max_length,max_new_tokens
        self.usage = []

    def generate(self,messages,tools):
        start = time.perf_counter()
        inputs = self.tokenizer.apply_chat_template(messages,tools=tools,add_generation_prompt=True,
                                                    tokenize=True,return_dict=True,return_tensors="pt")
        prompt_tokens = inputs["input_ids"].shape[-1]
        if prompt_tokens+self.max_new_tokens > self.max_length:
            raise ContextBudgetExceeded("No evidence truncation allowed")
        inputs = {k:v.to(self.model.device) for k,v in inputs.items()}
        with self.torch.no_grad():
            output = self.model.generate(**inputs,max_new_tokens=self.max_new_tokens,do_sample=False)
        tokens = output[0][prompt_tokens:]
        text = self.tokenizer.decode(tokens,skip_special_tokens=False)
        if self.tokenizer.eos_token:
            while text.rstrip().endswith(self.tokenizer.eos_token):
                text = text.rstrip()[:-len(self.tokenizer.eos_token)]
        self.usage.append(dict(prompt_tokens=prompt_tokens,generated_tokens=len(tokens),
                               seconds=time.perf_counter()-start))
        return text

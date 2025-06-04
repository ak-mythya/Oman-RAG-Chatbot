import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/models"
os.environ["HF_DATASETS_CACHE"] = "/workspace/models"


import logging
import torch
from config import llm, tokenizer

class TextGenerator:
    def __init__(self):
        self.model = llm
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(__name__)

    def generate(self, prompt, task='llm', max_new_tokens=4096, temperature=0.7, do_sample=True):
        try:
            messages = [{"role": "system", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")
            input_token_count = inputs["input_ids"].shape[1]
            
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample
            )
            generated_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            generation = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            output_token_count = len(generated_tokens)
            total_token_count = input_token_count + output_token_count
            
            self.logger.info(f"Generation completed for task '{task}'. Output: {generation}")
            self.logger.info(f"Token counts - Input: {input_token_count}, Output: {output_token_count}, Total: {total_token_count}")
            return generation
        except Exception as error:
            self.logger.error(f"Error during generation for task '{task}': {str(error)}")
            return f"Error during generation: {str(error)}"

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: list[str],
        task: str = "llm",
        max_new_tokens: int = 4096,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> list[str]:
        try:
            messages = [[{"role": "system", "content": p}] for p in prompts]
            formatted = [self.tokenizer.apply_chat_template(m, tokenize=False) for m in messages]
            inputs = self.tokenizer(formatted, return_tensors="pt", padding=True, truncation=True).to("cuda")
            input_lens = inputs["attention_mask"].sum(dim=1).tolist()
            
            outputs_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )
            
            generations = []
            for ids, in_len in zip(outputs_ids, input_lens):
                gen_tokens = ids[in_len:]
                generation = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                generations.append(generation)
            
            total_input_tokens = sum(input_lens)
            total_output_tokens = sum(len(ids) - in_len for ids, in_len in zip(outputs_ids, input_lens))
            total_tokens = total_input_tokens + total_output_tokens
            
            self.logger.info(f"Batch generation completed for task '{task}'. Outputs: {generations}")
            self.logger.info(f"Token counts - Total Input: {total_input_tokens}, Total Output: {total_output_tokens}, Total: {total_tokens}")
            return generations
        except Exception as error:
            self.logger.error(f"Error during batch generation for task '{task}': {str(error)}")
            return [f"Error during generation: {str(error)}"] * len(prompts)

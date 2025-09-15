from openai import OpenAI
import os
import torch
from src.context_matters.agents.agent import Agent

class GPTAgent(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.prompt_chain = []
        
    def _chat(self, messages, temperature=None, top_p=None):
            completion_args = {
                "model": self.model,
                "messages": messages
            }
            if temperature is not None:
                completion_args["temperature"] = temperature
            if top_p is not None:
                completion_args["top_p"] = top_p

            completion = self.client.chat.completions.create(**completion_args)
            return completion.choices[0].message.content

    def reset(self):
        self.prompt_chain = []
        torch.cuda.empty_cache()

    def llm_call(self, content: str, prompt: str, temperature=None, top_p=None):
        messages = [
            {"role": "system", "content": content},
            {"role": "user",   "content": prompt}
        ]
        return self._chat(messages, temperature=temperature, top_p=top_p)

    def init_prompt_chain(self, content: str, prompt: str):
        assert len(self.prompt_chain) == 0, "Prompt chain is not empty!"
        self.prompt_chain = [
            {"role": "system", "content": content},
            {"role": "user",   "content": prompt}
        ]

    def update_prompt_chain(self, content: str, prompt: str):
        assert self.prompt_chain, "Prompt chain is empty. Call init_prompt_chain first."
        assert self.prompt_chain[0].get("role") == "system", "First message must be a system message."

        self.prompt_chain[0]["content"] = content
        self.prompt_chain.append({"role": "user", "content": prompt})
        
    def update_prompt_chain_with_response(self, response: str, role: str = "assistant"):
        assert self.prompt_chain, "Prompt chain is empty. Call init_prompt_chain first."
        self.prompt_chain.append({"role": role, "content": response})

    def query_msg_chain(self, temperature=None, top_p=None):
        assert self.prompt_chain, "Prompt chain is empty. Call init_prompt_chain first."
        return self._chat(self.prompt_chain, temperature=temperature, top_p=top_p)
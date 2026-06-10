from typing import Callable, Dict, Optional
from ..core.optimizer import TokenOptimizer


class OptimizedLLMWrapper:
    def __init__(self, llm_callable: Callable[[str], str], optimizer: Optional[TokenOptimizer] = None):
        self.llm_callable = llm_callable
        self.optimizer = optimizer or TokenOptimizer()

    def invoke(self, prompt: str, context=None, query: str = "") -> Dict[str, object]:
        opt = self.optimizer.optimize(prompt=prompt, context=context, query=query)
        response = self.llm_callable(opt.optimized_prompt)
        return {"response": response, "optimization": opt.to_dict()}

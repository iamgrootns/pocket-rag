"""
LLM wrapper (external model = wrapper class, uniform interface).
One class, one concern: grounded transcript context + question -> cited answer.
Only this file knows about llama.cpp - swap models without touching Manager.
"""

import os

from config import LLM_MAX_SNIPPETS, LLM_MAX_TOKENS, LLM_MODEL_PATH, LLM_TEMPERATURE
from loggerutil import logger_util


class LLMWrapper:
    """Local GGUF generation through llama-cpp-python."""

    def __init__(self):
        self.logger = logger_util("llm")
        self.model = None

        if not os.path.isfile(LLM_MODEL_PATH):
            self.logger.error(f"LLM file not found: {LLM_MODEL_PATH}")
            self.logger.error("Put your Qwen2.5-3B Q4 GGUF in backend/Models/ (config.py) and restart.")
            return

        try:
            from llama_cpp import Llama

            gpu_layers = int(os.environ.get("RAG_GPU_LAYERS", "36"))
            if gpu_layers > 0:
                try:
                    self.model = Llama(
                        model_path=LLM_MODEL_PATH,
                        n_ctx=4096,
                        n_batch=256,
                        n_gpu_layers=gpu_layers,
                        verbose=False,
                    )
                    self.logger.info(f"LLM loaded on GPU (layers={gpu_layers})")
                except Exception as e:
                    self.logger.error(f"GPU load failed ({e}) - falling back to CPU")
                    self.model = None

            if self.model is None:
                self.model = Llama(
                    model_path=LLM_MODEL_PATH,
                    n_ctx=4096,
                    n_batch=256,
                    n_gpu_layers=0,
                    verbose=False,
                )
                self.logger.info("LLM loaded on CPU")
        except Exception as e:
            self.logger.error(f"LLM failed to load: {e}")

    def generate_grounded_answer(self, question: str, snippets: list[dict]) -> str:
        """snippets = [{"number","content","source"}...] used strictly as context."""
        if self.model is None:
            return "LLM is not loaded. Place the GGUF file in Models/ and restart the backend."

        context = "\n\n".join(
            f"[{s['number']}] (source: {s['source']})\n{s['content']}"
            for s in snippets[:LLM_MAX_SNIPPETS]
        )

        system = (
            "You are a precise assistant answering ONLY from the numbered context below.\n"
            "Rules:\n"
            "1. Use only facts from the context.\n"
            "2. Cite the snippets you use like [1] or [2][3].\n"
            "3. If the context does not contain the answer, reply exactly: "
            "This is not covered in your documents.\n"
            "4. Keep the answer short and factual, no invented details."
        )

        try:
            out = self.model.create_chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}",
                    },
                ],
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
            )
            return out["choices"][0]["message"]["content"].strip()
        except Exception as e:
            self.logger.error(f"generation failed: {e}")
            return "Generation failed - check the backend log."

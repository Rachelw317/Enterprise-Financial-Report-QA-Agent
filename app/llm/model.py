import os
from typing import Any

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek


DEFAULT_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com"


class DeepSeekLLM(ChatDeepSeek):
	"""ChatDeepSeek configured for the DeepSeek API."""

	def __init__(
		self,
		model: str = DEFAULT_MODEL,
		api_key: str | None = None,
		base_url: str = DEFAULT_BASE_URL,
		temperature: float = 0.0,
		**kwargs: Any,
	) -> None:
		load_dotenv()
		resolved_api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
		if not resolved_api_key:
			raise RuntimeError(
				"DEEPSEEK_API_KEY is required. Set it in the environment or a .env file."
			)
		super().__init__(
			model=model,
			api_key=resolved_api_key,
			base_url=base_url,
			temperature=temperature,
			**kwargs,
		)


def create_deepseek_llm(**kwargs: Any) -> DeepSeekLLM:
	return DeepSeekLLM(**kwargs)


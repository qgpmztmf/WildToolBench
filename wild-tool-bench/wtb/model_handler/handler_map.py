from .api_inference.oai import OpenAIHandler
from .api_inference.deepseek import DeepSeekAPIHandler


api_inference_handler_map = {
    "gpt-4o-2024-11-20": OpenAIHandler,
    "deepseek-chat": DeepSeekAPIHandler,
}

HANDLER_MAP = {**api_inference_handler_map}

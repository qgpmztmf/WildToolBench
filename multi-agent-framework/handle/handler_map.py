from .oai import OpenAIHandler
from .deepseek import DeepSeekAPIHandler


api_inference_handler_map = {
    "gpt-4o-2024-11-20": OpenAIHandler,
    "deepseek-chat": DeepSeekAPIHandler,
}

agent_handle_map = {**api_inference_handler_map}

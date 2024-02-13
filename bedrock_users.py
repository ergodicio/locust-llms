from locust import User, task, constant
from test_clients import BedrockClient


class _BedrockLLAMA270bUser_(User):
    abstract = True

    def __init__(self, env):
        super().__init__(env)
        self.client = BedrockClient(model="meta.llama2-70b-chat-v1")

    wait_time = constant(1)


class BedrockLLAMA270bUser(_BedrockLLAMA270bUser_):
    wait_time = constant(1)

    @task
    def chat_completion(self):
        msg = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "What is the role of automatic differentiation in neural networks?",
            },
        ]
        self.client.chat_completion(msg)


class _BedrockLLAMA213bUser_(User):
    abstract = True

    def __init__(self, env):
        super().__init__(env)
        self.client = BedrockClient(model="meta.llama2-13b-chat-v1")

    wait_time = constant(1)


class BedrockLLAMA213bUser(_BedrockLLAMA213bUser_):
    wait_time = constant(1)

    @task
    def chat_completion(self):
        msg = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "What is the role of automatic differentiation in neural networks?",
            },
        ]
        self.client.chat_completion(msg)

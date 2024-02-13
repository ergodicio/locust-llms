from locust import User, task, constant
from test_clients import OpenAIClient


class _GPT35turboUser_(User):
    abstract = True

    def __init__(self, env):
        super().__init__(env)
        self.client = OpenAIClient("gpt-3.5-turbo")

    wait_time = constant(1)


class GPT35turboUser(_GPT35turboUser_):
    wait_time = constant(1)

    @task
    def chat_completion(self):
        msg = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the role of automatic differentiation in neural networks?"},
        ]
        self.client.chat_completion(msg)


class _GPT35turbo16kUser_(User):
    abstract = True

    def __init__(self, env):
        super().__init__(env)
        self.client = OpenAIClient("gpt-3.5-turbo-16k")

    wait_time = constant(1)


class GPT35turbo16kUser(_GPT35turbo16kUser_):
    wait_time = constant(1)

    @task
    def chat_completion(self):
        msg = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the role of automatic differentiation in neural networks?"},
        ]
        self.client.chat_completion(msg)


class _GPT35turbo1106User_(User):
    abstract = True

    def __init__(self, env):
        super().__init__(env)
        self.client = OpenAIClient("gpt-3.5-turbo-1106")

    wait_time = constant(1)


class GPT35turbo1106User(_GPT35turbo1106User_):
    wait_time = constant(1)

    @task
    def chat_completion(self):
        msg = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the role of automatic differentiation in neural networks?"},
        ]
        self.client.chat_completion(msg)


class _GPT40613User_(User):
    abstract = True

    def __init__(self, env):
        super().__init__(env)
        self.client = OpenAIClient("gpt-4-0613")

    wait_time = constant(1)


class GPT40613User(_GPT40613User_):
    wait_time = constant(1)

    @task
    def chat_completion(self):
        msg = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the role of automatic differentiation in neural networks?"},
        ]
        self.client.chat_completion(msg)

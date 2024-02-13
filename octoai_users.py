from locust import User, task, constant
from test_clients import OctoAIClient, TextModel


class _Llama213BUser_(User):
    abstract = True

    def __init__(self, env):
        super().__init__(env)
        self.client = OctoAIClient(model=TextModel.LLAMA_2_13B_CHAT_FP16)

    wait_time = constant(1)


class Llama213BUser(_Llama213BUser_):
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


class _Llama270BUser_(User):
    abstract = True

    def __init__(self, env):
        super().__init__(env)
        self.client = OctoAIClient(model=TextModel.LLAMA_2_70B_CHAT_FP16)

    wait_time = constant(1)


class Llama270BUser(_Llama270BUser_):
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


class _Llama270BI4User_(User):
    abstract = True

    def __init__(self, env):
        super().__init__(env)
        self.client = OctoAIClient(model=TextModel.LLAMA_2_70B_CHAT_INT4)

    wait_time = constant(1)


class Llama270BI4User(_Llama270BI4User_):
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


class _Mixtral8x7User_(User):
    abstract = True

    def __init__(self, env):
        super().__init__(env)
        self.client = OctoAIClient(model=TextModel.MIXTRAL_8X7B_INSTRUCT_FP16)

    wait_time = constant(1)


class Mixtral8x7User(_Mixtral8x7User_):
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


class _Mistral7BUser_(User):
    abstract = True

    def __init__(self, env):
        super().__init__(env)
        self.client = OctoAIClient(model=TextModel.MISTRAL_7B_INSTRUCT_FP16)

    wait_time = constant(1)


class Mistral7BUser(_Mistral7BUser_):
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

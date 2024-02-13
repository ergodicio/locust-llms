import time, json, enum

from openai import OpenAI
import boto3
from octoai.chat import TextModel
from octoai.client import Client

from locust import events


class OpenAIClient:
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.openaiclient = OpenAI()
        self.model = model

    def chat_completion(self, msg):
        request_meta = {
            "request_type": "OpenAI chat completion",
            "name": self.model,
            "start_time": time.time(),
            "response_length": 0,
            "response": None,
            "context": {},
            "exception": None,
        }
        start_perf_counter = time.perf_counter()

        try:
            response = self.openaiclient.chat.completions.create(model=self.model, messages=msg, max_tokens=512)
            request_meta["response"] = response.choices[0].message
            request_meta["response_length"] = response.usage.completion_tokens
        except Exception as e:
            request_meta["exception"] = e

        request_meta["response_time"] = (time.perf_counter() - start_perf_counter) * 1000

        events.request.fire(**request_meta)


class OctoAIClient:
    def __init__(self, model: enum = TextModel.LLAMA_2_70B_CHAT_FP16):
        self.octoaiclient = Client()
        self.model = model

    def chat_completion(self, msg):
        request_meta = {
            "request_type": "OctoAI chat completion",
            "name": f"{self.model.name}",
            "start_time": time.time(),
            "response_length": 0,
            "response": None,
            "context": {},
            "exception": None,
        }
        start_perf_counter = time.perf_counter()

        try:
            response = self.octoaiclient.chat.completions.create(model=self.model, messages=msg, max_tokens=512)
            request_meta["response"] = response.choices[0].message
            # request_meta["response_length"] = response.usage.total_tokens
            request_meta["response_length"] = response.usage.completion_tokens
        except Exception as e:
            request_meta["exception"] = e

        request_meta["response_time"] = (time.perf_counter() - start_perf_counter) * 1000

        events.request.fire(**request_meta)


class BedrockClient:
    def __init__(self, model: str = "meta.llama2-70b-chat-v1"):
        self.bedrockclient = boto3.client(service_name="bedrock-runtime")
        self.model = model

    def chat_completion(self, msg):
        request_meta = {
            "request_type": "Amazon AI chat completion",
            "name": f"{self.model}",
            "start_time": time.time(),
            "response_length": 0,
            "response": None,
            "context": {},
            "exception": None,
        }
        start_perf_counter = time.perf_counter()

        try:
            body = json.dumps(
                {
                    "prompt": msg,
                    "max_gen_len": 512,
                    # "temperature": 0.7,
                    # "top_p": 0.9,
                }
            )

            modelId = self.model
            accept = "application/json"
            contentType = "application/json"

            response = self.bedrockclient.invoke_model(
                body=body, modelId=modelId, accept=accept, contentType=contentType
            )
            response_body = json.loads(response.get("body").read())

            request_meta["response"] = response_body["generation"]
            request_meta["response_length"] = response_body[
                "generation_token_count"
            ]  # + response_body["generation_token_count"]
        except Exception as e:
            request_meta["exception"] = e

        request_meta["response_time"] = (time.perf_counter() - start_perf_counter) * 1000

        events.request.fire(**request_meta)

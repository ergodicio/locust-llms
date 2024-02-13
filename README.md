# LLM Endpoint load testing using Locust
This code repository performs testing of LLM API endpoints using [locust](http://locust.io). 

Clients Implemented
- OpenAI
- OctoAI
- Amazon Bedrock

Each of these clients uses each providers' SDK to make requests to various model endpoints.

### Installation
1. Clone the repo
    ```bash
    git clone https://github.com/ergodicio/locustllms.git
    ```
2.  Create a new virtual environment using `venv`:
    ```bash
    python -m venv myenv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

### Usage
1. Make sure environment variables are configured for each service. OpenAI, OctoAI, and boto3
2. From the terminal
    ```bash
    locust
    ```
3. Go to `http://0.0.0.0:8089`


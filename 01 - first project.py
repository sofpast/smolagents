from smolagents import CodeAgent, HfApiModel, DuckDuckGoSearchTool, InferenceClientModel
from dotenv import load_dotenv
import os
load_dotenv()

# Load the Hugging Face API key from environment variables
api_key = os.getenv("HUGGINGFACE_API_KEY")

# model = InferenceClientModel(
#     # model="google/flan-t5-base",
#     api_key=api_key,
#     # inference_client="huggingface"
# )

# Initialize the agent
agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel(token=api_key))

agent.run("how to make a cake?")
from smolagents import CodeAgent, HfApiModel
from smolagents import tool
from dotenv import load_dotenv
from huggingface_hub import list_models
import os

load_dotenv()

@tool
def model_most_download_tool(task: str) -> str:
    """
    Returns the most downloaded model from Hugging Face.
    Args:
        task (str): The task for which to find the most downloaded model (e.g., "text-generation").     
    Returns:   
        str: The ID of the most downloaded model for the specified task.
    """
    # import pdb
    # pdb.set_trace()
    most_downloaded_model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
    return most_downloaded_model.id

api_key = os.getenv("HUGGINGFACE_API_KEY")

agent = CodeAgent(
    tools=[model_most_download_tool],
    model=HfApiModel(api_key=api_key),
    # additional_authorized_imports=["requests", "markdownify"],
    # executor_type="e2b",  # Use E2B executor
)

agent.run("What is the most downloaded model for text generation on Hugging Face?")  # This will use the E2B executor
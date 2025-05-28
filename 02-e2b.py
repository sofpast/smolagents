from smolagents import CodeAgent, HfApiModel, VisitWebpageTool, DuckDuckGoSearchTool
from dotenv import load_dotenv
# from smolagents.executors import E2BCodeExecutor


import os

load_dotenv()

api_key = os.getenv("HUGGINGFACE_API_KEY")

# executor = E2BCodeExecutor()

agent = CodeAgent(
    tools=[VisitWebpageTool()],
    model=HfApiModel(api_key=api_key),
    additional_authorized_imports=["requests", "markdownify"],
    executor_type="e2b",  # Use E2B executor
)

agent.run("what is birthday of Abraham Lincoln?")  # This will use the E2B executor
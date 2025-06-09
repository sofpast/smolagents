from smolagents import CodeAgent, HfApiModel
from smolagents import tool, Tool
from huggingface_hub import InferenceClient
from huggingface_hub import list_models
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("HUGGINGFACE_API_KEY")

@tool
def model_download_tool(task: str) -> str:
    """
    This is a tool that returns the most downloaded model of a given task on the Hugging Face Hub.
    It returns the name of the checkpoint.
    
    Args:
        task: The task for which to get the download count.
    """
    try:
        # For text-to-image, return known working models
        if "text-to-image" in task.lower() or "image" in task.lower():
            # List of reliable text-to-image models
            reliable_models = [
                "stabilityai/stable-diffusion-xl-base-1.0",
                "runwayml/stable-diffusion-v1-5",
                "stabilityai/stable-diffusion-2-1",
                "CompVis/stable-diffusion-v1-4"
            ]
            return reliable_models[0]
        
        # For other tasks, try the original approach
        models = list(list_models(filter=task, sort="downloads", direction=-1, limit=5))
        
        if not models:
            return "stabilityai/stable-diffusion-xl-base-1.0"
        
        most_downloaded_model = models[0]
        return most_downloaded_model.id
    except Exception as e:
        print(f"Error fetching model: {e}")
        return "stabilityai/stable-diffusion-xl-base-1.0"

class TextToImageTool(Tool):
    description = "This tool creates an image according to a prompt, which is a text description."
    name = "image_generator"
    inputs = {
        "prompt": {
            "type": "string", 
            "description": "The image generator prompt. Don't hesitate to add details in the prompt to make the image look better, like 'high-res, photorealistic', etc."
        },
        "model": {
            "type": "string",
            "description": "The Hugging Face model ID to use for image generation. If not provided, will use the default model.",
            "nullable": True
        }
    }
    output_type = "image"
    current_model = "stabilityai/stable-diffusion-xl-base-1.0"
    
    def __init__(self):
        super().__init__()
        self.client = None
    
    def forward(self, prompt, model=None):
        try:
            # Validate API key
            if not api_key:
                return "Error: HUGGINGFACE_API_KEY not found in environment variables"
            
            # Update model if provided and different from current
            if model and model != self.current_model:
                self.current_model = model
                self.client = None  # Reset client to reinitialize with new model
            
            # Initialize client if not already done
            if not self.client:
                self.client = InferenceClient(
                    model=self.current_model,
                    token=api_key
                )
            
            print(f"Generating image with model: {self.current_model}")
            print(f"Prompt: {prompt}")
            
            # Generate image
            try:
                image = self.client.text_to_image(prompt)
                
                if image:
                    image.save("image.png")
                    return f"Successfully saved image with prompt: '{prompt}' using model: {self.current_model}"
                else:
                    return "Error: No image was generated"
                    
            except Exception as api_error:
                error_msg = str(api_error)
                print(f"API Error: {error_msg}")
                
                # If the current model fails, try a fallback model
                if "not supported" in error_msg.lower() or "not found" in error_msg.lower() or "does not seem to be supported" in error_msg.lower():
                    fallback_model = "runwayml/stable-diffusion-v1-5"
                    print(f"Trying fallback model: {fallback_model}")
                    
                    try:
                        self.client = InferenceClient(
                            model=fallback_model,
                            token=api_key
                        )
                        image = self.client.text_to_image(prompt)
                        if image:
                            image.save("image.png")
                            return f"Successfully saved image with prompt: '{prompt}' using fallback model: {fallback_model}"
                    except Exception as fallback_error:
                        return f"Error with both primary and fallback models: {str(fallback_error)}"
                
                return f"Error generating image: {error_msg}"
            
        except Exception as e:
            return f"Error in image generation setup: {str(e)}"

# Initialize the image generator tool
image_generator = TextToImageTool()

# Set up the agent
model_id = "Qwen/QwQ-32B-Preview"

agent = CodeAgent(
    tools=[image_generator, model_download_tool], 
    model=HfApiModel(api_key=api_key)
)

# Run the agent
agent.run(
    "Improve this prompt, then generate an image of it. Prompt: A cat wearing a hazmat suit in contaminated area. "
    "Get the latest model for text-to-image from the Hugging Face Hub."
)
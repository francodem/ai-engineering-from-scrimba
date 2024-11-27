from openai import OpenAI


class GPT:
    """
    A class to interact with the OpenAI GPT model.
    """

    def __init__(self, api_key: str = None):
        """
        Initialize the GPT class with an OpenAI client.
        """
        self.client = OpenAI(api_key=api_key)

    def get_completion_from_obj(self, obj: list[dict], model: str = "gpt-4o-mini") -> str:
        """
        Get the completion from a stream object
        
        Args:
            obj (list[dict]): The stream object
            model (str): The model to use for the completion. Default is "gpt-4o-mini".
        
        Returns:
            response (str): The completion
        """
        stream = self.client.chat.completions.create(
        model=model,
        messages=obj,
            stream=True
        )
        response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

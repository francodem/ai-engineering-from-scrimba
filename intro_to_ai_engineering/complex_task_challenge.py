from models.gpt import GPT
from os import getenv
from dotenv import load_dotenv

load_dotenv()


class TopicExplainer:

    def __init__(self, years_old: int, topic: str):
        self.years_old = years_old
        self.topic = topic

    def get_template_obj(self) -> list[dict[str, str]]:
        """
        Get the template for the topic explainer.
        
        Returns:
            obj: list[dict[str, str]]: The template for the topic explainer.
        """
        obj = [
            {
                "role": "system",
                "content": f"""You are a teacher of {self.topic} that teaches like \
                    a teacher of {self.years_old} years old children. Do not answer \
                    questions that are not related to the topic, this is a unbreakable \
                    rule.
                    """
            }
        ]
        return obj


def chat():
    gpt = GPT(api_key=getenv("OPENAI_API_KEY"))
    topic_explainer = TopicExplainer(years_old=30, topic="nvidia company")
    chat = topic_explainer.get_template_obj()

    msgs_count = 0
    while msgs_count <= 10:
        user_input = input("\nMe: ")
        chat.append({"role": "user", "content": user_input})
        
        response = ""
        print("AI: ", end="", flush=True)
        for chunk in gpt.get_completion_from_obj(obj=chat):
            print(chunk, end="", flush=True)
            response += chunk

        chat.append({"role": "assistant", "content": response})
        msgs_count += 1


if __name__ == "__main__":
    chat()

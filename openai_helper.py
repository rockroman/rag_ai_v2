import requests
import json
from dotenv import load_dotenv
import os


load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
print( "this is API key ",openai_api_key)

class OpenAIChat:
    def __init__(
            self,
            openai_api_key: str | None = None,
            openai_model: str = 'gpt-3.5-turbo',
            response_keys: tuple = ('text', 'reference_url', 'image_url')
    ):
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.openai_model = openai_model
        self.url = 'https://api.openai.com/v1/chat/completions'
        self.messages = [
            {
                'role': 'system',
                'content': f'You are a helpful assistant designed to output JSON. Use these keys only if related: {", ".join(response_keys)}'
            }
        ]

    def send_message(self, message):
        self.add_message('user', message)
        return self.get_response()

    def add_message(self, role, content):
        self.messages.append({'role': role, 'content': content})

    def get_response(self) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}"
        }
        data = {
            'model': self.openai_model,
            'response_format': {'type': 'json_object'},
            'messages': self.messages
        }
        try:
            response = requests.post(self.url, headers=headers, json=data, timeout=20)
            response.raise_for_status()
        except requests.exceptions.RequestException as err:
            return {'text': str(err)}
        else:
            response_data = response.json()
            if 'choices' in response_data and len(response_data['choices']) > 0:
                last_message = response_data['choices'][-1]['message']['content']
                self.add_message('assistant', last_message)
                return json.loads(last_message)
            else:
                return {'text': 'No response from OpenAI API'}


if __name__ == '__main__':
    chat_example = OpenAIChat()
    print(chat_example.send_message('Hello, how are you?'))

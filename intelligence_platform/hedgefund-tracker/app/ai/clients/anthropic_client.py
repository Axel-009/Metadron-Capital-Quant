from app.ai.clients.base_client import AIClient
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
import os


class AnthropicClient(AIClient):
    """
    Anthropic Claude client implementation using the Anthropic Messages API.
    Default model: claude-opus-4-6
    """
    DEFAULT_MODEL = "claude-opus-4-6"

    def __init__(self, model: str = DEFAULT_MODEL):
        load_dotenv()
        self.model = model
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("WARNING: ANTHROPIC_API_KEY not set.")

        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            self._use_native = True
        except ImportError:
            # Fallback to OpenAI SDK with Anthropic-compatible endpoint
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.anthropic.com/v1"
            )
            self._use_native = False

    def get_model_name(self) -> str:
        return self.model

    @retry(
        wait=wait_exponential(multiplier=2, min=1, max=8),
        stop=stop_after_attempt(3),
        before_sleep=lambda rs: print(f"Retrying in {rs.next_action.sleep:.2f}s... (Attempt #{rs.attempt_number})")
    )
    def _generate_content_impl(self, prompt: str, **kwargs) -> str:
        try:
            if self._use_native:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text or ""
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content or ""
        except Exception as e:
            print(f"ERROR - AnthropicClient: API call failed for model {self.model}: {e}")
            raise

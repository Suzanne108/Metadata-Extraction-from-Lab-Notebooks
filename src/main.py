from providers.factory import LLMProviderFactory
from providers.base_provider import Provider

if __name__ == "__main__":
    provider_factory = LLMProviderFactory()
    provider_factory.set_provider(Provider.OPENAI)
    provider_factory.set_model("gpt-4o")

    provider = provider_factory.get_provider()

    provider.query("Hello, how are you?")
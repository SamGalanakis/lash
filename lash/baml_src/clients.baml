client<llm> DefaultClient {
  provider "openai-generic"
  options {
    base_url "https://openrouter.ai/api/v1"
    api_key env.KAML_API_KEY
    model "anthropic/claude-sonnet-4-5"
  }
}

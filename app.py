import gradio as gr
from answer import answer_question
from model_config import ModelConfig, get_llm


def gradio_chat(message, history, provider, api_key, model_name):
    if not provider:
        raise gr.Error("Please select a provider.")
    if not api_key:
        raise gr.Error("Please paste your API key.")
    
    config = ModelConfig(provider=provider, api_key=api_key, model_name=model_name or None)
    llm = get_llm(config)
    reply = answer_question(message, llm=llm, history=history)
    return reply


demo = gr.ChatInterface(
    fn=gradio_chat,
    type="messages",
    title="Local Drive Assistant",
    additional_inputs=[
        gr.Radio(["openai", "google", "anthropic"], label="Provider", value="openai"),
        gr.Textbox(label="API key (It is only stored local, safe to paste)", type="password"),
        gr.Textbox(label="Model name (optional)", value="gpt-4.1-mini"),
    ],
)


if __name__ == "__main__":
    demo.launch()

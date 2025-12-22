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

def reset_chat():
    return [],[]


with gr.Blocks() as demo:
    with gr.Sidebar(position="left"):
        gr.Markdown("## Settings")
        provider = gr.Radio(["openai", "google", "anthropic"], label="Provider", value="openai")
        api_key = gr.Textbox(label="API key (It is only stored local, safe to paste)", type="password")
        model = gr.Textbox(label="Model name (optional)", value="gpt-4.1-mini")

    bot = gr.Chatbot(type="messages",render=False)

    demo_chat = gr.ChatInterface(
        fn=gradio_chat,
        chatbot=bot,
        type="messages",
        title="Local Drive Assistant",
        additional_inputs=[provider, api_key, model],
    )
    reset_btn = gr.Button("Reset chat")
    reset_btn.click(fn=reset_chat, outputs=[bot, demo_chat.chatbot_state])
    


if __name__ == "__main__":
    demo.launch()

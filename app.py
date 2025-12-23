import gradio as gr
from pathlib import Path
from answer import answer_question
from model_config import ModelConfig, get_llm
from ingest import vectorize_db

def gradio_chat(message, history, provider, api_key, model_name, reranker_feature):
    if not api_key and provider != "ollama(free)":
        raise gr.Error("Please paste your API key.")
    
    config = ModelConfig(provider=provider, api_key=api_key, model_name=model_name or None)
    llm = get_llm(config)

    if reranker_feature:
        reply = answer_question(message, llm=llm, history=history, reranker_feature=True)
    else:
        reply = answer_question(message, llm=llm, history=history)
    
    return reply

def reset_chat():
    return [],[]


with gr.Blocks() as demo:
    gr.Markdown("""
        <h2 style="text-align:center; margin: 0.2rem 0 0.2rem 0; font-weight: 800;">
            Local RAG Assistant
        </h2>
        <div style="text-align:center; opacity:0.7; margin-bottom:   0.8rem;">
            Chat with your local docs
        </div>
        """)

    with gr.Sidebar(position="left"):
        gr.Markdown("## Settings")
        provider = gr.Radio(["openai", "google", "anthropic", "ollama(free)"], label="Provider", value="openai")
        api_key = gr.Textbox(label="API key (It is only stored local and no need to paste for ollama)", type="password")
        model = gr.Textbox(label="Model name (optional)")
        reranker_feature = gr.Checkbox(label="Enable Reranker Feature (more precise but slightly slower)", value=False)
        db_path = gr.FileExplorer(
            label="Database",
            root_dir=str(Path.home()),
            file_count="single",
            ignore_glob="**/.?*",
        )
        ingest_btn = gr.Button("Vectorize The Database (saved locally)")
        ingest_status = gr.Markdown()
        ingest_btn.click(fn=vectorize_db, inputs=[db_path], outputs=[ingest_status])


    bot = gr.Chatbot(type="messages",render=False)

    demo_chat = gr.ChatInterface(
        fn=gradio_chat,
        chatbot=bot,
        type="messages",
        title="",
        additional_inputs=[provider, api_key, model, reranker_feature],
    )
    reset_btn = gr.Button("Reset chat")
    reset_btn.click(fn=reset_chat, outputs=[bot, demo_chat.chatbot_state])
    


if __name__ == "__main__":
    demo.launch()

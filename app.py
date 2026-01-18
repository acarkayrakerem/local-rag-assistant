import json
import re
import gradio as gr
from pathlib import Path
from difflib import SequenceMatcher
from answer import answer_question
from model_config import ModelConfig, get_llm
from ingest import vectorize_db
from sdg import generate_sd

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
    return [], []

def run_sdg_task(provider, api_key, model_name):

    config = ModelConfig(
        provider=provider,
        api_key=api_key,
        model_name=model_name or None
    )
    llm = get_llm(config)
    dataset = generate_sd(llm)

    json_result= dataset.model_dump_json(indent=2)

    return json_result, json_result

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()

def similarity(a: str, b: str) -> float:
    ma_tokens = set(normalize(a).split())
    ex_tokens = set(normalize(b).split())

    if not ex_tokens:
        return 0.0

    matched = ex_tokens & ma_tokens
    return round(len(matched) / len(ex_tokens), 3)

def run_eval_task(sdg_json_str, provider, api_key, model_name, reranker_feature):
    if not sdg_json_str:
        raise gr.Error("Generate a test set first (SDG).")
    
    data = json.loads(sdg_json_str)
    pairs = data.get("pairs", [])

    config = ModelConfig(provider=provider, api_key=api_key, model_name=model_name or None)
    llm = get_llm(config)

    results = []
    scores = []
    i=0

    for p in pairs:
        q = p.get("question", "")
        expected = p.get("answer", "")

        model_answer = answer_question(q, llm=llm, history=[], reranker_feature=reranker_feature)
        score = similarity(model_answer, expected)
        scores.append(score)

        i+=1
    
        results.append({
            "progress": f"{i}/10",
            "question": q,
            "expected_answer": expected,
            "model_answer": model_answer,
            "similarity": score
        })
        yield json.dumps({"results": results}, indent=2)
    
    average = round(sum(scores) / len(scores), 3) if scores else 0.0

    yield json.dumps(
        {
            "progress": "done",
            "average_similarity": average,
            "results" : results
        },
        indent=2
    )


with gr.Blocks(theme=gr.themes.Soft(primary_hue="amber", secondary_hue="gray")) as demo:
    gr.Markdown("""
        <h2 style="text-align:center; margin: 0.5rem 0; font-weight: 800;">
            Local RAG Assistant Professional
        </h2>
        """)

    with gr.Tabs():
        with gr.Tab("Chat"):
            with gr.Sidebar(position="left"):
                gr.Markdown("## Settings")
                provider = gr.Radio(["openai", "google", "anthropic", "ollama(free)"], label="Provider", value="openai")
                api_key = gr.Textbox(label="API key", type="password")
                model = gr.Textbox(label="Model name (optional)")
                reranker_feature = gr.Checkbox(label="Enable Reranker Feature", value=False)
                
                gr.Markdown("---")
                db_path = gr.FileExplorer(
                    label="Database Selection",
                    root_dir=str(Path.home()),
                    file_count="single",
                    ignore_glob="**/.?*",
                )
                ingest_btn = gr.Button("Vectorize Database", variant="primary")
                ingest_status = gr.Markdown()
                ingest_btn.click(fn=vectorize_db, inputs=[db_path], outputs=[ingest_status])

            bot = gr.Chatbot(type="messages")
            
            demo_chat = gr.ChatInterface(
                fn=gradio_chat,
                chatbot=bot,
                type="messages",
                additional_inputs=[provider, api_key, model, reranker_feature],
            )
            
            reset_btn = gr.Button("Clear Conversation")
            reset_btn.click(fn=reset_chat, outputs=[bot, demo_chat.chatbot_state])

        with gr.Tab("SDG & Testing"):
            gr.Markdown("### 🛠 Synthetic Data Generation & Model Evaluation")

            sdg_state = gr.State("")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 1. Generate Questions")
                    sdg_btn = gr.Button("Generate Test Set")
                    sdg_output = gr.Code(label="Synthetic Q&A", language="json")
                    
                with gr.Column():
                    gr.Markdown("#### 2. Run Accuracy Test")
                    run_eval = gr.Button("Start Batch Evaluation", variant="primary")
                    eval_output = gr.Code(label="Evaluation Results", language="json")

            sdg_btn.click(fn=run_sdg_task, inputs=[provider, api_key, model], outputs=[sdg_output, sdg_state])
            run_eval.click(fn=run_eval_task, inputs=[sdg_state, provider, api_key, model, reranker_feature], outputs=[eval_output])

if __name__ == "__main__":
    demo.launch()
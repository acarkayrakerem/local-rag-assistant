import gradio as gr

def answer_question(question: str) -> str:
    return f"You asked {question}"

demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(label = "Your question"),
    outputs=gr.Textbox(label = "Answer"),
    title = "Prototype",
    description="Echo"
)
if __name__ == "__main__":
    demo.launch()
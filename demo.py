"""
Hinglish NER — Gradio Demo
Calls FastAPI /ner endpoint — does NOT load the model directly.
Requires FastAPI server running at http://localhost:8000
"""

import httpx
import gradio as gr

API_URL = "http://localhost:8000/ner"

COLORS = {
    "PERSON":   "#a8d8ea",
    "ORG":      "#f9d976",
    "LOCATION": "#b5ead7",
    "PRODUCT":  "#ffb7b2",
    "DATE":     "#d5aaff",
    "MONEY":    "#c7f2a4",
}

def api_response_to_highlighted(text: str, entities: list) -> list:
    """
    Convert FastAPI response entities into (text, label|None) tuples
    that gr.HighlightedText expects.
    Entities come with start/end character offsets from the API.
    """
    result = []
    last = 0
    for ent in sorted(entities, key=lambda e: e["start"]):
        if last < ent["start"]:
            result.append((text[last:ent["start"]], None))
        result.append((ent["text"], ent["label"]))
        last = ent["end"]
    if last < len(text):
        result.append((text[last:], None))
    return result

def predict(text: str):
    if not text.strip():
        return [], "Enter some Hinglish text above."

    try:
        response = httpx.post(API_URL, json={"text": text}, timeout=30)
        response.raise_for_status()
        data = response.json()
    except httpx.ConnectError:
        return [], "❌ Cannot connect to API. Is FastAPI running at localhost:8000?"
    except Exception as e:
        return [], f"❌ Error: {e}"

    entities = data["entities"]
    highlighted = api_response_to_highlighted(data["text"], entities)
    entity_list = "\n".join(
        f"{e['label']:12s} | {e['text']}" for e in entities
    ) or "No entities found."

    return highlighted, entity_list

# ── UI ────────────────────────────────────────────────────────────────────────
EXAMPLES = [
    ["Virat ne Mumbai mein Zomato se biryani order ki aur 500 rupaye pay kiye."],
    ["Priya Google mein kaam karti hai aur kal Delhi jayegi."],
    ["Shah Rukh Khan ki nayi film Netflix pe aayegi next Friday."],
    ["Rohit ne Amazon se iPhone 15 kharida sirf 80000 mein."],
]

with gr.Blocks(title="Hinglish NER Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🇮🇳 Hinglish NER — Named Entity Recognition
        Extract **PERSON, ORG, LOCATION, PRODUCT, DATE, MONEY** from Hindi-English mixed text.
        > Model: XLM-RoBERTa fine-tuned on ~1,000 manually annotated Hinglish sentences · F1 = 74.10
        """
    )

    inp = gr.Textbox(
        label="Hinglish Text",
        placeholder="e.g. Virat ne Mumbai mein Zomato se biryani order ki...",
        lines=3,
    )

    btn = gr.Button("Extract Entities", variant="primary")

    with gr.Row():
        out_highlighted = gr.HighlightedText(
            label="Highlighted Entities",
            color_map=COLORS,
            show_legend=True,
            combine_adjacent=False,
        )
        out_table = gr.Textbox(label="Entity List", lines=8)

    gr.Examples(examples=EXAMPLES, inputs=inp)

    btn.click(fn=predict, inputs=inp, outputs=[out_highlighted, out_table])
    inp.submit(fn=predict, inputs=inp, outputs=[out_highlighted, out_table])

if __name__ == "__main__":
    demo.launch(share=False)
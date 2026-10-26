import streamlit as st
from google import genai
from google.genai import types
from openai import OpenAI
import anthropic

st.set_page_config(page_title="RedAI Multi-Engine", page_icon="🔴")
st.title("🔴 RedAI: Multi-Engine Orchestrator")
st.caption("Synthesizing knowledge from Gemini, ChatGPT, and Claude simultaneously.")

# Fetch keys securely from cloud secrets or local variables
GEMINI_KEY = st.secrets.get("GEMINI_KEY", "")
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "")
CLAUDE_KEY = st.secrets.get("ANTHROPIC_API_KEY", "YOUR_CLAUDE_KEY")

SYSTEM_PROMPT = """
Your name is RedAI. You are an advanced AI Orchestrator. 
Your unique power is that you gather perspectives from Gemini, ChatGPT, and Claude.
When a user asks a question, you will be provided with the raw answers from those three models.
Your job is to read all three, extract the best details, resolve contradictions, and present a single master response.
Always respond as RedAI, and mention what the various models agreed on if it adds value!
"""

def get_chatgpt_response(prompt):
    try:
        client = OpenAI(api_key=OPENAI_KEY)
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices.message.content
    except Exception as e:
        return f"OpenAI Error: {e}"

def get_claude_response(prompt):
    try:
        client = anthropic.Anthropic(api_key=CLAUDE_KEY)
        message = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content.text
    except Exception as e:
        return f"Anthropic Error: {e}"

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.gemini_client = genai.Client(api_key=GEMINI_KEY)
    st.session_state.chat = st.session_state.gemini_client.chats.create(
        model="gemini-3.1-flash-lite",
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.5)
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🔴" if msg["role"] == "assistant" else None):
        st.markdown(msg["content"])

if user_query := st.chat_input("Ask RedAI anything..."):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant", avatar="🔴"):
        with st.status("🔴 RedAI Orchestrating Answers...", expanded=True) as status:
            st.write("📥 Querying OpenAI (ChatGPT)...")
            gpt_ans = get_chatgpt_response(user_query)
            st.write("📥 Querying Anthropic (Claude)...")
            claude_ans = get_claude_response(user_query)
            st.write("🧠 Synthesizing into RedAI Master Brain...")
            
            orchestrator_payload = f"User Query: {user_query}\n\n[ChatGPT]: {gpt_ans}\n\n[Claude]: {claude_ans}"
            status.update(label="Synthesis complete!", state="complete", expanded=False)

        try:
            response = st.session_state.chat.send_message(orchestrator_payload)
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
        except Exception as e:
            st.error(f"RedAI Synthesis Error: {e}")

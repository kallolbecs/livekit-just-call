import asyncio
import yaml

from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import azure, silero, openai

load_dotenv()

# Load sales prompt from YAML file
def load_sales_prompt():
    with open('sales_prompt.yaml', 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data

# Default callback for before_tts_cb that returns the text unchanged.
def identity_before_tts(agent, text):
    return text

# Default async callback for before_llm_cb that returns the chat context unchanged.
async def identity_before_llm(agent, chat_ctx):
    return chat_ctx

async def entrypoint(ctx: JobContext):
    # Load the sales prompt
    sales_data = load_sales_prompt()
    sales_script = sales_data.get('sales_script', '')

    # Create system prompt with sales script and language instructions
    system_prompt = (
        "You are a female outbound sales voice assistant making calls to potential customers. "
        "Your primary language is Hindi (hi-IN) with English (en-IN) as a secondary language. "
        "You should primarily speak in Hindi but can use English for technical terms. "
        "Only use the English words that are specifically allowed in the language settings. "
        "Keep your responses natural, conversational, and follow the sales script below. "
        "Speak clearly and be respectful of the customer's time and preferences. "
        "Do not say sir repeatedly, and make the responses concise unless asked for details. "
        "Encourage the retailer to stock the product immediately after the initial conversation. "
        f"\n\nSales Script:\n{sales_script}"
    )

    # Initialize the conversation context
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=system_prompt,
    )
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Configure Azure STT with Hindi as the primary language.
    azure_stt = azure.STT(
        languages=['hi-IN', 'en-IN'],  # Specify Hindi as the primary language and English as secondary.]
    )

    # Configure OpenAI's LLM with Azure configuration.
    azure_llm = openai.LLM.with_azure(
        model="gpt-4o-mini",
        api_version="2024-08-01-preview",
    )

    # Configure Azure TTS with Hindi voice settings.
    azure_tts = azure.TTS(
        voice="hi-IN-AnanyaNeural",  # Specify the Hindi voice.
        language="hi-IN",
    )

    # CHANGE 1: Lower VAD's min_speech_duration for quicker speech detection.
    vad = silero.VAD.load(min_speech_duration=0.1)

    # CHANGE 2: Create a VoicePipelineAgent with additional parameters for interruption handling,
    # VAD settings, and default callbacks for LLM and TTS.
    agent = VoicePipelineAgent(
        vad=vad,
        stt=azure_stt,
        llm=azure_llm,
        tts=azure_tts,
        chat_ctx=initial_ctx,
        allow_interruptions=True,         # Enable user interruptions.
        interrupt_speech_duration=0.5,      # Minimum duration (in seconds) for detecting intentional interruptions.
        interrupt_min_words=0,              # Minimum words required to treat an interruption as valid.
        min_endpointing_delay=0.5,          # Minimum silence duration to consider the end of a turn.
        before_llm_cb=identity_before_llm,  # Provide a default callback that returns the chat context unchanged.
        before_tts_cb=identity_before_tts,  # Provide a default callback that returns text unchanged.
    )

    # CHANGE 3: Start the agent.
    agent.start(ctx.room)

    # Start with a Hindi greeting.
    initial_greeting = (
        "नमस्ते सर, मैं SP Sales की तरफ से sgithruti बात कर रही हूँ। "
        "क्या आप अभी 2 minute बात कर सकते हैं?"
    )
    await asyncio.sleep(1)
    await agent.say(initial_greeting, allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

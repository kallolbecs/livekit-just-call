import asyncio
import yaml

from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import azure, silero, openai
from api import AssistantFnc

load_dotenv()

# Load sales prompt from YAML file
def load_sales_prompt():
    with open('sales_prompt.yaml', 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data

async def entrypoint(ctx: JobContext):
    # Load the sales prompt
    sales_data = load_sales_prompt()
    sales_script = sales_data.get('sales_script', '')

    # Create system prompt with sales script and language instructions
    system_prompt = (
        "You are a male outbound sales voice assistant making calls to potential customers. "
        "Your primary language is Hindi (hi-IN) with English (en-IN) as a secondary language. "
        "You should primarily speak in Hindi but can use English for technical terms. "
        "Only use the English words that are specifically allowed in the language settings. "
        "Keep your responses natural, conversational, and follow the sales script below. "
        "Speak clearly and be respectful of the customer's time and preferences. "
        f"\n\nSales Script:\n{sales_script}"
    )

    # Initialize the conversation context
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=system_prompt,
    )
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    fnc_ctx = AssistantFnc()

    # Configure Azure STT with Hindi as the primary language.
    azure_stt = azure.STT(
        languages=['hi-IN','en-IN']
    )

    # Use OpenAI's LLM with Azure configuration.
    azure_llm = openai.LLM.with_azure(
        model="gpt-4o-mini",
        api_version="2024-08-01-preview",
    )

    # Configure Azure TTS with Hindi voice settings by passing parameters during instantiation.
    azure_tts = azure.TTS(
        voice="hi-IN-AnanyaNeural",      # Specify the Hindi voice.
        language="hi-IN",                # Set the language to Hindi.
       # speech_synthesis_output_format="audio-24khz-48kbitrate-mono-mp3"
    )

    # Create the voice assistant using the configured plugins.
    assistant = VoiceAssistant(
        vad=silero.VAD.load(min_speech_duration=0.5),
        stt=azure_stt,
        llm=azure_llm,
        tts=azure_tts,
        chat_ctx=initial_ctx,
        fnc_ctx=fnc_ctx,
    )
    assistant.start(ctx.room)

    # Start with a Hindi greeting
    initial_greeting = "नमस्ते सर, मैं श्रुति बोल रही हूँ रॉकेट सिंग कंपनी से। क्या आप हमारे नए product के बारे में जानना चाहेंगे?"
    await asyncio.sleep(1)
    await assistant.say(initial_greeting, allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

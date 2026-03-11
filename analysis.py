import json
import time
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from newspaper import Article as Scraper
from supabase import create_client, Client

# Import the baseline bias dictionary from your config file
from config import SOURCE_BIAS

# Load Environment
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_KEY")
os.environ["GOOGLE_CLOUD_PROJECT"] = os.getenv("GOOGLE_CLOUD_PROJECT")
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1" 

# Initialize the client for Vertex AI
client = genai.Client(
    vertexai=True,
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location=os.environ["GOOGLE_CLOUD_LOCATION"]
)

def get_body_content(url):
    """Scrapes the full body of the article."""
    try:
        a = Scraper(url, request_timeout=5, config=config)
        a.download()
        a.parse()
        return a.text[:3500] 
    except Exception as e:
        print(f"  [Scrape Error] {url}: {e}")
        return None

def clean_json_response(text):
    """Cleans the AI response to extract only the JSON part."""
    try:
        # Find content between { and }
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)
    except Exception as e:
        print(f"  [Parse Error] Failed to clean JSON: {e}")
        print(text)
        return None

def get_ai_insight(headline, body, source_name):
    google_search_tool = types.Tool(google_search=types.GoogleSearch())
    baseline = SOURCE_BIAS.get(source_name.lower(), 0.0)

    system_instruction = (
        "You are a UK Media Structural Auditor. It is Feb 2026. "
        "CRITICAL: You must provide your analysis in a RAW JSON block. "
        "Do not provide a conversational summary. Do not use markdown headers. "
        "Only output the JSON object."
    )
    
    # We add "STRICT JSON" to the end to remind the model right before it generates
    prompt = f"""
    Analyze this article from {source_name}: {headline}
    CONTEXT: Source historical baseline bias is {baseline}, do not change left -> right or vice versa, sources tend to stick to their side. (IMPORTANT)
    Also just because the article is talking about the right, does not mean they are right, think about the framing.
    Body: {body}

    OUTPUT THE FOLLOWING JSON DATA ONLY:
    {{
      "bias_score": float, (-1.5 to 1.5 where -1.5 is extreme left, and 1.5 is extreme right)
      "fact_score": int, (based off of the truth of the facts stated)
      "insight": "Briefly describe framing.",
      "key_phrases": []
    }}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[google_search_tool],
                temperature=0.1,
            ),
            contents=prompt
        )
        
        # Check if text exists, then parse
        if response.text:
            return clean_json_response(response.text)
        return None
    except Exception as e:
        print(f"  [API Error] {e}")
        return None

def run_analysis():
    print("--- STARTING DB-DRIVEN AUDIT ---")
    
    response = supabase.table("articles") \
        .select("*") \
        .eq("insight", "AI analysis queued...") \
        .execute()

    pending_articles = response.data

    if not pending_articles:
        print("No pending articles found.")
        return

    print(f"Found {len(pending_articles)} articles to analyze.")

    for article in pending_articles:
        a_id = article['id']
        source = article['source']
        headline = article['headline']
        
        print(f"Analyzing {source}: {headline[:50]}...")
        
        full_body = get_body_content(article['url'])
        input_text = full_body if full_body and len(full_body) > 200 else article['summary']
        
        # --- RETRY LOGIC START ---
        analysis = None
        for attempt in range(2):  # Try twice
            analysis = get_ai_insight(headline, input_text, source)
            if analysis:
                break
            print(f"  [Retry] Attempt {attempt + 1} failed for {a_id}. Retrying...")
            time.sleep(2) # Short breather between retries

        if analysis:
            try:
                supabase.table("articles").update(analysis).eq("id", a_id).execute()
                print(f"  [Success] Saved to Supabase.")
            except Exception as e:
                print(f"  [DB Update Error] Failed to update {a_id}: {e}")
        else:
            # --- PURGE LOGIC ---
            print(f"  [Critical Failure] Could not parse JSON for {a_id} after 2 tries. PURGING.")
            try:
                supabase.table("articles").delete().eq("id", a_id).execute()
                print(f"  [Deleted] Removed broken article from DB.")
            except Exception as e:
                print(f"  [DB Delete Error] Failed to purge {a_id}: {e}")
        # --- RETRY LOGIC END ---
        
        time.sleep(1) 

if __name__ == "__main__":
    run_analysis()
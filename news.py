import hashlib
import feedparser
from datetime import datetime, timezone
from typing import List
from concurrent.futures import ThreadPoolExecutor

from newspaper import Article as Scraper
from newspaper import Config
from sentence_transformers import SentenceTransformer, util
from supabase import create_client, Client
import os
from dotenv import load_dotenv

# Import your custom models and config
from models import Article, Story
from config import RSS_FEEDS, SOURCE_BIAS

load_dotenv()
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
model = SentenceTransformer('all-MiniLM-L6-v2')

config = Config()
config.user_agent = 'NarrativeUK-Bot/1.0 (+https://narrativeuk.co.uk/about)'
config.request_timeout = 10

SIMILARITY_THRESHOLD = 0.55
ARTICLE_CAP = 40 

def generate_article_id(url: str) -> str:
    return hashlib.sha256(url.encode('utf-8')).hexdigest()[:12]

def get_full_content(article_obj: Article) -> str:
    try:
        a = Scraper(article_obj.url, config=config)
        a.download()
        a.parse()
        return f"{a.title} {a.text[:1000]}"
    except:
        return article_obj.headline

def build_smart_stories(articles: List[Article]) -> List[Story]:
    if not articles: return []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        all_texts = list(executor.map(get_full_content, articles))

    all_embeddings = model.encode(all_texts, convert_to_tensor=True)
    unique_stories = []
    
    for i, article in enumerate(articles):
        matched = False
        article_emb = all_embeddings[i]

        for story in unique_stories:
            score = util.cos_sim(article_emb, all_embeddings[story._seed_idx])
            if score > SIMILARITY_THRESHOLD:
                if not any(existing.source == article.source for existing in story.articles):
                    story.articles.append(article)
                matched = True
                break
        
        if not matched:
            s_id = f"story-{hashlib.md5(article.headline.encode()).hexdigest()[:10]}"
            new_story = Story(story_id=s_id, topic=article.headline, articles=[article])
            new_story._seed_idx = i
            unique_stories.append(new_story)

    return [s for s in unique_stories if len(s.articles) >= 3]

def push_and_clean_db(stories: List[Story]):
    current_story_ids = [s.story_id for s in stories]
    
    # NEW LOGIC: Clear staging at the start of each run to prevent stale clusters
    supabase.table("articles_staging").delete().neq("id", "0").execute()
    
    for story in stories:
        supabase.table("stories").upsert({
            "story_id": story.story_id,
            "topic": story.topic,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }).execute()

        for a in story.articles:
            # Check if article is already in the LIVE audited table
            existing_live = supabase.table("articles").select("id").eq("id", a.id).execute()
            
            payload = {
                "id": a.id,
                "story_id": story.story_id,
                "source": a.source,
                "headline": a.headline,
                "summary": a.summary,
                "url": a.url,
                "published_at": a.published_at.isoformat(),
                "bias_score": a.bias_score
            }
            
            if not existing_live.data:
                # OPTION 1: Only push to staging if it hasn't been audited yet
                supabase.table("articles_staging").insert(payload).execute()
            else:
                # If it's already live, just update the story_id mapping in case clustering changed
                supabase.table("articles").update({"story_id": story.story_id}).eq("id", a.id).execute()

    if current_story_ids:
        supabase.table("stories").delete().not_.in_("story_id", current_story_ids).execute()

def get_stories(force_refresh: bool = False):
    raw_articles = []
    for source_id, feed_url in RSS_FEEDS.items():
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:ARTICLE_CAP]:
            published = getattr(entry, "published_parsed", None)
            dt = datetime(*published[:6], tzinfo=timezone.utc) if published else datetime.now(timezone.utc)
            raw_articles.append(Article(
                id=generate_article_id(entry.link),
                source=source_id,
                headline=getattr(entry, "title", ""),
                summary=getattr(entry, "summary", ""),
                url=getattr(entry, "link", ""),
                published_at=dt,
                bias_score=SOURCE_BIAS.get(source_id, 0.0)
            ))
    stories = build_smart_stories(raw_articles)
    push_and_clean_db(stories)

if __name__ == "__main__":
    get_stories()
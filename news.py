import hashlib
import feedparser
from datetime import datetime, timezone
from typing import List
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

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
    # URL Normalization: Strip query parameters to prevent duplicate IDs for the same article
    clean_url = urlparse(url)._replace(query="").geturl()
    return hashlib.sha256(clean_url.encode('utf-8')).hexdigest()[:12]

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
    
    # Pre-deduplicate raw articles by ID to ensure we don't process the same link twice
    seen_ids = set()
    unique_raw_articles = []
    for a in articles:
        if a.id not in seen_ids:
            unique_raw_articles.append(a)
            seen_ids.add(a.id)

    with ThreadPoolExecutor(max_workers=10) as executor:
        all_texts = list(executor.map(get_full_content, unique_raw_articles))

    all_embeddings = model.encode(all_texts, convert_to_tensor=True)
    unique_stories = []
    
    for i, article in enumerate(unique_raw_articles):
        matched = False
        article_emb = all_embeddings[i]

        for story in unique_stories:
            score = util.cos_sim(article_emb, all_embeddings[story._seed_idx])
            if score > SIMILARITY_THRESHOLD:
                # FIX: Strictly allow only ONE article per source per story cluster
                if not any(existing.source == article.source for existing in story.articles):
                    story.articles.append(article)
                matched = True
                break
        
        if not matched:
            s_id = f"story-{hashlib.md5(article.headline.encode()).hexdigest()[:10]}"
            new_story = Story(story_id=s_id, topic=article.headline, articles=[article])
            new_story._seed_idx = i
            unique_stories.append(new_story)

    return unique_stories

def push_and_clean_db(stories: List[Story]):
    if not stories:
        print("No stories clustered. Skipping DB push.")
        return

    # 1. Clear staging to prevent "ghost" articles from previous runs
    try:
        supabase.table("articles_staging").delete().neq("id", "0").execute()
    except Exception as e:
        print(f"  [DB Warning] Could not clear staging: {e}")

    for story in stories:
        if len(story.articles) >= 3:
            
            supabase.table("stories").upsert({
                "story_id": story.story_id,
                "topic": story.topic,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }).execute()

            for a in story.articles:
                # Check if this article is already in the LIVE 'articles' table
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
                    # New article: push to staging for the Brain to audit
                    supabase.table("articles_staging").upsert(payload).execute()
                else:
                    # Existing article: update the story_id mapping
                    supabase.table("articles").update({"story_id": story.story_id}).eq("id", a.id).execute()

    # 4. SMART CLEANUP
    try:
        supabase.rpc('delete_orphaned_stories').execute()
    except Exception as e:
        print(f"  [DB Warning] Smart cleanup failed: {e}")
    
    print(f"--- DB SYNC COMPLETE: {len(stories)} clusters processed ---")

def get_stories(force_refresh: bool = False):
    raw_articles = []
    for source_id, feed_url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(feed_url)
            if feed.bozo:
                print(f"  [Feed Warning] Issue with {source_id}: {feed.bozo_exception}")

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
        except Exception as e:
            print(f"  [Feed Error] Skipped {source_id}: {e}")
            continue 

    if not raw_articles:
        print("No articles fetched. Exiting.")
        return

    stories = build_smart_stories(raw_articles)
    push_and_clean_db(stories)

if __name__ == "__main__":
    get_stories()
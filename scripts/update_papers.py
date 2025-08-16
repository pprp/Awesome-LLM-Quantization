#!/usr/bin/env python3
"""
ArXiv Paper Update Script for LLM Quantization Repository
This script searches for new papers related to LLM quantization and updates the README.md
"""

import re
import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
import requests
import feedparser
from dateutil.parser import parse as parse_date

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ArXivTracker:
    """Handles tracking of processed arXiv papers to avoid duplicates"""
    
    def __init__(self, tracking_file: str = "scripts/processed_papers.json"):
        self.tracking_file = tracking_file
        self.processed_ids: Set[str] = set()
        self.load_processed_ids()
        
    def load_processed_ids(self):
        """Load previously processed arXiv IDs from JSON file"""
        try:
            if os.path.exists(self.tracking_file):
                with open(self.tracking_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.processed_ids = set(data.get('processed_arxiv_ids', []))
                    logger.info(f"Loaded {len(self.processed_ids)} previously processed arXiv IDs")
            else:
                logger.info("No tracking file found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading tracking file: {e}")
            
    def save_processed_ids(self):
        """Save processed arXiv IDs to JSON file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.tracking_file), exist_ok=True)
            
            # Prepare data structure
            data = {
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_papers": len(self.processed_ids),
                "processed_arxiv_ids": sorted(list(self.processed_ids))
            }
            
            with open(self.tracking_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Updated tracking file with {len(self.processed_ids)} arXiv IDs")
            
        except Exception as e:
            logger.error(f"Error saving tracking file: {e}")
            
    def is_processed(self, arxiv_id: str) -> bool:
        """Check if an arXiv paper has already been processed"""
        return arxiv_id in self.processed_ids
        
    def add_processed(self, arxiv_id: str):
        """Add an arXiv ID to the processed list"""
        self.processed_ids.add(arxiv_id)
        
    def filter_new_papers(self, papers: List[Dict]) -> List[Dict]:
        """Filter out papers that have already been processed"""
        new_papers = []
        
        for paper in papers:
            if not self.is_processed(paper['id']):
                new_papers.append(paper)
            else:
                logger.info(f"Skipping already processed paper: {paper['title']}")
                
        logger.info(f"Filtered {len(papers)} papers -> {len(new_papers)} new papers")
        return new_papers

class ArXivPaperFetcher:
    """Handles fetching papers from arXiv API"""
    
    BASE_URL = "http://export.arxiv.org/api/query"
    
    def __init__(self):
        self.search_terms = [
            "quantization",
            "large language model", 
            "LLM",
            "compression",
            "low-bit",
            "post-training quantization",
            "quantization aware training"
        ]
        
    def build_query(self, days_back: int = 1) -> str:
        """Build arXiv API query string - simplified and working approach"""
        
        # Start with simple, proven query patterns
        query_parts = [
            # Basic combinations that work reliably
            'all:"quantization" AND all:"language model"',
            'all:"quantized" AND all:"LLM"',
            'ti:"GPTQ"',
            'ti:"AWQ"', 
            'ti:"BitNet"',
            'ti:"QLoRA"',
            'abs:"post-training quantization"',
            'cat:cs.CL AND ti:"quantization"'
        ]
        
        query = " OR ".join(query_parts)
        
        logger.info(f"Built simplified query with {len(query_parts)} parts")
        logger.info(f"Query: {query[:200]}...")  # Log first 200 chars for debugging
        return query
        
    def is_relevant_paper(self, paper: Dict) -> bool:
        """Check if paper is actually relevant to LLM quantization"""
        title_lower = paper['title'].lower()
        abstract_lower = paper['summary'].lower()
        
        # Must have quantization-related terms
        quant_keywords = [
            'quantization', 'quantized', 'quantize', 'low-bit', 'low precision',
            'int4', 'int8', '4-bit', '8-bit', '2-bit', '3-bit', 'ptq', 'qat',
            'gptq', 'awq', 'smoothquant', 'bitnet', 'qlora', 'compression'
        ]
        
        # Must have LLM-related terms  
        llm_keywords = [
            'large language model', 'language model', 'llm', 'transformer', 
            'bert', 'gpt', 'llama', 'chatgpt', 'generative', 'nlp'
        ]
        
        # Check if both types of keywords are present
        has_quant = any(keyword in title_lower or keyword in abstract_lower for keyword in quant_keywords)
        has_llm = any(keyword in title_lower or keyword in abstract_lower for keyword in llm_keywords)
        
        # Additional check: exclude papers that are clearly not about LLM quantization
        exclude_keywords = [
            'motion generation', 'video', 'image', 'computer vision', 'cv',
            'robotics', 'control', 'mechanical', 'hardware design', 'circuit'
        ]
        
        has_exclude = any(keyword in title_lower or keyword in abstract_lower for keyword in exclude_keywords)
        
        is_relevant = has_quant and has_llm and not has_exclude
        
        if not is_relevant:
            logger.info(f"Filtered out irrelevant paper: {paper['title']}")
            
        return is_relevant
        
    def fetch_recent_papers(self, days_back: int = 1) -> List[Dict]:
        query = self.build_query(days_back)
        
        params = {
            'search_query': query,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending',
            'max_results': 50  # Get more results to filter by date
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            logger.info(f"ArXiv API response status: {response.status_code}")
            logger.info(f"Response content length: {len(response.content)} bytes")
            
            # Parse the Atom feed
            feed = feedparser.parse(response.content)
            
            logger.info(f"Feed parsed. Total entries: {len(feed.entries)}")
            
            papers = []
            cutoff_date = datetime.now() - timedelta(days=days_back)
            logger.info(f"Looking for papers newer than: {cutoff_date}")
            
            for entry in feed.entries:
                # Parse submission date
                published_date = parse_date(entry.published)
                
                # Only include papers from the last N days AND are relevant
                if published_date.replace(tzinfo=None) > cutoff_date:
                    paper = {
                        'id': entry.id.split('/')[-1],  # Extract arXiv ID
                        'title': entry.title.replace('\n', ' ').strip(),
                        'authors': [author.name for author in entry.authors],
                        'summary': entry.summary.replace('\n', ' ').strip(),
                        'published': published_date,
                        'link': entry.id,
                        'pdf_url': entry.id.replace('/abs/', '/pdf/') + '.pdf'
                    }
                    
                    # Apply relevance filter
                    if self.is_relevant_paper(paper):
                        papers.append(paper)
                        logger.info(f"Added relevant paper: {paper['title']}")
                    
                    
            logger.info(f"Found {len(papers)} recent papers")
            return papers
            
        except Exception as e:
            logger.error(f"Error fetching papers: {e}")
            return []

class PaperSummarizer:
    """Handles paper summarization using LLM APIs"""
    
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openai_base_url = os.getenv('OPENAI_BASE_URL')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        
    def summarize_paper(self, paper: Dict) -> Optional[str]:
        """Summarize a paper using available LLM APIs"""
        
        prompt = f"""
        Please provide a concise summary of this research paper about LLM quantization in 2-3 sentences. 
        Focus on the key contributions, methods, and results. End with relevant hashtags.
        
        Title: {paper['title']}
        Authors: {', '.join(paper['authors'])}
        Abstract: {paper['summary'][:1500]}...
        
        Format your response as a single paragraph summary followed by hashtags like #PTQ #4-bit #LLM
        """
        
        # Try OpenAI first
        if self.openai_api_key:
            try:
                summary = self._summarize_with_openai(prompt)
                if summary:
                    return summary
            except Exception as e:
                logger.warning(f"OpenAI API failed: {e}")
                
        # Try Anthropic as fallback
        if self.anthropic_api_key:
            try:
                summary = self._summarize_with_anthropic(prompt)
                if summary:
                    return summary
            except Exception as e:
                logger.warning(f"Anthropic API failed: {e}")
                
        # Return a basic summary if APIs fail
        return self._generate_basic_summary(paper)
        
    def _summarize_with_openai(self, prompt: str) -> Optional[str]:
        """Summarize using OpenAI API"""
        try:
            import openai
            
            # Initialize client with custom base URL if provided
            if self.openai_base_url:
                client = openai.OpenAI(
                    api_key=self.openai_api_key,
                    base_url=self.openai_base_url
                )
            else:
                client = openai.OpenAI(api_key=self.openai_api_key)
            
            response = client.chat.completions.create(
                model="gemini-2.0-flash",
                messages=[
                    {"role": "system", "content": "You are an expert in machine learning and LLM quantization research."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.4
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return None
            
    def _summarize_with_anthropic(self, prompt: str) -> Optional[str]:
        """Summarize using Anthropic API"""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return None
            
    def _generate_basic_summary(self, paper: Dict) -> str:
        """Generate a basic summary when APIs are unavailable"""
        return f"This paper presents research on {paper['title'].lower()}. {paper['summary'][:150]}... <br/>#Quantization #LLM"

class ReadmeUpdater:
    """Handles updating the README.md file"""
    
    def __init__(self, readme_path: str = "README.md"):
        self.readme_path = readme_path
        
    def load_readme(self) -> str:
        """Load the current README.md content"""
        try:
            with open(self.readme_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading README: {e}")
            return ""
            
    def save_readme(self, content: str) -> bool:
        """Save the updated README.md content"""
        try:
            with open(self.readme_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            logger.error(f"Error saving README: {e}")
            return False
            
    def format_paper_entry(self, paper: Dict, summary: str) -> str:
        """Format a paper entry for the README table"""
        # Format authors - limit to first 3 for space
        authors = paper['authors'][:3]
        if len(paper['authors']) > 3:
            authors_str = ', '.join(authors) + ', et al.'
        else:
            authors_str = ', '.join(authors)
            
        # Format date
        date_str = paper['published'].strftime('%Y-%m-%d')
        
        # Create the table row
        entry = f"| Arxiv{paper['published'].year} <br/> [{paper['title']}]({paper['link']}) <br/> {authors_str} | {summary} |"
        
        return entry
        
    def update_papers_table(self, content: str, new_papers: List[Dict], summaries: List[str]) -> str:
        """Update the papers table with new entries"""
        if not new_papers:
            logger.info("No new papers to add")
            return content
            
        # Find the papers table
        table_start = content.find("| Title & Author & Link")
        if table_start == -1:
            logger.error("Could not find papers table in README")
            return content
            
        # Find the end of the table by looking for the next section or end of file
        # Look for the next markdown header (starting with #) after the table
        next_section = content.find("\n#", table_start)
        if next_section == -1:
            # If no next section found, append at the end of content
            table_end = len(content)
        else:
            # Find the last table row before the next section
            table_end = next_section
            
        logger.info(f"Adding {len(new_papers)} new papers to README at the end of table")
            
        # Insert new papers at the end of the table
        new_entries = []
        for paper, summary in zip(new_papers, summaries):
            entry = self.format_paper_entry(paper, summary)
            new_entries.append(entry)
            
        # Insert new entries at the end of the table
        new_content = (
            content[:table_end] + 
            '\n'.join(new_entries) + '\n' +
            content[table_end:]
        )
        
        return new_content

def main():
    """Main execution function"""
    logger.info("Starting daily arXiv paper update")
    
    # Initialize components
    tracker = ArXivTracker()
    fetcher = ArXivPaperFetcher()
    summarizer = PaperSummarizer()
    updater = ReadmeUpdater()
    
    # Check if we have API keys
    if not (summarizer.openai_api_key or summarizer.anthropic_api_key):
        logger.error("No LLM API keys found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        sys.exit(1)
        
    # Try multiple day ranges to find new papers
    day_ranges = [30, 60, 80]
    new_papers = []
    
    for days_back in day_ranges:
        logger.info(f"Fetching recent papers from arXiv (last {days_back} days)...")
        papers = fetcher.fetch_recent_papers(days_back=days_back)
        
        if not papers:
            logger.info(f"No papers found from arXiv in last {days_back} days")
            continue
            
        # Filter out papers we've already processed
        new_papers = tracker.filter_new_papers(papers)
        
        if new_papers:
            logger.info(f"Found {len(new_papers)} new unprocessed papers in last {days_back} days")
            break
        else:
            logger.info(f"No new unprocessed papers found in last {days_back} days")
    
    if not new_papers:
        logger.info("No new unprocessed papers found after trying all day ranges - exiting without changes")
        sys.exit(0)
        
    # Limit to maximum 3 papers per day
    MAX_PAPERS_PER_DAY = 1
    if len(new_papers) > MAX_PAPERS_PER_DAY:
        logger.info(f"Found {len(new_papers)} new papers, limiting to {MAX_PAPERS_PER_DAY} papers per day")
        new_papers = new_papers[:MAX_PAPERS_PER_DAY]
        
    logger.info(f"Processing {len(new_papers)} new papers...")
    
    # Process each paper separately and create individual commits
    total_added = 0
    
    for i, paper in enumerate(new_papers):
        logger.info(f"Processing paper {i+1}/{len(new_papers)}: {paper['title']}")
        
        # Mark paper as processed first to avoid reprocessing even if update fails
        tracker.add_processed(paper['id'])
        
        # Summarize the paper
        summary = summarizer.summarize_paper(paper)
        
        # Load current README
        readme_content = updater.load_readme()
        if not readme_content:
            logger.error("Failed to load README.md")
            continue
            
        # Update with single paper
        updated_content = updater.update_papers_table(readme_content, [paper], [summary])
        
        # Check if content actually changed
        if len(updated_content) != len(readme_content):
            # Save the updated README
            if updater.save_readme(updated_content):
                logger.info(f"Successfully added paper: {paper['title']}")
                total_added += 1
                
                # Create individual commit for this paper
                os.system(f'git add README.md')
                commit_message = f'Add paper: {paper["title"][:60]}{"..." if len(paper["title"]) > 60 else ""}'
                os.system(f'git commit -m "{commit_message}"')
                
            else:
                logger.error(f"Failed to save README.md for paper: {paper['title']}")
        else:
            logger.info(f"Paper content didn't change README, skipped: {paper['title']}")
            
        # Add delay to avoid rate limiting
        time.sleep(1)
    
    # Always save the tracking file regardless of whether papers were added to README
    tracker.save_processed_ids()
    
    if total_added > 0:
        logger.info(f"Successfully processed {total_added} new papers with individual commits")
        logger.info("Git commits created - workflow should detect changes")
    else:
        logger.info("No new papers were added")
        
    logger.info("Paper update completed successfully")

if __name__ == "__main__":
    main()
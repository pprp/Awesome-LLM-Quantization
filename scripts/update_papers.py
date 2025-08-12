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
from typing import List, Dict, Optional
import requests
import feedparser
from dateutil.parser import parse as parse_date

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        """Build arXiv API query string - simplified and tested approach"""
        
        # Use multiple simple queries that we know work
        query_parts = [
            # Direct phrase searches that work
            'all:"LLM quantization"',
            'all:"quantization of large language models"',
            'all:"quantized language model"',
            'all:"quantization for LLM"',
            
            # Title searches for specific methods
            'ti:"GPTQ"',
            'ti:"AWQ"', 
            'ti:"SmoothQuant"',
            'ti:"BitNet"',
            'ti:"QLoRA"',
            'ti:"OmniQuant"',
            'ti:"QuIP"',
            'ti:"AQLM"',
            'ti:"BiLLM"',
            
            # Abstract + title combinations (simple)
            '(ti:"quantization" AND abs:"large language model")',
            '(ti:"quantized" AND abs:"LLM")',
            '(ti:"4-bit" AND abs:"language model")',
            '(ti:"8-bit" AND abs:"transformer")',
            '(ti:"low-bit" AND abs:"LLM")',
            '(abs:"post-training quantization" AND abs:"LLM")',
            
            # Category-based searches
            '(cat:cs.CL AND ti:"quantization")',
            '(cat:cs.LG AND ti:"quantization" AND abs:"language model")',
        ]
        
        query = " OR ".join(query_parts)
        
        logger.info(f"Built query with {len(query_parts)} parts")
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
            
            # Parse the Atom feed
            feed = feedparser.parse(response.content)
            
            papers = []
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
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
            
        # Find the end of the table header (after the separator line)
        header_end = content.find("| ---", table_start)
        if header_end == -1:
            logger.error("Could not find table header separator")
            return content
            
        # Find the end of the separator line
        separator_end = content.find("\n", header_end)
        if separator_end == -1:
            logger.error("Could not find end of separator line")
            return content
            
        # Check for duplicate papers (basic check by arXiv ID)
        existing_arxiv_ids = []
        for line in content.split('\n'):
            if 'arxiv.org/abs/' in line:
                # Extract arXiv ID from line
                import re
                match = re.search(r'arxiv\.org/abs/(\d+\.\d+)', line)
                if match:
                    existing_arxiv_ids.append(match.group(1))
        
        # Filter out papers that already exist
        new_unique_papers = []
        new_unique_summaries = []
        for paper, summary in zip(new_papers, summaries):
            if paper['id'] not in existing_arxiv_ids:
                new_unique_papers.append(paper)
                new_unique_summaries.append(summary)
            else:
                logger.info(f"Skipping duplicate paper: {paper['title']}")
        
        if not new_unique_papers:
            logger.info("All papers already exist in README")
            return content
            
        logger.info(f"Adding {len(new_unique_papers)} new unique papers")
            
        # Insert new papers at the beginning of the table (after header)
        new_entries = []
        for paper, summary in zip(new_unique_papers, new_unique_summaries):
            entry = self.format_paper_entry(paper, summary)
            new_entries.append(entry)
            
        new_content = (
            content[:separator_end + 1] + 
            '\n'.join(new_entries) + '\n' +
            content[separator_end + 1:]
        )
        
        return new_content

def main():
    """Main execution function"""
    logger.info("Starting daily arXiv paper update")
    
    # Initialize components
    fetcher = ArXivPaperFetcher()
    summarizer = PaperSummarizer()
    updater = ReadmeUpdater()
    
    # Check if we have API keys
    if not (summarizer.openai_api_key or summarizer.anthropic_api_key):
        logger.error("No LLM API keys found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        sys.exit(1)
        
    # Fetch recent papers
    logger.info("Fetching recent papers from arXiv...")
    papers = fetcher.fetch_recent_papers(days_back=1)
    
    if not papers:
        logger.info("No new papers found - exiting without changes")
        sys.exit(0)
        
    logger.info(f"Processing {len(papers)} papers...")
    
    # Process each paper separately and create individual commits
    total_added = 0
    
    for i, paper in enumerate(papers):
        logger.info(f"Processing paper {i+1}/{len(papers)}: {paper['title']}")
        
        # Summarize the paper
        summary = summarizer.summarize_paper(paper)
        
        # Load current README
        readme_content = updater.load_readme()
        if not readme_content:
            logger.error("Failed to load README.md")
            continue
            
        # Update with single paper
        updated_content = updater.update_papers_table(readme_content, [paper], [summary])
        
        # Check if content actually changed (paper wasn't duplicate)
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
            logger.info(f"Paper already exists, skipped: {paper['title']}")
            
        # Add delay to avoid rate limiting
        time.sleep(1)
        
    if total_added > 0:
        logger.info(f"Successfully processed {total_added} new papers with individual commits")
        logger.info("Git commits created - workflow should detect changes")
    else:
        logger.info("No new papers were added")
        
    logger.info("Paper update completed successfully")

if __name__ == "__main__":
    main()
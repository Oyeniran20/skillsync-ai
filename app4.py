# ==================== ENTERPRISE-READY IMPORTS ====================
import streamlit as st
import json
import re
import time
import fitz  # PyMuPDF
from groq import Groq
import http.client
from datetime import datetime
import os
import traceback
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import hashlib
from dataclasses import dataclass
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ==================== CONFIGURATION & CONSTANTS ====================
class AppConfig:
    """Centralized configuration management"""
    PAGE_TITLE = "🎯 SkillSync AI Pro"
    PAGE_ICON = "🎯"
    LAYOUT = "wide"
    
    # Model Configuration
    MODEL_NAME = "llama-3.3-70b-versatile"  # Updated to faster model
    TEMPERATURE = 0.1
    MAX_TOKENS = 4096
    
    # Validation Limits
    MIN_CV_LENGTH = 150
    MIN_JD_LENGTH = 100
    MAX_CV_SIZE_MB = 10
    MAX_FILES = 20
    
    # API Settings
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3
    RATE_LIMIT_DELAY = 1.0
    
    # UI Settings
    THEME_COLORS = {
        "primary": "#4F46E5",
        "secondary": "#10B981",
        "danger": "#EF4444",
        "warning": "#F59E0B",
        "info": "#3B82F6"
    }
    
    # Cache Settings
    SESSION_TIMEOUT_MINUTES = 60
    MAX_CACHE_SIZE = 100

# ==================== LOGGING SETUP ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('skillsync.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== DATA MODELS ====================
@dataclass
class CVData:
    """Data model for CV information"""
    filename: str
    content: str
    size: int
    upload_time: str
    hash_id: str
    
    @classmethod
    def from_upload(cls, filename: str, content: str):
        """Create CVData from uploaded file"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return cls(
            filename=filename,
            content=content[:3000],  # Token limit
            size=len(content),
            upload_time=datetime.now().isoformat(),
            hash_id=content_hash
        )

@dataclass
class AnalysisResult:
    """Data model for analysis results"""
    cv_filename: str
    overall_score: int
    skills_match: int
    experience_match: int
    education_match: int
    matching_skills: List[str]
    missing_skills: List[str]
    missing_education: List[str]
    cv_improvements: List[Dict]
    youtube_videos: List[Dict]
    analysis_time: str
    processing_time: float
    
    def to_dict(self):
        return {
            "cv_filename": self.cv_filename,
            "overall_score": self.overall_score,
            "skills_match": self.skills_match,
            "experience_match": self.experience_match,
            "education_match": self.education_match,
            "matching_skills": self.matching_skills,
            "missing_skills": self.missing_skills,
            "missing_education": self.missing_education,
            "cv_improvements": self.cv_improvements,
            "youtube_videos": self.youtube_videos,
            "analysis_time": self.analysis_time,
            "processing_time": self.processing_time
        }

# ==================== UTILITY FUNCTIONS ====================
def setup_app_directories():
    """Initialize required directories and files"""
    directories = [".streamlit", "temp", "exports", "logs"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
    
    # Create secrets template if needed
    secrets_path = ".streamlit/secrets.toml"
    if not os.path.exists(secrets_path):
        secrets_template = """# SkillSync AI Secrets Configuration
# Get your API keys from:
# Groq: https://console.groq.com/keys
# Serper: https://serper.dev/api-key

GROQ_API_KEY = ""
SERPER_API_KEY = ""
"""
        with open(secrets_path, "w") as f:
            f.write(secrets_template)
        logger.info("Created secrets.toml template")

def get_api_keys() -> Tuple[Optional[str], Optional[str]]:
    """Get API keys from multiple sources with priority"""
    sources = [
        ("Streamlit Secrets", lambda: (
            st.secrets.get("GROQ_API_KEY"),
            st.secrets.get("SERPER_API_KEY")
        )),
        ("Environment Variables", lambda: (
            os.environ.get("GROQ_API_KEY"),
            os.environ.get("SERPER_API_KEY")
        )),
        ("Session State", lambda: (
            st.session_state.get("groq_api_key"),
            st.session_state.get("serper_api_key")
        ))
    ]
    
    for source_name, getter in sources:
        groq_key, serper_key = getter()
        if groq_key and serper_key:
            logger.info(f"Loaded API keys from {source_name}")
            return groq_key, serper_key
    
    return None, None

def extract_text_from_pdf(uploaded_file) -> Optional[str]:
    """Extract and clean text from PDF with error handling"""
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text_parts = []
        
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            if page_text.strip():
                text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
        
        full_text = "\n".join(text_parts)
        
        # Enhanced text cleaning
        clean_text = re.sub(r'\s+', ' ', full_text)
        clean_text = re.sub(r'[^\w\s.,!?\-@()/:&+#%]', ' ', clean_text)
        clean_text = clean_text.strip()
        
        return clean_text if len(clean_text) >= AppConfig.MIN_CV_LENGTH else None
        
    except Exception as e:
        logger.error(f"PDF extraction error: {str(e)}")
        return None

def validate_cv_text(text: str, filename: str) -> Tuple[bool, List[str]]:
    """Validate CV content"""
    errors = []
    
    if len(text.strip()) < AppConfig.MIN_CV_LENGTH:
        errors.append(f"Too short ({len(text)} chars, minimum {AppConfig.MIN_CV_LENGTH})")
    
    cv_lower = text.lower()
    required_sections = ["experience", "education", "skill", "work", "project", "employment"]
    found_sections = [s for s in required_sections if s in cv_lower]
    
    if len(found_sections) < 2:
        errors.append(f"Missing key sections (found: {', '.join(found_sections)})")
    
    # Check for suspicious content (too many special chars)
    special_char_ratio = len(re.findall(r'[^\w\s]', text)) / len(text) if text else 0
    if special_char_ratio > 0.3:
        errors.append("Unusual character patterns detected")
    
    return len(errors) == 0, errors

# ==================== API CLIENTS ====================
class GroqClient:
    """Thread-safe Groq API client with retry logic"""
    
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.lock = threading.Lock()
        self.request_count = 0
        
    def analyze_cv_job(self, cv_text: str, job_desc: str) -> Dict:
        """Analyze CV against job description with robust prompt engineering"""
        
        prompt = f"""
        # EXPERT CV ANALYSIS FRAMEWORK
        You are a senior HR analyst with 15+ years experience. Analyze this CV against the job description.
        
        ## ANALYSIS CRITERIA:
        1. Overall Fit (0-100): Holistic match considering all factors
        2. Technical Skills Match (0-100): Required technical competencies
        3. Experience Relevance (0-100): Years and relevance of experience
        4. Education Alignment (0-100): Academic qualifications match
        
        ## CRITICAL RULES:
        - Return ONLY valid JSON
        - Scores must be integers 0-100
        - Provide actionable, specific suggestions
        - Identify actual missing skills from JD
        
        ## INPUT DATA:
        CV: {cv_text[:2500]}
        
        JOB DESCRIPTION: {job_desc[:2500]}
        
        ## REQUIRED JSON OUTPUT FORMAT:
        {{
          "overall_score": 85,
          "skills_match": 80,
          "experience_match": 90,
          "education_match": 85,
          "matching_skills": ["Python", "AWS", "Docker", "CI/CD"],
          "missing_skills": ["Kubernetes", "Terraform", "Prometheus"],
          "missing_education": ["AWS Certified Solutions Architect"],
          "cv_improvement_suggestions": [
            {{"section": "Technical Skills", "suggestion": "Add Kubernetes and Terraform to skills section", "priority": "high"}},
            {{"section": "Experience", "suggestion": "Quantify achievements with metrics (e.g., 'Improved deployment time by 40%')", "priority": "medium"}}
          ],
          "youtube_search_query": "Kubernetes tutorial for DevOps 2024"
        }}
        """
        
        for attempt in range(AppConfig.MAX_RETRIES):
            try:
                with self.lock:
                    self.request_count += 1
                    if self.request_count % 5 == 0:
                        time.sleep(AppConfig.RATE_LIMIT_DELAY)
                
                completion = self.client.chat.completions.create(
                    model=AppConfig.MODEL_NAME,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a JSON-only API. Output must be valid JSON with exact field names as specified."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=AppConfig.TEMPERATURE,
                    max_tokens=AppConfig.MAX_TOKENS,
                    response_format={"type": "json_object"},
                    timeout=AppConfig.REQUEST_TIMEOUT
                )
                
                result_text = completion.choices[0].message.content.strip()
                
                # Robust JSON extraction
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    result = json.loads(json_str)
                    
                    # Validate structure
                    required_keys = ["overall_score", "missing_skills", "cv_improvement_suggestions"]
                    if all(key in result for key in required_keys):
                        return result
                
                logger.warning(f"Invalid JSON format on attempt {attempt + 1}")
                
            except Exception as e:
                logger.error(f"Groq API attempt {attempt + 1} failed: {str(e)}")
                if attempt == AppConfig.MAX_RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
        
        raise Exception("All retry attempts failed")

class SerperClient:
    """YouTube search client with caching"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.cache = {}
    
    def search_youtube_videos(self, query: str, max_results: int = 3) -> List[Dict]:
        """Search YouTube videos with caching"""
        cache_key = hashlib.md5(query.encode()).hexdigest()
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            conn = http.client.HTTPSConnection('google.serper.dev')
            payload = json.dumps({
                "q": query,
                "num": max_results,
                "gl": "us",
                "hl": "en",
                "type": "video"
            })
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json"
            }
            
            conn.request("POST", "/search", payload, headers)
            response = conn.getresponse()
            
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                videos = []
                
                for item in data.get("organic", [])[:max_results]:
                    if "youtube.com" in item.get("link", ""):
                        videos.append({
                            "title": item.get("title", "No title"),
                            "link": item.get("link", ""),
                            "snippet": item.get("snippet", "")[:200],
                            "channel": item.get("source", "").replace(" - YouTube", "")
                        })
                
                self.cache[cache_key] = videos
                return videos
                
        except Exception as e:
            logger.error(f"Serper API error: {str(e)}")
        
        return []

# ==================== UI COMPONENTS ====================
def create_sidebar() -> Tuple[Optional[str], Optional[str]]:
    """Create professional sidebar with API configuration"""
    with st.sidebar:
        # Logo and Title
        st.image("https://img.icons8.com/color/96/000000/resume.png", width=80)
        st.title("SkillSync AI Pro")
        st.caption("Enterprise CV Analysis Platform")
        
        st.markdown("---")
        
        # API Configuration Section
        st.subheader("🔧 API Configuration")
        
        groq_key, serper_key = get_api_keys()
        
        # Manual input if not found
        if not groq_key or not serper_key:
            with st.expander("Enter API Keys", expanded=True):
                groq_key = st.text_input(
                    "Groq API Key",
                    type="password",
                    help="Get from https://console.groq.com",
                    placeholder="gsk_...",
                    key="groq_api_input"
                )
                serper_key = st.text_input(
                    "Serper API Key",
                    type="password",
                    help="Get from https://serper.dev",
                    placeholder="Your Serper key",
                    key="serper_api_input"
                )
                
                if groq_key and serper_key:
                    st.session_state.groq_api_key = groq_key
                    st.session_state.serper_api_key = serper_key
                    st.success("✅ Keys saved to session")
        
        # API Status
        st.markdown("---")
        st.subheader("📊 API Status")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Groq", "✅ Ready" if groq_key else "❌ Missing", 
                     delta="Connected" if groq_key else "Required")
        with col2:
            st.metric("Serper", "✅ Ready" if serper_key else "❌ Missing",
                     delta="Connected" if serper_key else "Required")
        
        # Quick Actions
        st.markdown("---")
        st.subheader("⚡ Quick Actions")
        
        if st.button("🔄 Clear Cache", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        if st.button("📋 Copy Session ID", use_container_width=True):
            session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
            st.code(session_id)
        
        # System Info
        st.markdown("---")
        st.caption(f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.caption(f"📁 {len(st.session_state.get('cv_data', []))} CVs loaded")
        
        return groq_key, serper_key

def create_job_description_section() -> str:
    """Create job description input with templates"""
    st.header("📋 1. Job Description Analysis")
    
    # Template selector
    templates = {
        "Custom": "",
        "DevOps Engineer": """DevOps Engineer
Responsibilities:
- Design, implement, and maintain CI/CD pipelines
- Manage cloud infrastructure (AWS/Azure/GCP)
- Implement infrastructure as code (Terraform/CloudFormation)
- Monitor system performance and ensure reliability
- Automate deployment and scaling processes

Requirements:
- 3+ years DevOps experience
- Proficiency in AWS, Docker, Kubernetes
- Experience with Terraform or similar IaC tools
- Strong scripting skills (Python/Bash)
- CI/CD tools experience (Jenkins, GitLab CI, GitHub Actions)
- Monitoring tools (Prometheus, Grafana)""",
        
        "Data Scientist": """Data Scientist
Responsibilities:
- Develop machine learning models
- Analyze large datasets
- Create data visualizations
- Collaborate with engineering teams
- Present findings to stakeholders

Requirements:
- Master's in Computer Science or related field
- Python, SQL, R proficiency
- ML frameworks (TensorFlow, PyTorch)
- Statistical analysis experience
- Data visualization skills""",
        
        "Full Stack Developer": """Full Stack Developer
Responsibilities:
- Develop frontend and backend features
- Design and implement APIs
- Optimize application performance
- Collaborate with UX/UI designers
- Write clean, maintainable code

Requirements:
- 3+ years full-stack development
- React/Vue.js/Angular experience
- Node.js/Python/Django experience
- Database design (SQL/NoSQL)
- REST API design
- Git version control"""
    }
    
    selected_template = st.selectbox(
        "📄 Choose Template (Optional)",
        list(templates.keys()),
        help="Start with a template or paste your own"
    )
    
    # Text area with template
    default_text = templates[selected_template] if selected_template != "Custom" else ""
    
    job_desc = st.text_area(
        "📝 Paste Job Description",
        height=250,
        value=default_text,
        placeholder="Paste the complete job description here...",
        help="Include requirements, responsibilities, qualifications",
        key="job_description_input"
    )
    
    # Word count
    if job_desc:
        word_count = len(job_desc.split())
        st.caption(f"📊 {word_count} words | {len(job_desc)} characters")
    
    return job_desc

def create_cv_upload_section() -> List[CVData]:
    """Create enhanced CV upload section"""
    st.header("📄 2. Upload CVs (PDF)")
    
    # Upload area with drag-and-drop style
    uploaded_files = st.file_uploader(
        "📤 Drag & drop or click to upload CVs",
        type=['pdf'],
        accept_multiple_files=True,
        help=f"Max {AppConfig.MAX_FILES} files | Max {AppConfig.MAX_CV_SIZE_MB}MB each",
        key="cv_upload_area"
    )
    
    # Initialize session state
    if 'cv_data' not in st.session_state:
        st.session_state.cv_data = []
    
    # Process new uploads
    if uploaded_files and len(uploaded_files) > 0:
        # Limit number of files
        if len(uploaded_files) > AppConfig.MAX_FILES:
            st.warning(f"⚠️ Maximum {AppConfig.MAX_FILES} files allowed. Showing first {AppConfig.MAX_FILES}.")
            uploaded_files = uploaded_files[:AppConfig.MAX_FILES]
        
        # Process each file
        success_count = 0
        error_count = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            
            try:
                # Check file size
                file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
                if file_size > AppConfig.MAX_CV_SIZE_MB:
                    st.warning(f"⚠️ {uploaded_file.name}: File too large ({file_size:.1f}MB)")
                    error_count += 1
                    continue
                
                # Extract text
                cv_text = extract_text_from_pdf(uploaded_file)
                
                if cv_text:
                    # Validate
                    is_valid, errors = validate_cv_text(cv_text, uploaded_file.name)
                    
                    if is_valid:
                        # Create CVData object
                        cv_data = CVData.from_upload(uploaded_file.name, cv_text)
                        
                        # Check for duplicates
                        existing_hashes = [cv.hash_id for cv in st.session_state.cv_data]
                        if cv_data.hash_id not in existing_hashes:
                            st.session_state.cv_data.append(cv_data)
                            success_count += 1
                        else:
                            st.info(f"📄 {uploaded_file.name}: Already loaded")
                    else:
                        st.warning(f"⚠️ {uploaded_file.name}: {', '.join(errors)}")
                        error_count += 1
                else:
                    st.error(f"❌ {uploaded_file.name}: Could not extract text")
                    error_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing {uploaded_file.name}: {str(e)}")
                st.error(f"❌ {uploaded_file.name}: Processing error")
                error_count += 1
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        status_text.empty()
        progress_bar.empty()
        
        if success_count > 0:
            st.success(f"✅ Successfully loaded {success_count} CV(s)")
        if error_count > 0:
            st.error(f"❌ Failed to load {error_count} CV(s)")
    
    # Display loaded CVs
    if st.session_state.cv_data:
        st.subheader("📋 Loaded CVs")
        
        # Create a nice grid view
        cols = st.columns(3)
        for idx, cv in enumerate(st.session_state.cv_data):
            with cols[idx % 3]:
                with st.container(border=True):
                    st.markdown(f"**{cv.filename}**")
                    st.caption(f"📏 {cv.size:,} chars | 🆔 {cv.hash_id}")
                    st.caption(f"⏰ {datetime.fromisoformat(cv.upload_time).strftime('%H:%M:%S')}")
                    
                    # Quick preview
                    with st.expander("Preview"):
                        st.text(cv.content[:200] + "..." if len(cv.content) > 200 else cv.content)
                    
                    # Remove button
                    if st.button("🗑️ Remove", key=f"remove_{cv.hash_id}"):
                        st.session_state.cv_data = [c for c in st.session_state.cv_data if c.hash_id != cv.hash_id]
                        st.rerun()
        
        # Clear all button
        if st.button("🗑️ Clear All CVs", type="secondary", use_container_width=True):
            st.session_state.cv_data = []
            st.rerun()
    
    return st.session_state.cv_data

def create_analysis_section(groq_key: str, serper_key: str, job_desc: str, cv_data: List[CVData]):
    """Create analysis section with parallel processing"""
    st.header("🤖 3. Run Analysis")
    
    if not job_desc or len(cv_data) == 0:
        st.warning("⚠️ Please provide job description and upload CVs first")
        return
    
    # Analysis configuration
    with st.expander("⚙️ Analysis Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            enable_youtube = st.checkbox("YouTube Recommendations", value=True)
            enable_parallel = st.checkbox("Parallel Processing", value=True)
        with col2:
            analysis_depth = st.select_slider(
                "Analysis Depth",
                options=["Basic", "Standard", "Detailed"],
                value="Standard"
            )
    
    # Start analysis button
    if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
        with st.spinner("🚀 Initializing analysis..."):
            # Clear previous results
            if 'analysis_results' in st.session_state:
                del st.session_state.analysis_results
            
            # Initialize clients
            groq_client = GroqClient(groq_key)
            serper_client = SerperClient(serper_key) if enable_youtube else None
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_container = st.empty()
            
            all_results = {}
            start_time = time.time()
            
            def process_single_cv(cv: CVData, idx: int):
                """Process single CV with progress updates"""
                try:
                    # Update status
                    status_text.text(f"📊 Analyzing {cv.filename}... ({idx + 1}/{len(cv_data)})")
                    
                    # Call Groq API
                    analysis_start = time.time()
                    result = groq_client.analyze_cv_job(cv.content, job_desc)
                    
                    # Get YouTube videos if enabled
                    youtube_videos = []
                    if serper_client and result.get("youtube_search_query"):
                        search_query = result.get("youtube_search_query", 
                                                 f"{result['missing_skills'][0]} tutorial 2024" 
                                                 if result.get('missing_skills') else "")
                        youtube_videos = serper_client.search_youtube_videos(search_query)
                    
                    # Create AnalysisResult object
                    analysis_result = AnalysisResult(
                        cv_filename=cv.filename,
                        overall_score=result.get('overall_score', 0),
                        skills_match=result.get('skills_match', 0),
                        experience_match=result.get('experience_match', 0),
                        education_match=result.get('education_match', 0),
                        matching_skills=result.get('matching_skills', []),
                        missing_skills=result.get('missing_skills', []),
                        missing_education=result.get('missing_education', []),
                        cv_improvements=result.get('cv_improvement_suggestions', []),
                        youtube_videos=youtube_videos,
                        analysis_time=datetime.now().isoformat(),
                        processing_time=time.time() - analysis_start
                    )
                    
                    return cv.filename, analysis_result
                    
                except Exception as e:
                    logger.error(f"Analysis failed for {cv.filename}: {str(e)}")
                    return cv.filename, {
                        "error": str(e),
                        "cv_filename": cv.filename,
                        "analysis_time": datetime.now().isoformat()
                    }
            
            # Process CVs
            if enable_parallel and len(cv_data) > 1:
                # Parallel processing
                with ThreadPoolExecutor(max_workers=min(4, len(cv_data))) as executor:
                    futures = {
                        executor.submit(process_single_cv, cv, idx): (cv, idx) 
                        for idx, cv in enumerate(cv_data)
                    }
                    
                    completed = 0
                    for future in as_completed(futures):
                        cv_name, result = future.result()
                        all_results[cv_name] = result
                        completed += 1
                        progress_bar.progress(completed / len(cv_data))
            else:
                # Sequential processing
                for idx, cv in enumerate(cv_data):
                    cv_name, result = process_single_cv(cv, idx)
                    all_results[cv_name] = result
                    progress_bar.progress((idx + 1) / len(cv_data))
            
            # Calculate total time
            total_time = time.time() - start_time
            
            # Store results
            st.session_state.analysis_results = all_results
            st.session_state.analysis_metadata = {
                "total_cvs": len(cv_data),
                "processing_time": total_time,
                "timestamp": datetime.now().isoformat(),
                "job_description_preview": job_desc[:200]
            }
            
            status_text.text(f"✅ Analysis complete in {total_time:.1f} seconds")
            progress_bar.progress(1.0)
            time.sleep(1)
            
            st.rerun()

def create_results_display():
    """Create beautiful results display with visualizations"""
    if not hasattr(st.session_state, 'analysis_results'):
        return
    
    st.header("📊 4. Analysis Results")
    
    # Summary statistics
    successful_results = {
        k: v for k, v in st.session_state.analysis_results.items() 
        if not isinstance(v, dict) or 'error' not in v
    }
    
    if not successful_results:
        st.error("No successful analyses to display")
        return
    
    # Overall statistics
    st.subheader("📈 Overall Performance")
    
    scores = [r.overall_score if hasattr(r, 'overall_score') else 0 
              for r in successful_results.values()]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average Score", f"{np.mean(scores):.1f}", 
                 delta=f"{len(successful_results)} CVs")
    with col2:
        st.metric("Highest Score", f"{max(scores):.0f}")
    with col3:
        st.metric("Lowest Score", f"{min(scores):.0f}")
    with col4:
        st.metric("Processing Time", 
                 f"{st.session_state.analysis_metadata.get('processing_time', 0):.1f}s")
    
    # Create tabs for each CV
    tab_names = list(successful_results.keys())
    tabs = st.tabs([f"📄 {name}" for name in tab_names])
    
    for idx, (cv_name, result) in enumerate(successful_results.items()):
        with tabs[idx]:
            # Header with score
            score_color = "green" if result.overall_score >= 70 else "orange" if result.overall_score >= 50 else "red"
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <h2 style="color: white; margin: 0;">{cv_name}</h2>
                <h1 style="color: white; margin: 0; font-size: 3em;">{result.overall_score}/100</h1>
                <p style="color: rgba(255,255,255,0.8); margin: 0;">Overall Match Score</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Score breakdown in columns
            st.subheader("📊 Score Breakdown")
            col1, col2, col3, col4 = st.columns(4)
            
            metrics = [
                ("Skills Match", result.skills_match, "🔧"),
                ("Experience Match", result.experience_match, "💼"),
                ("Education Match", result.education_match, "🎓"),
                ("Processing Time", f"{result.processing_time:.1f}s", "⚡")
            ]
            
            for (title, value, icon), col in zip(metrics, [col1, col2, col3, col4]):
                with col:
                    if isinstance(value, (int, float)):
                        st.metric(f"{icon} {title}", value)
                        st.progress(value / 100)
                    else:
                        st.metric(f"{icon} {title}", value)
            
            # Skills Analysis
            st.subheader("🔍 Skills Analysis")
            col_left, col_right = st.columns(2)
            
            with col_left:
                # Matching Skills
                if result.matching_skills:
                    st.markdown("### ✅ Matching Skills")
                    for skill in result.matching_skills[:15]:
                        st.success(f"• {skill}")
                
                # CV Improvements
                if result.cv_improvements:
                    st.markdown("### 🛠️ CV Improvement Suggestions")
                    for improvement in result.cv_improvements[:5]:
                        priority = improvement.get('priority', 'medium')
                        priority_color = {
                            'high': 'danger',
                            'medium': 'warning',
                            'low': 'info'
                        }.get(priority, 'info')
                        
                        st.markdown(f"""
                        <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; margin: 5px 0; 
                                    border-left: 4px solid {AppConfig.THEME_COLORS.get(priority_color, '#3B82F6')}">
                            <strong>{improvement.get('section', 'General')}</strong><br>
                            {improvement.get('suggestion', '')}
                        </div>
                        """, unsafe_allow_html=True)
            
            with col_right:
                # Missing Skills
                if result.missing_skills:
                    st.markdown("### ❌ Missing Skills")
                    for skill in result.missing_skills[:15]:
                        st.error(f"• {skill}")
                
                # Education Gaps
                if result.missing_education:
                    st.markdown("### 🎓 Education Gaps")
                    for edu in result.missing_education[:5]:
                        st.warning(f"• {edu}")
            
            # YouTube Recommendations
            if result.youtube_videos:
                st.subheader("🎬 Recommended Learning Resources")
                
                for video in result.youtube_videos:
                    with st.container(border=True):
                        col_v1, col_v2 = st.columns([3, 1])
                        with col_v1:
                            st.markdown(f"**{video.get('title', 'No title')}**")
                            if video.get('channel'):
                                st.caption(f"🎥 {video['channel']}")
                            if video.get('snippet'):
                                st.caption(video['snippet'])
                        with col_v2:
                            if video.get('link'):
                                st.link_button("▶️ Watch", video['link'])
    
    # Comparison Dashboard
    st.header("📋 Comparison Dashboard")
    
    # Create comparison DataFrame
    comparison_data = []
    for cv_name, result in successful_results.items():
        comparison_data.append({
            "CV": cv_name,
            "Overall": result.overall_score,
            "Skills": result.skills_match,
            "Experience": result.experience_match,
            "Education": result.education_match,
            "Matching Skills": len(result.matching_skills),
            "Missing Skills": len(result.missing_skills),
            "Improvements": len(result.cv_improvements)
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Display comparison table
    st.dataframe(
        df.sort_values("Overall", ascending=False),
        use_container_width=True,
        hide_index=True
    )
    
    # Visualizations
    st.subheader("📊 Visual Analysis")
    
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        # Radar chart for top 3 CVs
        top_3 = df.nlargest(3, "Overall")
        if len(top_3) >= 2:
            fig = go.Figure()
            
            categories = ['Skills', 'Experience', 'Education', 'Matching Skills', 'Missing Skills']
            
            for idx, row in top_3.iterrows():
                values = [row['Skills'], row['Experience'], row['Education'], 
                         row['Matching Skills'] * 10, (100 - row['Missing Skills'] * 5)]
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=row['CV'][:20]
                ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                title="Top 3 CVs Comparison"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col_viz2:
        # Bar chart for scores
        fig2 = px.bar(
            df,
            x='CV',
            y=['Overall', 'Skills', 'Experience', 'Education'],
            title="Score Breakdown by CV",
            barmode='group'
        )
        fig2.update_layout(
            xaxis_title="CV",
            yaxis_title="Score",
            legend_title="Metric"
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Export options
    st.subheader("📥 Export Results")
    
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    
    with col_exp1:
        # JSON export
        export_data = {
            "metadata": st.session_state.analysis_metadata,
            "results": {
                k: v.to_dict() if hasattr(v, 'to_dict') else v 
                for k, v in st.session_state.analysis_results.items()
            }
        }
        
        json_str = json.dumps(export_data, indent=2)
        st.download_button(
            label="📊 Download JSON",
            data=json_str,
            file_name=f"skillsync_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col_exp2:
        # CSV export
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📈 Download CSV",
            data=csv_data,
            file_name=f"skillsync_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_exp3:
        # Report generation
        if st.button("📋 Generate PDF Report", use_container_width=True):
            with st.spinner("Generating report..."):
                # Placeholder for PDF generation
                st.info("PDF report generation feature coming soon!")
                # In production, integrate with ReportLab or WeasyPrint

# ==================== MAIN APP ====================
def main():
    """Main application entry point"""
    
    # Setup
    setup_app_directories()
    
    # Page configuration
    st.set_page_config(
        page_title=AppConfig.PAGE_TITLE,
        page_icon=AppConfig.PAGE_ICON,
        layout=AppConfig.LAYOUT,
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4F46E5, #10B981);
    }
    .css-1d391kg {padding: 1rem;}
    </style>
    """, unsafe_allow_html=True)
    
    # App Header
    st.markdown('<h1 class="main-header">🎯 SkillSync AI Pro</h1>', unsafe_allow_html=True)
    st.markdown("""
    **Enterprise-grade CV analysis platform** | Upload multiple CVs and compare against job descriptions with AI-powered insights.
    """)
    
    # Create sidebar and get API keys
    groq_key, serper_key = create_sidebar()
    
    # Main workflow
    job_desc = create_job_description_section()
    cv_data = create_cv_upload_section()
    
    if groq_key and serper_key:
        create_analysis_section(groq_key, serper_key, job_desc, cv_data)
    
    # Display results if available
    if hasattr(st.session_state, 'analysis_results'):
        create_results_display()
    
    # Footer
    st.markdown("---")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f2:
        st.caption(f"""
        🚀 **SkillSync AI Pro v2.0**  
        📅 {datetime.now().strftime('%Y-%m-%d')}  
        ⚡ Enterprise CV Analysis Platform
        """)

# ==================== APPLICATION START ====================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        st.error(f"🚨 Application error: {str(e)}")
        with st.expander("Technical Details"):
            st.code(traceback.format_exc())
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
# from youtube_transcript_api import YouTubeTranscriptApi, YouTubeRequestFailed, TranscriptsDisabled, NoTranscriptFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import YoutubeLoader
import urllib.parse
import streamlit as st
import os
import time
from datetime import datetime
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize the model
@st.cache_resource
def load_model():
    return ChatGoogleGenerativeAI(
        model='gemini-2.5-flash',
        temperature=0
    )

model = load_model()
parser = StrOutputParser()

def extract_video_id(url):
    """Extract video ID from various YouTube URL formats"""
    try:
        parsed_url = urllib.parse.urlparse(url)
        if parsed_url.netloc == 'youtu.be':
            return parsed_url.path.strip('/')
        elif parsed_url.netloc in ['www.youtube.com', 'youtube.com']:
            query_params = urllib.parse.parse_qs(parsed_url.query)
            return query_params.get('v', [None])[0]
        return None
    except Exception as e:
        logger.error(f"Error extracting video ID: {e}")
        return None

def get_transcript_with_retry(video_id, max_retries=3):
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    for attempt in range(max_retries):
        try:
            loader = YoutubeLoader.from_youtube_url(youtube_url, language=['en', 'hi'])
            transcript_text = " ".join([doc.page_content for doc in loader.load()])
            return transcript_text, None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None, str(e)
    return None, "Failed after retries"

def get_video_title(video_id):
    """Get video title using YouTube oEmbed API"""
    try:
        response = requests.get(f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json")
        if response.status_code == 200:
            return response.json().get('title', 'Unknown Title')
    except Exception as e:
        logger.error(f"Error fetching video title: {e}")
    return "Unknown Title"

def summarize_transcript(transcript, query="Summarize the key points"):
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=300)
        chunks = splitter.split_text(transcript)

        progress_bar = st.progress(0)
        status_text = st.empty()
        summary_prompt = PromptTemplate(
            input_variables=["chunk"],
            template="""
    You are a helpful assistant. Summarize the following part of a YouTube video transcript clearly:

    Transcript segment:
    {chunk}
    """
        )

        summarize_chain = summary_prompt | model | parser
        partial_summaries = []

        for i, chunk in enumerate(chunks):
            summary = summarize_chain.invoke({'chunk': chunk})
            partial_summaries.append(summary)
            progress_bar.progress((i + 1) / len(chunks))

        combined_summary = " ".join(partial_summaries)
        final_prompt = PromptTemplate(
            input_variables=["combined_summary", "user_query"],
            template="""
    You are a highly intelligent assistant that understands YouTube videos and provides accurate, well-structured responses based on their content.

    🎬 Combined Transcript Summary:
    {combined_summary}

    💬 User Query:
    {user_query}

    🧠 Instructions:
    - Use the above *combined summary* (already condensed from the full transcript[0]) to answer the user's query.
    - Your goal is to provide a **final summary** that:
    - Gives a concise overview of the entire video.
    - Directly answers the user's question.
    - Maintains logical flow and coherence.
    - Avoids repetition or unnecessary details.
    - If the query cannot be answered from the content, say:
    **"Sorry, the video doesn't contain information about that."**
    - Write in a natural, engaging tone.

    📝 Final Answer:
    """
        )

        chain = final_prompt | model | parser
        final_result = chain.invoke({'combined_summary': combined_summary, 'user_query': query})

        return final_result, len(chunks), len(transcript.split())

        
    except Exception as e:
        logger.error(f"Error in summarize_transcript: {e}")
        return f"Error during summarization: {str(e)}", 0, 0

def main():
    st.set_page_config(
        page_title="YouTube Video Summarizer",
        page_icon="https://upload.wikimedia.org/wikipedia/commons/4/42/YouTube_icon_%282013-2017%29.png",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .video-info {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .summary-box {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border-left: 4px solid #FF4B4B;
        margin: 1rem 0;
    }
    .feature-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }
    .error-box {
        background-color: #ffe6e6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #ff4b4b;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
<h1 class="main-header">
    <img src="https://upload.wikimedia.org/wikipedia/commons/4/42/YouTube_icon_%282013-2017%29.png" 
         alt="YouTube" width="50" style="vertical-align:middle;"> 
    YouTube Video Summarizer
</h1>
""", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API Configuration
        st.subheader("Model Configuration")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
        
        # Features
        st.subheader("✨ Features")
        enable_qa = st.checkbox("Enable Q&A Mode", value=True)
        enable_stats = st.checkbox("Show Statistics", value=True)
        save_history = st.checkbox("Save to History", value=True)
        
        # Retry settings
        st.subheader("🔧 Advanced")
        max_retries = st.slider("Max Retry Attempts", 1, 5, 3)
        
        # History
        if 'history' not in st.session_state:
            st.session_state.history = []
        
        if st.session_state.history:
            st.subheader("📚 History")
            for i, item in enumerate(st.session_state.history[-5:]):
                with st.expander(f"Video {i+1}: {item['timestamp']}"):
                    st.write(f"**Title:** {item.get('title', 'N/A')}")
                    st.write(f"**URL:** {item['url']}")
                    st.write(f"**Query:** {item['query']}")
                    if st.button(f"Load #{i+1}", key=f"load_{i}"):
                        st.session_state.current_url = item['url']
                        st.session_state.current_query = item['query']
                        st.rerun()
        
        # Clear history
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("🔗 Enter YouTube URL")
        
        # URL input with session state
        if 'current_url' not in st.session_state:
            st.session_state.current_url = ""
        
        youtube_url = st.text_input(
            "YouTube Video URL",
            value=st.session_state.current_url,
            placeholder="https://www.youtube.com/watch?v=... or https://youtu.be/..."
        )
        
        # Query input
        if 'current_query' not in st.session_state:
            st.session_state.current_query = "Summarize the key points"
        
        user_query = st.text_area(
            "What would you like to know about this video?",
            value=st.session_state.current_query,
            placeholder="e.g., Summarize the key points, What are the main arguments?, Explain the conclusion..."
        )
        
        # Process button
        process_clicked = st.button("🚀 Process Video", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Video info display
        if youtube_url:
            video_id = extract_video_id(youtube_url)
            if video_id:
                st.markdown('<div class="video-info">', unsafe_allow_html=True)
                st.subheader("📹 Video Information")
                st.write(f"**Video ID:** `{video_id}`")
                
                # Get and display video title
                video_title = get_video_title(video_id)
                st.write(f"**Title:** {video_title}")
                
                # Embed video preview
                try:
                    st.video(youtube_url)
                except Exception as e:
                    st.write("Video preview unavailable")
                    logger.error(f"Video preview error: {e}")
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("❌ Could not extract Video ID from the URL")
    
    with col2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("📊 Quick Stats")
        
        if 'last_processing_stats' in st.session_state:
            stats = st.session_state.last_processing_stats
            metric1, metric2, metric3 = st.columns(3)
            
            with metric1:
                st.metric("Transcript Length", f"{stats['word_count']:,} words")
            with metric2:
                st.metric("Chunks Processed", stats['chunk_count'])
            with metric3:
                st.metric("Processing Time", f"{stats['processing_time']:.1f}s")
        else:
            st.info("Process a video to see statistics here")
        
        st.subheader("💡 Example Queries")
        
        example_queries = [
            "Summarize the main points in bullet points",
            "What are the key takeaways?",
            "Explain the main arguments presented",
            "What problem does this video solve and how?",
            "List the step-by-step process described"
        ]
        
        for i, example in enumerate(example_queries):
            if st.button(example, key=f"example_{i}", use_container_width=True):
                st.session_state.current_query = example
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Processing and results
    if process_clicked and youtube_url:
        video_id = extract_video_id(youtube_url)
        
        if not video_id:
            st.error("❌ Invalid YouTube URL. Please check the URL and try again.")
            return
        
        # Fetch transcript[0] with retries
        with st.spinner("🔄 Fetching transcript..."):
            transcript, error = get_transcript_with_retry(video_id, max_retries)
        
        if error:
            st.markdown('<div class="error-box">', unsafe_allow_html=True)
            st.error(f"❌ {error}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show troubleshooting tips
            with st.expander("🔧 Troubleshooting Tips"):
                st.markdown("""
                **Common solutions:**
                - Check if the video has captions enabled
                - Try a different YouTube video
                - Check your internet connection
                - Wait a few minutes and try again (YouTube API might be temporarily unavailable)
                - Try reducing the number of retry attempts
                """)
            return
        
        if transcript:
            st.success(f"✅ Successfully fetched transcript ({len(transcript.split())} words)")
            
            # Show raw transcript[0] in expander
            with st.expander("📄 View Raw Transcript"):
                st.text_area("Transcript", transcript, height=200, key=f"raw_transcript_{video_id}")
            
            # Process transcript[0]
            start_time = time.time()
            
            with st.spinner("🤖 Analyzing transcript with AI..."):
                summary, chunk_count, word_count = summarize_transcript(transcript, user_query)
            
            processing_time = time.time() - start_time
            
            # Store stats
            st.session_state.last_processing_stats = {
                'chunk_count': chunk_count,
                'word_count': word_count,
                'processing_time': processing_time
            }
            
            # Display results
            st.markdown("---")
            st.markdown('<div class="summary-box">', unsafe_allow_html=True)
            st.subheader("📝 Summary & Analysis")
            st.write(summary)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Save to history
            if save_history:
                video_title = get_video_title(video_id)
                history_item = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'url': youtube_url,
                    'title': video_title,
                    'query': user_query,
                    'summary': summary,
                    'video_id': video_id
                }
                st.session_state.history.append(history_item)
                st.sidebar.success(f"✅ Saved to history")
            
            # Q&A Section
            if enable_qa and transcript:
                st.markdown("---")
                st.subheader("❓ Ask Follow-up Questions")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    new_question = st.text_input("Ask another question about this video:", placeholder="e.g., Can you explain more about...", key="followup_question")
                with col2:
                    ask_btn = st.button("Ask Question", use_container_width=True, key="ask_followup")
                
                if ask_btn and new_question:
                    with st.spinner("🤖 Thinking..."):
                        qa_result, _, _ = summarize_transcript(transcript, new_question)
                    
                    st.markdown(f"**Q:** {new_question}")
                    st.markdown(f"**A:** {qa_result}")
    
    # Instructions
    with st.expander("ℹ️ How to use this tool"):
        st.markdown("""
        ### 🎯 Step-by-Step Guide
        
        1. **Enter YouTube URL**: Paste any YouTube video link in the input field
        2. **Ask Your Question**: Specify what you want to know about the video
        3. **Click Process**: The tool will fetch the transcript[0] and generate insights
        4. **Explore Results**: View summary, ask follow-up questions, and see statistics
        
        ### 🔍 Supported URL Formats
        - `https://www.youtube.com/watch?v=VIDEO_ID`
        - `https://youtu.be/VIDEO_ID`
        - `https://youtube.com/watch?v=VIDEO_ID`
        
        ### 💡 Pro Tips
        - Use specific questions for better answers
        - Enable Q&A mode for interactive conversations
        - Check history to revisit previous videos
        - Use example queries for inspiration
        
        ### ⚠️ Limitations
        - Requires English or Hindi captions
        - Works best with informational/educational content
        - Processing time depends on video length
        """)

if __name__ == "__main__":
    main() 

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import time
import gc
import logging
from datetime import datetime
import json

# Configure page
st.set_page_config(
    page_title="TinyLlama Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamlitChatbot:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_name = model_name
        self.pipe = None
        self.device = None
        self.dtype = None
        
    def load_model(self):
        """Load TinyLlama model optimized for Streamlit"""
        try:
            # Detect hardware capabilities
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            logger.info(f"Loading model on {self.device} with dtype {self.dtype}")
            
            # Load with memory optimization
            self.pipe = pipeline(
                "text-generation",
                model=self.model_name,
                dtype=self.dtype,
                device_map="auto" if self.device == "cuda" else None,
                model_kwargs={"low_cpu_mem_usage": True}
            )
            
            logger.info("Model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return False
    
    def generate_response(self, user_input, conversation_history):
        """Generate response with conversation context"""
        if not self.pipe:
            return "❌ Model not loaded. Please refresh the page and try again."
        
        try:
            # Build conversation context
            messages = [
                {
                    "role": "system", 
                    "content": "You are a helpful AI assistant. Provide clear, concise, and practical responses. Be friendly but professional."
                }
            ]
            
            # Add recent conversation history (last 8 messages to save memory)
            recent_history = conversation_history[-8:] if len(conversation_history) > 8 else conversation_history
            messages.extend(recent_history)
            
            # Add current user input
            messages.append({"role": "user", "content": user_input})
            
            # Format for TinyLlama
            prompt = self.pipe.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Generate response with memory management
            with torch.no_grad():
                outputs = self.pipe(
                    prompt,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                    pad_token_id=self.pipe.tokenizer.eos_token_id,
                    return_full_text=False
                )
            
            response = outputs[0]["generated_text"].strip()
            
            # Clean up response
            if not response:
                response = "I'm not sure how to respond to that. Could you try asking differently?"
            
            return response
            
        except torch.cuda.OutOfMemoryError:
            return "🚨 GPU memory full! Try asking a shorter question or clear the chat history."
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"⚠️ Error generating response: {str(e)[:100]}..."
    
    def clear_gpu_memory(self):
        """Clear GPU memory and cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

# Initialize chatbot in session state
@st.cache_resource
def load_chatbot_model():
    """Load and cache the chatbot model"""
    chatbot = StreamlitChatbot()
    
    with st.spinner("🔄 Loading TinyLlama model (this may take 2-3 minutes first time)..."):
        success = chatbot.load_model()
    
    if success:
        st.success("✅ Model loaded successfully!")
        return chatbot
    else:
        st.error("❌ Failed to load model")
        return None

# Initialize session state
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "conversation_count" not in st.session_state:
        st.session_state.conversation_count = 0
    
    if "response_times" not in st.session_state:
        st.session_state.response_times = []
    
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0

# Main app
def main():
    # Initialize session state
    initialize_session_state()
    
    # Load chatbot
    chatbot = load_chatbot_model()
    
    # Header
    st.title("🤖 TinyLlama AI Assistant")
    st.markdown("*Your personal AI assistant powered by TinyLlama - running locally!*")
    
    if not chatbot:
        st.error("Cannot start chat - model failed to load. Please refresh the page.")
        return
    
    # Status indicator
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success(f"🖥️ Device: {chatbot.device.upper()}")
    with col2:
        st.info(f"🧠 Model: TinyLlama-1.1B")
    with col3:
        gpu_memory = ""
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1e9
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            gpu_memory = f"💾 GPU: {memory_used:.1f}GB/{memory_total:.1f}GB"
            st.metric("GPU Memory", f"{memory_used:.1f}GB")
        else:
            st.metric("Mode", "CPU")
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # Session Statistics
        with st.expander("📊 Session Stats", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Messages", len(st.session_state.messages))
            with col2:
                user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
                st.metric("Questions", user_messages)
            
            if st.session_state.response_times:
                avg_time = sum(st.session_state.response_times) / len(st.session_state.response_times)
                st.metric("Avg Response Time", f"{avg_time:.1f}s")
        
        # Controls
        st.markdown("---")
        
        # Clear chat
        if st.button("🔄 Clear Chat", type="secondary", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_count = 0
            st.session_state.response_times = []
            chatbot.clear_gpu_memory()
            st.success("Chat cleared!")
            st.rerun()
        
        # Clear GPU memory
        if st.button("🧹 Clear GPU Memory", type="secondary", use_container_width=True):
            chatbot.clear_gpu_memory()
            st.success("GPU memory cleared!")
        
        # Download conversation
        if st.session_state.messages:
            conversation_text = generate_conversation_export()
            st.download_button(
                label="💾 Download Chat",
                data=conversation_text,
                file_name=f"tinyllama_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        # Settings
        st.markdown("---")
        st.header("⚙️ Settings")
        
        # Temperature control
        temperature = st.slider(
            "🌡️ Response Creativity", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.7, 
            step=0.1,
            help="Lower = more focused, Higher = more creative"
        )
        
        # Max tokens
        max_tokens = st.slider(
            "📝 Response Length", 
            min_value=50, 
            max_value=500, 
            value=200, 
            step=50,
            help="Maximum number of tokens in response"
        )
        
        # System prompt
        system_prompt = st.text_area(
            "🎭 System Prompt",
            value="You are a helpful AI assistant. Provide clear, concise, and practical responses. Be friendly but professional.",
            height=100,
            help="Customize the AI's personality and behavior"
        )
        
        # Quick prompts
        st.markdown("---")
        st.header("💡 Quick Prompts")
        
        quick_prompts = [
            "Explain quantum computing simply",
            "Write a Python function to sort a list",
            "Give me 5 business ideas for 2025",
            "How do I improve my resume?",
            "Explain machine learning basics"
        ]
        
        for prompt in quick_prompts:
            if st.button(f"💬 {prompt}", key=f"quick_{prompt}", use_container_width=True):
                # Add the quick prompt as if user typed it
                process_user_input(prompt, chatbot, temperature, max_tokens, system_prompt)
                st.rerun()
        
        # Model info
        st.markdown("---")
        st.header("ℹ️ Model Info")
        st.info(f"""
        **Model**: TinyLlama-1.1B-Chat-v1.0
        **Parameters**: 1.1 Billion
        **License**: Apache 2.0
        **Local**: No API costs
        **Device**: {chatbot.device.upper()}
        **Precision**: {str(chatbot.dtype).split('.')[-1]}
        """)
    
    # Chat interface
    st.markdown("---")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show timestamp for assistant messages
            if message["role"] == "assistant" and "timestamp" in message:
                st.caption(f"⏱️ {message['timestamp']}")
    
    # Chat input
    if prompt := st.chat_input("Ask me anything... 💭", key="chat_input"):
        process_user_input(prompt, chatbot, temperature, max_tokens, system_prompt)

def process_user_input(user_input, chatbot, temperature=0.7, max_tokens=200, system_prompt=None):
    """Process user input and generate response"""
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            start_time = time.time()
            
            # Update chatbot parameters
            if hasattr(chatbot, 'pipe') and chatbot.pipe:
                # Get conversation history (exclude current user message)
                conversation_history = st.session_state.messages[:-1].copy()
                
                # Always initialize modified_history
                modified_history = []
                for msg in conversation_history:
                    if msg["role"] != "system":  # Skip existing system messages
                        modified_history.append(msg)
                
                response = chatbot.generate_response(user_input, modified_history)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Display response
            st.markdown(response)
            
            # Show response time
            timestamp = f"Response time: {response_time:.1f}s"
            st.caption(f"⏱️ {timestamp}")
        
        # Track response time
        st.session_state.response_times.append(response_time)
        st.session_state.conversation_count += 1
    
    # Save assistant response to history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response,
        "timestamp": timestamp
    })

def generate_conversation_export():
    """Generate formatted conversation for export"""
    export_data = {
        "export_date": datetime.now().isoformat(),
        "model": "TinyLlama-1.1B-Chat-v1.0",
        "total_messages": len(st.session_state.messages),
        "conversation": st.session_state.messages
    }
    
    # Create readable text format
    text_export = f"""TinyLlama Chatbot Conversation
Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: TinyLlama-1.1B-Chat-v1.0
Total Messages: {len(st.session_state.messages)}

{'='*50}

"""
    
    for i, message in enumerate(st.session_state.messages, 1):
        role = "🤖 Assistant" if message["role"] == "assistant" else "👤 You"
        text_export += f"{role}: {message['content']}\n\n"
        
        if message["role"] == "assistant" and "timestamp" in message:
            text_export += f"   {message['timestamp']}\n\n"
        
        text_export += "-" * 30 + "\n\n"
    
    return text_export

# Run the app
if __name__ == "__main__":
    main()

import os
import json
import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

# LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Construction Cost Estimator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS for Professional Look
# -----------------------------
st.markdown("""
<style>
    /* Ensure all text in the main body is light */
    .stApp {
        color: #ffffff;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #f8f9fa; /* Light gray for header text */
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px rgba(255,255,255,0.05); /* Light shadow for dark mode */
    }
    
    /* --- MONOCHROMATIC USER MESSAGE (Darker Background, White Text) --- */
    .user-message {
        background-color: #343a40; /* Dark gray background */
        color: #ffffff; /* **CRITICAL: White text for readability** */
        border: 1px solid #495057; 
        border-left: 5px solid #adb5bd; /* Light gray emphasis bar */
    }
    
    /* --- MONOCHROMATIC ASSISTANT MESSAGE (Lighter Background, White Text) --- */
    .assistant-message {
        background-color: #212529; /* Slightly lighter dark gray for contrast */
        color: #ffffff; /* **CRITICAL: White text for readability** */
        border: 1px solid #343a40;
        border-left: 5px solid #f8f9fa; /* Very light gray/white emphasis bar */
    }
    .metric-card {
        background-color: #212529; /* Dark background for cards */
        color: #ffffff; /* White text for readability */
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    .stButton>button {
        width: 100%;
    }
    /* Fix for text within the message content (e.g., strong tags) */
    .chat-message strong {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)
# -----------------------------
# Configuration
# -----------------------------
# NOTE: Replace with your actual Groq API Key
os.environ.setdefault("GROQ_API_KEY", "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")

# -----------------------------
# Pydantic Models (as provided)
# -----------------------------
class KeyQuantity(BaseModel):
    description: str = Field(description="Quantity description")
    value: float = Field(description="Numeric value")
    unit: str = Field(description="Unit")

class Scope(BaseModel):
    key_quantities: List[KeyQuantity] = Field(default_factory=list)
    features: List[str] = Field(default_factory=list)
    project_type: Optional[str] = Field(default=None, description="Type of construction project")
    location: Optional[str] = Field(default=None, description="Project location")
    timeline: Optional[str] = Field(default=None, description="Expected timeline")

class BudgetRange(BaseModel):
    low: float = Field(description="Low estimate in USD")
    likely: float = Field(description="Likely estimate in USD")
    high: float = Field(description="High estimate in USD")

class BillItem(BaseModel):
    item: str = Field(description="Bill item")
    quantity: float = Field(description="Quantity")
    unit: str = Field(description="Unit")
    rate: float = Field(description="Unit rate")
    total: float = Field(description="Line total")

class DetailedEstimate(BaseModel):
    bill_items: List[BillItem] = Field(default_factory=list)
    prelims: float = Field(default=0.0)
    contingency: float = Field(default=0.0)
    grand_total: float = Field(description="Total cost")

class Scenario(BaseModel):
    variation: str = Field(description="Variation description")
    impact: float = Field(description="Cost impact")

class Analysis(BaseModel):
    scenarios: List[Scenario] = Field(default_factory=list)
    benchmarks: List[str] = Field(default_factory=list)
    anomalies: List[str] = Field(default_factory=list)

# -----------------------------
# Enhanced Sample Data (as provided)
# -----------------------------
ENHANCED_SAMPLE_DATA = [
    {
        "project": "Metropolitan Office Tower",
        "location": "Urban CBD",
        "project_type": "Commercial Office",
        "gross_floor_area_sqm": 8500,
        "num_floors": 12,
        "cost_per_sqm": 3200,
        "total_cost": 27200000,
        "frame_type": "Steel frame",
        "foundation": "Piled foundation",
        "features": "High-spec finishes, curtain wall facade, VRF HVAC, raised floors, 200 parking spaces",
        "completion_year": 2023,
        "timeline_months": 18
    },
    {
        "project": "Greenfield Elementary School",
        "location": "Suburban",
        "project_type": "Educational",
        "gross_floor_area_sqm": 4200,
        "num_floors": 2,
        "cost_per_sqm": 2500,
        "total_cost": 10500000,
        "frame_type": "Concrete frame",
        "foundation": "Strip foundation",
        "features": "Standard finishes, natural ventilation, suspended ceilings, 25 classrooms, sports hall",
        "completion_year": 2023,
        "timeline_months": 14
    },
    {
        "project": "Luxury Apartment Tower",
        "location": "Urban premium zone",
        "project_type": "Residential High-rise",
        "gross_floor_area_sqm": 18000,
        "num_floors": 25,
        "cost_per_sqm": 2800,
        "total_cost": 50400000,
        "frame_type": "Concrete core with steel",
        "foundation": "Deep piled foundation",
        "features": "Premium finishes, VRF HVAC, 150 luxury units, concierge, gym, pool, 3-level basement parking",
        "completion_year": 2023,
        "timeline_months": 28
    },
    {
        "project": "Industrial Warehouse",
        "location": "Industrial zone",
        "project_type": "Industrial",
        "gross_floor_area_sqm": 15000,
        "num_floors": 1,
        "cost_per_sqm": 850,
        "total_cost": 12750000,
        "frame_type": "Steel portal frame",
        "foundation": "Ground bearing slab",
        "features": "Basic finishes, natural ventilation, high bay lighting, loading docks, office area",
        "completion_year": 2024,
        "timeline_months": 10
    }
]

# -----------------------------
# Initialize Session State & Models
# -----------------------------
@st.cache_resource
def initialize_models():
    """Initialize LLM and embeddings (cached)."""
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0) 
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return llm, embeddings

@st.cache_resource
def initialize_vectorstore(_embeddings, persist_dir: str = "./cost_chroma_db"):
    """Initialize and seed vectorstore (cached)."""
    vectorstore = Chroma(embedding_function=_embeddings, persist_directory=persist_dir)

    # Seed with enhanced data if empty
    try:
        # Simple check for existing data
        if not vectorstore._collection.count(): 
            texts = [json.dumps(d) for d in ENHANCED_SAMPLE_DATA]
            metadatas = [{"project_type": d["project_type"], "location": d["location"]} for d in ENHANCED_SAMPLE_DATA]
            vectorstore.add_texts(texts, metadatas=metadatas)
    except:
        # Fallback to seed data if initial check fails (e.g., first run)
        texts = [json.dumps(d) for d in ENHANCED_SAMPLE_DATA]
        metadatas = [{"project_type": d["project_type"], "location": d["location"]} for d in ENHANCED_SAMPLE_DATA]
        vectorstore.add_texts(texts, metadatas=metadatas)

    return vectorstore

# Initialize models
llm, embeddings = initialize_models()
vectorstore = initialize_vectorstore(embeddings)

# System message template
SYSTEM_MESSAGE_CONTENT = """You are a professional construction cost estimation assistant. Your primary function is to converse with the user to gather all required project scope details.

Required details include: Project type (Office, Residential, etc.), Total Floor Area (sqm), Number of Stories, Location (City, Urban/Suburban), Key Systems (HVAC, structure type), and any Target Budget/Timeline.

Ask clarifying questions naturally and conversationally. Be thorough but friendly.
DO NOT provide the final cost estimate in the chat.
When you have enough information (at minimum: project type, floor area, and key features), let the user know you have enough details to proceed with the cost estimate.
Then ask if they'd like to generate the estimate or add more details."""

# Initialize session state for conversation
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = [SystemMessage(content=SYSTEM_MESSAGE_CONTENT)]
if 'estimate_result' not in st.session_state:
    st.session_state.estimate_result = None
if 'show_estimate' not in st.session_state:
    st.session_state.show_estimate = False

# -----------------------------
# Cost Modeling Functions
# -----------------------------

def get_relevant_benchmarks(query: str, k: int = 3) -> str:
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

def extract_scope_from_conversation(conversation_text: str) -> Scope:
    scope_parser = PydanticOutputParser(pydantic_object=Scope)
    scope_prompt = ChatPromptTemplate.from_template(
        """Extract ALL project details from this conversation about a construction project.

Conversation:
{conversation}

Extract:
- Key quantities (floor area, number of floors, parking spaces, rooms, etc.) with values and units
- All features and specifications mentioned (frame type, finishes, systems, materials, etc.)
- Project type (office, residential, hospital, etc.)
- Location if mentioned
- Timeline if mentioned

Be thorough and capture every detail mentioned.

You MUST respond with ONLY valid JSON matching this format:
{format_instructions}"""
    ).partial(format_instructions=scope_parser.get_format_instructions())
    
    scope_chain = scope_prompt | llm | scope_parser
    return scope_chain.invoke({"conversation": conversation_text})

def quick_budget(scope: Scope) -> BudgetRange:
    budget_parser = PydanticOutputParser(pydantic_object=BudgetRange)
    budget_prompt = ChatPromptTemplate.from_template(
        """Based on these similar projects:
{context}

For this scope:
{scope}

Provide low, likely, and high budget estimates in USD.

You MUST respond with ONLY valid JSON matching this format:
{format_instructions}"""
    ).partial(format_instructions=budget_parser.get_format_instructions())
    
    budget_chain = budget_prompt | llm | budget_parser
    context = get_relevant_benchmarks(json.dumps(scope.model_dump()))
    return budget_chain.invoke({"context": context, "scope": json.dumps(scope.model_dump())})

def detailed_estimate(scope: Scope) -> DetailedEstimate:
    estimate_parser = PydanticOutputParser(pydantic_object=DetailedEstimate)
    estimate_prompt = ChatPromptTemplate.from_template(
        """Create a detailed construction estimate for this scope:
{scope}

Break down into bill items with quantities, units, rates, and totals.
Add prelims (10% of subtotal of bill items) and contingency (5% of subtotal of bill items).
Calculate grand_total as sum of all items + prelims + contingency.

You MUST respond with ONLY valid JSON matching this format:
{format_instructions}"""
    ).partial(format_instructions=estimate_parser.get_format_instructions())
    
    estimate_chain = estimate_prompt | llm | estimate_parser
    return estimate_chain.invoke({"scope": json.dumps(scope.model_dump())})

def scenario_analysis(estimate: DetailedEstimate) -> Analysis:
    scenario_parser = PydanticOutputParser(pydantic_object=Analysis)
    scenario_prompt = ChatPromptTemplate.from_template(
        """Analyze this construction estimate:
{estimate}

Consider these benchmarks:
{context}

Provide:
1. Alternative scenarios (material swaps, construction methods, timeline changes) and their cost impact.
2. Benchmark comparisons (how does this estimate compare to similar projects?).
3. Any cost anomalies or outliers.

You MUST respond with ONLY valid JSON matching this format:
{format_instructions}"""
    ).partial(format_instructions=scenario_parser.get_format_instructions())
    
    scenario_chain = scenario_prompt | llm | scenario_parser
    context = get_relevant_benchmarks(json.dumps(estimate.model_dump()))
    
    return scenario_chain.invoke({
        "context": context,
        "estimate": json.dumps(estimate.model_dump())
    })

def get_conversation_text() -> str:
    """Get full conversation as text."""
    text = ""
    for msg in st.session_state.conversation_history:
        if isinstance(msg, HumanMessage):
            text += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            text += f"Assistant: {msg.content}\n"
    return text

def generate_estimate() -> Dict[str, Any]:
    """Extract scope from conversation and generate complete cost estimate."""
    conversation = get_conversation_text()
    
    try:
        scope = extract_scope_from_conversation(conversation)
        budget = quick_budget(scope)
        estimate = detailed_estimate(scope)
        analysis = scenario_analysis(estimate)

        # Store analysis for future RAG
        text = json.dumps(analysis.model_dump())
        vectorstore.add_texts([text], metadatas=[{"type": "cost_model", "timestamp": datetime.now().isoformat()}])

        result = {
            "scope": scope.model_dump(),
            "budget_range": budget.model_dump(),
            "detailed_estimate": estimate.model_dump(),
            "analysis": analysis.model_dump()
        }
        
        return result

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# -----------------------------
# Streamlit UI
# -----------------------------

# Header
st.markdown('<div class="main-header">Construction Cost Estimator</div>', unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("Project Information")
    
    # View benchmark projects
    with st.expander("View Benchmark Projects"):
        for i, project in enumerate(ENHANCED_SAMPLE_DATA[:5]):
            st.markdown(f"**{project['project']}**")
            st.write(f"Type: {project['project_type']}")
            st.write(f"Area: {project['gross_floor_area_sqm']:,} sqm")
            st.write(f"Cost/sqm: ${project['cost_per_sqm']:,}")
            st.write(f"Total: ${project['total_cost']:,}")
            if i < len(ENHANCED_SAMPLE_DATA) - 1 and i < 4:
                st.divider()

    st.divider()

    # Actions
    st.header("Actions")

    if st.button("Reset Conversation", use_container_width=True):
        st.session_state.conversation_history = [SystemMessage(content=SYSTEM_MESSAGE_CONTENT)]
        st.session_state.estimate_result = None
        st.session_state.show_estimate = False
        st.rerun()

    if st.button("Generate Cost Estimate", use_container_width=True, type="primary"):
        with st.spinner("Generating comprehensive cost estimate..."):
            st.session_state.estimate_result = generate_estimate()
            st.session_state.show_estimate = True
        st.rerun()

    if st.session_state.estimate_result and not st.session_state.estimate_result.get("error"):
        json_str = json.dumps(st.session_state.estimate_result, indent=2)
        st.download_button(
            label="Download Estimate (JSON)",
            data=json_str,
            file_name=f"cost_estimate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
            type="secondary"
        )

# --- Main Content Area ---
if not st.session_state.show_estimate:
    # Chat interface
    st.header("Project Scope Discussion")

    # Display conversation
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.conversation_history[1:]: # Skip system message
            if isinstance(msg, HumanMessage):
                st.markdown(f'<div class="chat-message user-message"><strong>You:</strong><br>{msg.content}</div>',
                            unsafe_allow_html=True)
            elif isinstance(msg, AIMessage):
                st.markdown(f'<div class="chat-message assistant-message"><strong>Assistant:</strong><br>{msg.content}</div>',
                            unsafe_allow_html=True)

    # Chat input
    user_input = st.chat_input("Describe your construction project...")

    if user_input:
        # Add user message
        st.session_state.conversation_history.append(HumanMessage(content=user_input))

        # Get AI response
        with st.spinner("Thinking..."):
            response = llm.invoke(st.session_state.conversation_history)
            st.session_state.conversation_history.append(AIMessage(content=response.content))

        st.rerun()

else:
    # Display estimate results
    result = st.session_state.estimate_result
    
    # Back button
    if st.button("Back to Chat"):
        st.session_state.show_estimate = False
        st.rerun()

    if result.get("error"):
        st.error("Error generating estimate. Please check the scope details in the chat and try again.")
        st.code(result.get("traceback", "Unknown error"))
    else:
        st.header("Cost Estimate Report")

        # --- 1. Budget Overview ---
        st.subheader("1. Budget Range")
        budget = result['budget_range']
        col1, col2, col3 = st.columns(3)
        col1.metric("Low Estimate", f"${budget['low']:,.0f}")
        col2.metric("Likely Estimate", f"**${budget['likely']:,.0f}**")
        col3.metric("High Estimate", f"${budget['high']:,.0f}")

        st.divider()

        # --- 2. Project Scope Summary ---
        st.subheader("2. Project Scope Summary")
        scope = result['scope']

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Core Details**")
            st.write(f"**Type:** {scope.get('project_type', 'N/A')}")
            st.write(f"**Location:** {scope.get('location', 'N/A')}")
            st.write(f"**Timeline:** {scope.get('timeline', 'N/A')}")
        
        with col2:
            st.markdown("**Key Quantities**")
            for qty in scope['key_quantities']:
                st.write(f"- {qty['description']}: **{qty['value']:,.0f}** {qty['unit']}")
        
        if scope['features']:
            st.markdown("**Key Features & Specifications**")
            st.info(", ".join(scope['features']))

        st.divider()

        # --- 3. Detailed Estimate (B.O.Q.) ---
        st.subheader("3. Detailed Bill of Quantities")
        estimate = result['detailed_estimate']

        # Create table data
        bill_data = []
        for item in estimate['bill_items']:
            bill_data.append({
                "Item": item['item'],
                "Quantity": f"{item['quantity']:,.2f}",
                "Unit": item['unit'],
                "Rate ($)": f"{item['rate']:,.2f}",
                "Total ($)": f"{item['total']:,.2f}"
            })

        if bill_data:
            st.dataframe(bill_data, use_container_width=True, hide_index=True)

        # Summary Metrics
        col1, col2, col3, col4 = st.columns(4)
        # Calculate hard cost subtotal dynamically
        total_hard_cost = sum(float(item['Total ($)'].replace('$', '').replace(',', '')) for item in bill_data) if bill_data else 0
        
        col1.metric("Hard Cost Subtotal", f"${total_hard_cost:,.2f}")
        col2.metric("Preliminaries (10%)", f"${estimate['prelims']:,.2f}")
        col3.metric("Contingency (5%)", f"${estimate['contingency']:,.2f}")
        col4.metric("**GRAND TOTAL**", f"**${estimate['grand_total']:,.2f}**")

        st.divider()

        # --- 4. Scenario Analysis ---
        st.subheader("4. Risk and Value Analysis")
        analysis = result['analysis']

        colA, colB = st.columns(2)
        
        with colA:
            st.markdown("**Alternative Scenarios**")
            for scenario in analysis['scenarios']:
                impact_str = f"${scenario['impact']:,.2f}"
                if scenario['impact'] > 0:
                    st.error(f"Risk: {scenario['variation']} (Impact: {impact_str})")
                else:
                    st.success(f"Saving: {scenario['variation']} (Impact: {impact_str})")

        with colB:
            if analysis['anomalies']:
                st.markdown("**Cost Anomalies**")
                for anomaly in analysis['anomalies']:
                    st.warning(f"Anomaly: {anomaly}")
            
            if analysis['benchmarks']:
                st.markdown("**Benchmark Comparisons**")
                for benchmark in analysis['benchmarks']:
                    st.info(benchmark)
        
        st.divider()
        
        # --- 5. Raw JSON Output ---
        with st.expander("View Raw Structured JSON Output"):
            st.json(result)

# Footer
st.divider()
st.caption("Construction Cost Estimator | Powered by LangChain & Groq")
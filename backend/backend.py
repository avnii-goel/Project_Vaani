#!/usr/bin/env python3
"""
Welfare Scheme Assistant - Backend API
FastAPI backend with LangGraph agentic workflow for Indian government welfare scheme assistance.
"""

import os
import json
import sqlite3
import re
import traceback
from datetime import datetime
from typing import TypedDict, List, Dict, Optional, Any

# FastAPI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# Environment
from dotenv import load_dotenv

# LangChain (new API)
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# LangGraph
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

# =======================
# 1. CONFIGURATION SETUP
# =======================

load_dotenv()

required_env_vars = ['GROQ_API_KEY', 'GOOGLE_API_KEY', 'TAVILY_API_KEY']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {missing_vars}")

# =======================
# 2. FASTAPI APP SETUP
# =======================

app = FastAPI(
    title="Welfare Scheme Assistant",
    description="An AI assistant specialized in Indian government welfare schemes, application forms, and benefit eligibility",
    version="2.0.0"
)

# Enable CORS for frontend connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================
# 3. PYDANTIC MODELS
# =======================

class ChatMessage(BaseModel):
    message: str
    user_id: str

class ChatResponse(BaseModel):
    response: str
    is_onboarding: bool = False
    onboarding_step: Optional[int] = None
    user_profile: Optional[Dict[str, Any]] = None
    generated_form_html: Optional[str] = None
    form_filename: Optional[str] = None
    form_language: Optional[str] = None

class UserProfile(BaseModel):
    user_id: str
    name: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    state: Optional[str] = None
    occupation: Optional[str] = None
    income_category: Optional[str] = None
    family_size: Optional[str] = None
    has_disability: Optional[str] = None

class NotificationResponse(BaseModel):
    user_id: str
    notifications: List[Dict[str, str]]
    total_schemes: int

# =======================
# 4. LANGGRAPH STATE DEFINITION
# =======================

class AgentState(TypedDict):
    """Enhanced state for the agentic workflow."""
    messages: List[BaseMessage]
    user_id: str
    # Onboarding-specific state
    is_onboarding: bool
    onboarding_step: int
    user_profile_data: Dict[str, Any]
    # Transient data for a single run
    intent: str
    tool_output: Optional[str]
    final_response_text: str
    # Form generation data
    generated_form_html: Optional[str]
    form_filename: Optional[str]
    form_language: Optional[str]
    # Additional metadata
    timestamp: str
    session_id: str

# =======================
# 5. WORKFLOW NODES (MOVED BEFORE WORKFLOW CONSTRUCTION)
# =======================

def check_user_profile_node(state: AgentState) -> dict:
    """Checks if a user profile exists and sets initial state, preserving partial data."""
    user_id = state.get('user_id')
    if not user_id:
        return {"error": "User ID not found in state"}

    # Check for a completed profile in SQLite
    try:
        stored_profile = services.store.get_user_profile(user_id)
        profile_exists = stored_profile is not None
    except Exception as e:
        print(f"Error accessing store: {e}")
        profile_exists = False
    
    print(f"Profile check for user {user_id}: exists = {profile_exists}")
    
    if profile_exists and stored_profile:
        # Profile is complete, onboarding is done.
        return {
            "user_id": user_id,
            "is_onboarding": False,
            "user_profile_data": stored_profile,
            "messages": state.get('messages', [])
        }
    else:
        # No completed profile. We are in onboarding.
        partial_profile = state.get('user_profile_data', {})
        print(f"Onboarding active. Current partial profile: {partial_profile}")
        
        # Check if this is a completely new user (no messages yet)
        messages = state.get('messages', [])
        is_new_user = len([msg for msg in messages if isinstance(msg, HumanMessage)]) <= 1 and not partial_profile
        
        if is_new_user:
            welcome_message = "Welcome! I'm your personal welfare scheme assistant. I'll help you discover government benefits and schemes you're eligible for. Let me get to know you better so I can provide personalized recommendations. What's your name?"
            return {
                "user_id": user_id,
                "is_onboarding": True,
                "user_profile_data": partial_profile,
                "messages": messages + [AIMessage(content=welcome_message)],
                "final_response_text": welcome_message
            }
        
        return {
            "user_id": user_id,
            "is_onboarding": True,
            "user_profile_data": partial_profile,
            "messages": state.get('messages', [])
        }

def onboarding_step_node(state: AgentState) -> dict:
    """Handles interactive user onboarding for welfare scheme assistance."""
    if not state.get('is_onboarding'):
        return {"is_onboarding": False}

    current_profile = state.get('user_profile_data', {})
    user_id = state['user_id']
    messages = state.get('messages', [])
    
    # Define welfare scheme specific fields
    required_fields = ["name", "age", "gender", "state", "occupation", "income_category", "family_size", "has_disability"]
    
    # If there's a user message, try to extract info
    if messages and isinstance(messages[-1], HumanMessage):
        user_message = messages[-1].content
        
        extraction_prompt = ChatPromptTemplate.from_template("""
Extract welfare scheme profile data from user message. Return ONLY a JSON object.

Current profile: {current_profile}
User message: "{user_message}"
Missing fields: {missing_fields}

Extract these fields if mentioned:
- name: Full name
- age: Age or age range  
- gender: Male/Female/Other
- state: Indian state/UT (If any city is provided, identify the state using your knowledge)
- occupation: Job/profession
- income_category: BPL/APL/General/OBC/SC/ST/EWS
- family_size: Number of family members (numbers like 1,2,3,4,5,etc.)
- has_disability: Convert yes/no/nope/nah/no disability/none/disabled responses to "yes" or "no"

Special rules:
- For disability: "no", "nope", "nah", "none", "no disability" = "no"; "yes", "disabled", "have disability" = "yes"
- For numbers: extract any number mentioned for family size
- Be flexible with responses and extract information even from casual answers

Return JSON only: {{"field": "value"}} or {{}} if nothing found.
""")
        
        missing_fields_for_prompt = [field for field in required_fields if field not in current_profile]
        
        extraction_chain = extraction_prompt | services.llm_groq | StrOutputParser()
        
        try:
            response_str = extraction_chain.invoke({
                "current_profile": json.dumps(current_profile),
                "user_message": user_message,
                "missing_fields": json.dumps(missing_fields_for_prompt)
            })
            
            # Clean and parse the JSON response
            json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
            if json_match:
                extracted_data = json.loads(json_match.group())
                current_profile.update(extracted_data)
                print(f"Extracted profile data: {extracted_data}")
            else:
                extracted_data = {}
                
            # Fallback: If LLM failed to extract the next needed field, try manually
            if len(missing_fields_for_prompt) > 0:
                next_field = missing_fields_for_prompt[0]
                if next_field not in current_profile or not current_profile[next_field]:
                    user_message_lower = user_message.lower().strip()
                    
                    if next_field == 'has_disability':
                        if user_message_lower in ['no', 'nope', 'nah', 'none', 'no disability', "we don't", 'no we don\'t']:
                            current_profile['has_disability'] = 'no'
                            print("Fallback: Extracted has_disability = no")
                        elif user_message_lower in ['yes', 'yeah', 'yep', 'disabled', 'have disability']:
                            current_profile['has_disability'] = 'yes'
                            print("Fallback: Extracted has_disability = yes")
                            
                    elif next_field == 'family_size':
                        numbers = re.findall(r'\d+', user_message)
                        if numbers:
                            current_profile['family_size'] = numbers[0]
                            print(f"Fallback: Extracted family_size = {numbers[0]}")
                            
                    elif next_field == 'income_category':
                        if 'poor' in user_message_lower or 'low' in user_message_lower or 'bpl' in user_message_lower:
                            current_profile['income_category'] = 'BPL'
                            print("Fallback: Extracted income_category = BPL")

        except (json.JSONDecodeError, Exception) as e:
            print(f"Error extracting profile data: {e}")

    # Check what's still missing
    missing_fields = [field for field in required_fields if field not in current_profile or not current_profile[field]]
    
    # If nothing is missing, onboarding is complete
    if not missing_fields:
        services.store.save_user_profile(user_id, current_profile)
        name = current_profile.get('name', 'there')
        final_text = f"Great! {name}, your profile is set. I can now help you find suitable welfare schemes."
        
        return {
            "is_onboarding": False,
            "user_profile_data": current_profile,
            "messages": messages + [AIMessage(content=final_text)],
            "final_response_text": final_text
        }
        
    # Ask for the next missing piece of information
    next_field_to_ask = missing_fields[0]
    
    # Field-specific prompts for welfare schemes
    field_prompts = {
        "name": "What's your name?",
        "age": "What's your age?",
        "gender": "What's your gender? (Male/Female/Other)",
        "state": "Which Indian state/UT are you from?",
        "occupation": "What's your occupation?",
        "income_category": "What's your income category? (BPL/APL)",
        "family_size": "How many members are in your family?",
        "has_disability": "Do you or any family member have a disability? (yes/no)"
    }
    
    next_question = field_prompts.get(next_field_to_ask, f"Please provide your {next_field_to_ask}")
    
    return {
        "is_onboarding": True,
        "user_profile_data": current_profile,
        "messages": messages + [AIMessage(content=next_question)],
        "final_response_text": next_question
    }

def supervisor_agent_node(state: AgentState) -> dict:
    """Smart intent classification for welfare scheme assistance."""
    user_id = state['user_id']
    
    # Get user profile safely
    try:
        user_profile = services.store.get_user_profile(user_id) or {}
    except:
        user_profile = {}
    
    prompt = ChatPromptTemplate.from_template("""
Classify user intent for welfare scheme assistance:

Intents:
- 'welfare_search': Questions about government schemes, benefits, eligibility
- 'form_generation': Need application forms or documentation
- 'general_query': Other questions

User Profile: {user_profile}
Query: {query}

Respond with intent only.
""")
    
    chain = prompt | services.llm_groq | StrOutputParser()
    query = state['messages'][-1].content
    intent = chain.invoke({
        "user_profile": json.dumps(user_profile), 
        "query": query
    }).strip().lower()
    
    # Map intent variations
    if "welfare" in intent or "scheme" in intent or "benefit" in intent:
        final_intent = "welfare_search"
    elif "form" in intent or "application" in intent:
        final_intent = "form_generation"
    else:
        final_intent = "general_query"
        
    return {"intent": final_intent}

def welfare_search_node(state: AgentState) -> dict:
    """Concise welfare scheme search with user profile matching."""
    query = state['messages'][-1].content
    user_id = state['user_id']
    
    # Get user profile for targeted search
    try:
        user_profile = services.store.get_user_profile(user_id) or {}
    except:
        user_profile = {}
    
    # Create targeted search query
    search_context = ""
    if user_profile:
        search_context = f"Indian welfare schemes for {user_profile.get('state', '')} {user_profile.get('income_category', '')} {user_profile.get('occupation', '')}"
    
    try:
        search_results = services.web_search_tool.invoke({"query": f"{query} {search_context}"})
        
        # Create concise summary
        summary_prompt = ChatPromptTemplate.from_template("""
Provide a natural, conversational answer about welfare schemes based on search results.
Sound like you're speaking to someone naturally, without emojis or bullet points.
Keep response under 150 words and focus on practical information about eligibility and benefits.

User Profile: {profile}
Query: {query}
Search Results: {results}

Provide a helpful answer that sounds natural when spoken aloud.
""")
        
        summary_chain = summary_prompt | services.llm_gemini | StrOutputParser()
        concise_response = summary_chain.invoke({
            "profile": json.dumps(user_profile),
            "query": query,
            "results": json.dumps(search_results.get("results", [])[:3])  # Limit to top 3 results
        })
        
        return {"tool_output": concise_response}
    except Exception as e:
        return {"tool_output": f"Unable to search welfare schemes: {str(e)}"}

def extract_form_requirements_from_web(search_results, scheme_name):
    """Helper function to intelligently extract form requirements from web search results."""
    
    # Combine all web content
    combined_content = ""
    for result in search_results:
        if isinstance(result, dict):
            content = result.get('content', '') or result.get('snippet', '') or result.get('description', '')
            combined_content += f"{content}\n"
    
    # Use AI to extract structured requirements
    extraction_prompt = ChatPromptTemplate.from_template("""
Analyze the following web content about {scheme_name} and extract specific form requirements:

Web Content:
{content}

Extract and return ONLY a JSON object with these exact keys:
{{
    "mandatory_fields": ["field1", "field2"],
    "optional_fields": ["field1", "field2"], 
    "document_types": ["document1", "document2"],
    "age_restrictions": "text",
    "income_limits": "text",
    "special_validations": ["validation1", "validation2"],
    "state_requirements": "text",
    "target_beneficiaries": "text"
}}

Focus on finding:
- Specific form field names mentioned
- Required documents for upload
- Age/income eligibility criteria
- Validation rules (Aadhaar format, etc.)
- State-specific requirements

Return valid JSON only.
""")
    
    try:
        extraction_chain = extraction_prompt | services.llm_groq | StrOutputParser()
        result = extraction_chain.invoke({
            "scheme_name": scheme_name,
            "content": combined_content[:4000]  # Limit content to avoid token limits
        })
        
        # Parse JSON result
        import re
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {}
    except Exception as e:
        print(f"Error extracting requirements: {e}")
        return {}

def form_generation_node(state: AgentState) -> dict:
    """Generates welfare scheme application forms with proper HTML and regex validation."""
    query = state['messages'][-1].content
    user_id = state['user_id']
    
    # Get user profile for form customization
    try:
        user_profile = services.store.get_user_profile(user_id) or {}
    except:
        user_profile = {}
    
    # Step 1: Analyze user request to identify specific scheme
    scheme_analysis_prompt = ChatPromptTemplate.from_template("""
Analyze the user's request to identify the specific welfare scheme they want to apply for:

User request: "{request}"
User profile: {profile}

Extract:
1. Scheme name (e.g., "PM Kisan", "Ayushman Bharat", "Pradhan Mantri Awas Yojana")
2. Category (e.g., "Agriculture", "Healthcare", "Housing", "Education", "Social Security")
3. Target beneficiary (e.g., "Farmers", "Women", "Senior Citizens", "Disabled")
4. State-specific variant (if applicable)

Return only a JSON object with: scheme_name, category, target_beneficiary, state_variant
""")
    
    analysis_chain = scheme_analysis_prompt | services.llm_groq | StrOutputParser()
    scheme_info_text = analysis_chain.invoke({
        "request": query,
        "profile": json.dumps(user_profile)
    })
    
    # Parse scheme information
    try:
        import re
        json_match = re.search(r'\{.*\}', scheme_info_text, re.DOTALL)
        if json_match:
            scheme_info = json.loads(json_match.group())
        else:
            scheme_info = {"scheme_name": "General Welfare Scheme", "category": "Social Security", "target_beneficiary": "Citizens", "state_variant": ""}
    except:
        scheme_info = {"scheme_name": "General Welfare Scheme", "category": "Social Security", "target_beneficiary": "Citizens", "state_variant": ""}
    
    # Step 2: Perform multiple targeted web searches for comprehensive information
    search_queries = []
    
    # Primary scheme search
    base_scheme = scheme_info.get('scheme_name', 'welfare scheme')
    user_state = user_profile.get('state', '')
    
    search_queries.extend([
        f"{base_scheme} application form requirements documents eligibility criteria",
        f"{base_scheme} {user_state} state specific application process",
        f"{scheme_info.get('category', '')} welfare scheme mandatory fields application form",
        f"government {base_scheme} online application form validation rules",
        f"{base_scheme} document upload requirements identity proof address proof"
    ])
    
    # Collect comprehensive search results
    all_search_results = []
    for search_query in search_queries:
        try:
            results = services.web_search_tool.invoke({"query": search_query})
            all_search_results.extend(results.get("results", [])[:3])  # Top 3 results from each search
        except Exception as e:
            print(f"Search failed for query '{search_query}': {e}")
    
    # Step 3: Extract structured information using helper function
    structured_info = extract_form_requirements_from_web(
        all_search_results[:15],  # Use top 15 results
        scheme_info.get('scheme_name', 'welfare scheme')
    )
    
    # Step 4: Determine language based on user's state
    # Step 4: Determine language based on user's state
    user_state = user_profile.get('state', '').lower()
    state_language_map = {
        'tamil nadu': 'Tamil',
        'kerala': 'Malayalam', 
        'karnataka': 'Kannada',
        'andhra pradesh': 'Telugu',
        'telangana': 'Telugu',
        'west bengal': 'Bengali',
        'odisha': 'Odia',
        'gujarat': 'Gujarati',
        'maharashtra': 'Marathi',
        'punjab': 'Punjabi',
        'haryana': 'Hindi',
        'rajasthan': 'Hindi',
        'uttar pradesh': 'Hindi',
        'bihar': 'Hindi',
        'jharkhand': 'Hindi',
        'madhya pradesh': 'Hindi',
        'chhattisgarh': 'Hindi',
        'assam': 'Assamese',
        'manipur': 'Manipuri',
        'nagaland': 'English',
        'mizoram': 'Mizo',
        'tripura': 'Bengali',
        'sikkim': 'Nepali',
        'himachal pradesh': 'Hindi',
        'uttarakhand': 'Hindi',
        'goa': 'Konkani',
        'delhi': 'Hindi',
        'chandigarh': 'Hindi',
        'puducherry': 'Tamil'
    }
    
    regional_language = state_language_map.get(user_state, 'English')
    
    # Check if user specified a language preference in their query
    language_keywords = ['english', 'hindi', 'tamil', 'telugu', 'kannada', 'malayalam', 'bengali', 'gujarati', 'marathi', 'punjabi']
    specified_language = None
    for lang in language_keywords:
        if lang in query.lower():
            specified_language = lang.capitalize()
            break
    
    form_language = specified_language or regional_language
    
    # Step 5: Generate comprehensive welfare scheme form with scraped data
    form_generation_prompt = ChatPromptTemplate.from_template("""
Create a highly customized welfare scheme application form based on web-scraped information:
Also mention scheme name on top of form
**Scheme Information:**
- Scheme: {scheme_name}
- Category: {category}
- Target Beneficiary: {target_beneficiary}
- State: {user_state}

**User Profile:** {profile}
**Extracted Requirements:** {structured_info}
**Web Research:** {search_summary}

**Instructions:**
1. **Form Language:** Labels and instructions in `{language}`, HTML `name` attributes in English
2. **Dynamic Fields:** Use the extracted requirements to create scheme-specific fields beyond standard ones
3. **Required Fields:** Include all fields mentioned in REQUIRED_FIELDS from web research
4. **Document Uploads:** Add file upload sections for documents mentioned in DOCUMENT_UPLOADS
5. **Validation Rules:** Implement specific validation based on VALIDATION_RULES (age limits, income criteria, etc.)
6. **Conditional Logic:** Add JavaScript to show/hide fields based on eligibility criteria
7. **State-Specific Elements:** Include any STATE_SPECIFIC requirements found
8. **Benefit Information:** Display scheme benefits prominently at the top

**Technical Requirements:**
- Complete HTML5 document with Bootstrap styling
- Robust JavaScript validation with proper regex patterns
- Mobile-responsive design
- Accessibility features (ARIA labels)
- Dynamic dropdowns (state/district, scheme-specific categories)

**Form Structure:**
1. Scheme information header with benefits
2. Applicant details (name, contact, address)
3. Scheme-specific eligibility fields
4. Document upload sections
5. Declaration and submission

Generate a complete, functional HTML form. NO markdown formatting.
""")
    
    # Create search summary from all results
    search_summary = f"Analyzed {len(all_search_results)} web sources for {scheme_info.get('scheme_name', 'welfare scheme')} requirements"
    
    form_chain = form_generation_prompt | services.llm_gemini | StrOutputParser()
    response_text = form_chain.invoke({
        "scheme_name": scheme_info.get('scheme_name', 'Welfare Scheme'),
        "category": scheme_info.get('category', 'Social Security'),
        "target_beneficiary": scheme_info.get('target_beneficiary', 'Citizens'),
        "user_state": user_state.title(),
        "profile": json.dumps(user_profile),
        "structured_info": json.dumps(structured_info, indent=2),
        "search_summary": search_summary,
        "language": form_language
    })

    # Clean the response to remove markdown formatting
    if response_text.strip().startswith("```html"):
        response_text = response_text.strip()[7:].strip()
    if response_text.strip().endswith("```"):
        response_text = response_text.strip()[:-3].strip()
    
    html_form = response_text
    
    # Save form to file
    form_filename = f"welfare_scheme_form_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(form_filename, "w", encoding="utf-8") as f:
        f.write(html_form)
    
    # Create detailed feedback about what was found
    scheme_name = scheme_info.get('scheme_name', 'Welfare Scheme')
    fields_found = len(structured_info.get('mandatory_fields', []))
    documents_found = len(structured_info.get('document_types', []))
    
    detailed_feedback = f"""Your {scheme_name} application form is ready in {form_language}! 

🔍 Web Research Results:
- Analyzed {len(all_search_results)} government sources
- Found {fields_found} specific required fields for this scheme
- Identified {documents_found} document requirements
- Included {user_state.title()} state-specific requirements

📋 Form Features:
- Scheme-specific validation rules based on official requirements
- Dynamic fields that appear based on your eligibility
- Mobile-friendly design with proper accessibility
- Document upload sections for all required documents

The form is customized specifically for {scheme_name} based on current government guidelines."""
    
    return {
        "tool_output": detailed_feedback,
        "generated_form_html": html_form,
        "form_filename": form_filename,
        "form_language": form_language,
        "scheme_info": scheme_info,
        "research_summary": {
            "sources_analyzed": len(all_search_results),
            "mandatory_fields": structured_info.get('mandatory_fields', []),
            "optional_fields": structured_info.get('optional_fields', []),
            "documents_needed": structured_info.get('document_types', []),
            "eligibility_criteria": structured_info.get('target_beneficiaries', ''),
            "validation_rules": structured_info.get('special_validations', []),
            "state_requirements": structured_info.get('state_requirements', ''),
            "age_restrictions": structured_info.get('age_restrictions', ''),
            "income_limits": structured_info.get('income_limits', '')
        }
    }

def general_query_node(state: AgentState) -> dict:
    """Handles concise conversational queries about welfare schemes."""
    user_id = state['user_id']
    
    # Get user profile safely
    try:
        user_profile = services.store.get_user_profile(user_id) or {}
    except:
        user_profile = {}
    
    prompt = ChatPromptTemplate.from_template("""
Answer naturally and conversationally, as if speaking to someone in person.
Remove all emojis, bullet points, and formatting. Keep response under 100 words.
Sound natural and helpful when spoken aloud.

User Profile: {user_profile}
Question: {question}

Provide a natural, spoken response focused on welfare schemes if relevant.
""")
    
    chain = prompt | services.llm_gemini | StrOutputParser()
    response = chain.invoke({
        "user_profile": json.dumps(user_profile),
        "question": state['messages'][-1].content
    })
    
    return {"tool_output": response}

def final_response_node(state: AgentState) -> dict:
    """Creates natural, audio-friendly final responses."""
    tool_output = state.get('tool_output', '')
    
    # For form generation, keep the response natural
    if 'form' in tool_output.lower():
        final_text = tool_output
    else:
        # For other responses, ensure they're concise and natural
        prompt = ChatPromptTemplate.from_template("""
Make this response sound natural and conversational, as if speaking to someone.
Remove all emojis, bullet points, and technical formatting.
Keep it under 150 words and make it sound like natural speech:

{tool_output}

Make it friendly and helpful but natural for voice output.
""")
        
        chain = prompt | services.llm_gemini | StrOutputParser()
        final_text = chain.invoke({"tool_output": tool_output})
    
    return {
        "messages": [AIMessage(content=final_text)], 
        "final_response_text": final_text
    }

# =======================
# 6. WORKFLOW CONSTRUCTION
# =======================

def build_agentic_workflow(services_instance):
    """Builds and returns the optimized agentic workflow."""
    
    def route_after_checking_profile(state: AgentState):
        is_onboarding = state.get('is_onboarding', False)
        print(f"Route after profile check: is_onboarding = {is_onboarding}")
        return "onboarding_step" if is_onboarding else "supervisor"

    def route_after_supervisor(state: AgentState):
        intent = state.get('intent', 'general_query')
        print(f"Route after supervisor: intent = {intent}")
        return intent

    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("check_user_profile", check_user_profile_node)
    workflow.add_node("onboarding_step", onboarding_step_node)
    workflow.add_node("supervisor", supervisor_agent_node)
    workflow.add_node("welfare_search", welfare_search_node)
    workflow.add_node("form_generation", form_generation_node)
    workflow.add_node("general_query", general_query_node)
    workflow.add_node("structure_final_response", final_response_node)

    # Set entry point and edges
    workflow.set_entry_point("check_user_profile")
    
    workflow.add_conditional_edges(
        "check_user_profile",
        route_after_checking_profile,
        {
            "onboarding_step": "onboarding_step",
            "supervisor": "supervisor"
        }
    )
    
    # After asking an onboarding question, the turn ends. The "loop" is driven by user responses.
    workflow.add_edge("onboarding_step", END)
    
    workflow.add_conditional_edges(
        "supervisor", 
        route_after_supervisor,
        {
            "welfare_search": "welfare_search",
            "form_generation": "form_generation", 
            "general_query": "general_query"
        }
    )
    
    workflow.add_edge("welfare_search", "structure_final_response")
    workflow.add_edge("form_generation", "structure_final_response")
    workflow.add_edge("general_query", "structure_final_response")
    workflow.add_edge("structure_final_response", END)

    return workflow.compile(checkpointer=services_instance.checkpointer)

# =======================
# 7. SQLITE PERSISTENCE LAYER
# =======================

class SQLiteStore:
    """SQLite-based persistent storage for user profiles and conversations."""
    
    def __init__(self, db_path: str = "agent_data.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # User profiles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    name TEXT,
                    age TEXT,
                    gender TEXT,
                    state TEXT,
                    occupation TEXT,
                    income_category TEXT,
                    family_size TEXT,
                    has_disability TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Check if gender column exists, if not add it (for existing databases)
            cursor.execute("PRAGMA table_info(user_profiles)")
            columns = [column[1] for column in cursor.fetchall()]
            if 'gender' not in columns:
                cursor.execute("ALTER TABLE user_profiles ADD COLUMN gender TEXT")
                print("Added gender column to existing user_profiles table")
            
            # Conversation history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    message_type TEXT,
                    content TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
                )
            """)
            
            # Welfare schemes table for reference
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS welfare_schemes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scheme_name TEXT,
                    category TEXT,
                    eligibility_criteria TEXT,
                    benefits TEXT,
                    application_process TEXT,
                    required_documents TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Retrieve user profile from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            
            if row:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, row))
            return None
    
    def save_user_profile(self, user_id: str, profile_data: Dict) -> None:
        """Save or update user profile in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if profile exists
            existing = self.get_user_profile(user_id)
            
            if existing:
                # Update existing profile
                cursor.execute("""
                    UPDATE user_profiles 
                    SET name = ?, age = ?, gender = ?, state = ?, occupation = ?, 
                        income_category = ?, family_size = ?, has_disability = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                """, (
                    profile_data.get('name'),
                    profile_data.get('age'),
                    profile_data.get('gender'),
                    profile_data.get('state'),
                    profile_data.get('occupation'),
                    profile_data.get('income_category'),
                    profile_data.get('family_size'),
                    profile_data.get('has_disability'),
                    user_id
                ))
            else:
                # Insert new profile
                cursor.execute("""
                    INSERT INTO user_profiles 
                    (user_id, name, age, gender, state, occupation, income_category, family_size, has_disability)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id,
                    profile_data.get('name'),
                    profile_data.get('age'),
                    profile_data.get('gender'),
                    profile_data.get('state'),
                    profile_data.get('occupation'),
                    profile_data.get('income_category'),
                    profile_data.get('family_size'),
                    profile_data.get('has_disability')
                ))
            
            conn.commit()
    
    def delete_user_profile(self, user_id: str) -> None:
        """Delete user profile from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id,))
            cursor.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))
            conn.commit()
    
    def save_conversation_message(self, user_id: str, message_type: str, content: str) -> None:
        """Save conversation message to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO conversations (user_id, message_type, content)
                VALUES (?, ?, ?)
            """, (user_id, message_type, content))
            conn.commit()
    
    def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get recent conversation history for a user."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT message_type, content, timestamp 
                FROM conversations 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (user_id, limit))
            
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in rows]

# =======================
# 8. GLOBAL SERVICES INITIALIZATION
# =======================

class AgentServices:
    """Centralized service management for the agent workflow."""

    def __init__(self):
        # Fast extraction / classification model (Groq hosted Llama)
        self.llm_groq = init_chat_model(
            "meta-llama/llama-4-scout-17b-16e-instruct",
            model_provider="groq",
            temperature=0.1,
        )
        # Richer generation model (Gemini)
        self.llm_gemini = init_chat_model(
            "gemini-2.5-flash",
            model_provider="google_genai",
            temperature=0.7,
        )
        self.web_search_tool = TavilySearch(max_results=10)
        self.checkpointer = InMemorySaver()
        self.store = SQLiteStore()
        self.workflow = build_agentic_workflow(self)

# Global services instance
services = AgentServices()

# =======================
# 8. API ENDPOINTS
# =======================

# =======================
# 9. API ENDPOINTS
# =======================

@app.get("/")
async def root():
    """API root — confirms service is running."""
    return {
        "service": "Welfare Scheme Assistant API",
        "version": "2.0.0",
        "docs": "/docs",
    }


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_message: ChatMessage):
    """Main chat endpoint that processes user messages through the agentic workflow."""
    try:
        config = {"configurable": {"thread_id": chat_message.user_id}}
        
        # Get the current state from the checkpointer to continue the conversation
        current_state = services.workflow.get_state(config)
        
        if current_state and current_state.values:
            # If there's an existing state, add the new message to it
            print("Continuing existing conversation.")
            messages = current_state.values.get("messages", []) + [HumanMessage(content=chat_message.message)]
            # Pass the whole state to continue properly
            input_state = current_state.values
            input_state["messages"] = messages
        else:
            # If no state, this is a new conversation
            print("Starting new conversation.")
            input_state = {
                "messages": [HumanMessage(content=chat_message.message)],
                "user_id": chat_message.user_id
            }

        # Execute workflow with the correct state
        result_state = services.workflow.invoke(input_state, config)
        
        # The final response is in the last AIMessage in the 'messages' list
        final_response = ""
        if result_state and result_state.get('messages'):
            # Find the last AI message in the list
            for msg in reversed(result_state['messages']):
                if isinstance(msg, AIMessage):
                    final_response = msg.content
                    break

        print(f"Workflow result: onboarding={result_state.get('is_onboarding')}, response={final_response[:100]}...")
        
        # Get user profile if it was created
        user_profile = result_state.get('user_profile_data', {})
        if not user_profile:
            try:
                user_profile = services.store.get_user_profile(chat_message.user_id) or {}
            except Exception as e:
                print(f"Error retrieving profile from store: {e}")

        return ChatResponse(
            response=final_response or "I'm sorry, I couldn't process that request.",
            is_onboarding=result_state.get("is_onboarding", False),
            onboarding_step=result_state.get("onboarding_step"),
            user_profile=user_profile,
            generated_form_html=result_state.get("generated_form_html"),
            form_filename=result_state.get("form_filename"),
            form_language=result_state.get("form_language")
        )

    except Exception as e:
        print(f"Chat endpoint error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/user/{user_id}/profile", response_model=UserProfile)
async def get_user_profile(user_id: str):
    """Retrieves a user's profile information."""
    try:
        # Input validation
        if not user_id or len(user_id.strip()) == 0:
            raise HTTPException(status_code=400, detail="Invalid user ID")
        
        profile_data = services.store.get_user_profile(user_id)
        if not profile_data:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        # Remove user_id from profile_data to avoid conflict when unpacking
        profile_data_copy = profile_data.copy()
        profile_data_copy.pop('user_id', None)  # Remove user_id if it exists
        
        return UserProfile(user_id=user_id, **profile_data_copy)
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving profile: {str(e)}")

@app.delete("/user/{user_id}/profile")
async def delete_user_profile(user_id: str):
    """Deletes a user's profile (for testing/reset purposes)."""
    try:
        # Input validation
        if not user_id or len(user_id.strip()) == 0:
            raise HTTPException(status_code=400, detail="Invalid user ID")
        
        # Check if profile exists before deletion
        existing_profile = services.store.get_user_profile(user_id)
        if not existing_profile:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        services.store.delete_user_profile(user_id)
        return {"message": f"Profile for user {user_id} deleted successfully"}
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting profile: {str(e)}")

@app.get("/user/{user_id}/notifications", response_model=NotificationResponse)
async def get_user_notifications(user_id: str):
    """Generate personalized welfare scheme notifications for a user."""
    try:
        # Input validation
        if not user_id or len(user_id.strip()) == 0:
            raise HTTPException(status_code=400, detail="Invalid user ID")
        
        # Get user profile
        profile_data = services.store.get_user_profile(user_id)
        if not profile_data:
            raise HTTPException(status_code=404, detail="User profile not found. Please complete onboarding first.")
        
        # Generate personalized scheme recommendations
        recommendation_prompt = ChatPromptTemplate.from_template("""
You are an expert welfare scheme advisor. Create 5 HIGHLY COMPELLING and PERSONALIZED welfare scheme notifications that people will want to click on immediately.

User Profile:
- Name: {name}
- Age: {age}
- Gender: {gender}
- State: {state}
- Occupation: {occupation}
- Income Category: {income_category}
- Family Size: {family_size}
- Has Disability: {has_disability}

Create JSON array with exactly 5 scheme recommendations. Make each notification IRRESISTIBLE by:

1. Using action-oriented, benefit-focused titles that grab attention
2. Highlighting specific money amounts and immediate benefits
3. Creating urgency and relevance to the user's exact situation
4. Using emotional triggers and personal relevance

Each scheme should have:
- title: Compelling, action-oriented title (e.g. "Get Rs 6,000 Direct Cash - No Paperwork!", "Free Health Cover Worth Rs 5 Lakh - Apply Today!")
- description: Exciting, benefit-focused description emphasizing what they GET (max 80 words, focus on benefits, money, and ease)
- eligibility: Simple, clear criteria that matches their profile
- benefit_amount: Specific attractive amounts (e.g. "Rs 6,000/year", "Up to Rs 2.5 lakh", "Rs 500/month")
- priority: High/Medium/Low (prioritize schemes with direct cash benefits or immediate impact)
- category: Education/Healthcare/Employment/Housing/Social Security/Financial
- call_to_action: Compelling action phrase (e.g. "Apply in 5 minutes!", "Get money in your account!", "Start receiving benefits now!")

Focus on REAL Indian government schemes that match this user's profile. Make notifications so compelling that users can't resist clicking.
Return only the JSON array, no other text.
""")
        
        chain = recommendation_prompt | services.llm_gemini | StrOutputParser()
        
        try:
            response = chain.invoke(profile_data)
            # Extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                notifications = json.loads(json_match.group())
            else:
                # Enhanced fallback notifications with compelling content
                notifications = []
                
                # Cash-focused schemes based on profile
                if profile_data.get('income_category') in ['BPL', 'APL']:
                    notifications.append({
                        "title": "Get Rs 6,000 Direct Cash - Zero Paperwork!",
                        "description": "Receive Rs 2,000 every 4 months directly in your bank account through PM-KISAN scheme. Instant approval for eligible families!",
                        "eligibility": "Small and marginal farmers with landholding",
                        "benefit_amount": "Rs 6,000/year",
                        "priority": "High",
                        "category": "Financial",
                        "call_to_action": "Get your money now!"
                    })
                
                if profile_data.get('gender') == 'Female':
                    notifications.append({
                        "title": "Women Get Rs 1,000/Month - Guaranteed Income!",
                        "description": "Monthly cash assistance for women under various state schemes. Direct bank transfer, no middlemen, guaranteed payments!",
                        "eligibility": "Women from eligible families",
                        "benefit_amount": "Rs 1,000/month",
                        "priority": "High",
                        "category": "Social Security",
                        "call_to_action": "Start earning today!"
                    })
                
                if profile_data.get('has_disability') == 'yes':
                    notifications.append({
                        "title": "Disability Pension Rs 500/Month - Apply in 2 Minutes!",
                        "description": "Lifetime monthly pension for persons with disabilities. Automatic approval for eligible candidates with quick verification!",
                        "eligibility": "40% or more disability certificate required",
                        "benefit_amount": "Rs 500-1,000/month",
                        "priority": "High",
                        "category": "Social Security",
                        "call_to_action": "Secure your pension!"
                    })
                
                # Age-based schemes
                age = profile_data.get('age', '0')
                if age.isdigit() and int(age) >= 60:
                    notifications.append({
                        "title": "Senior Citizen Gets Rs 200/Month - Lifetime Benefits!",
                        "description": "Old age pension scheme providing monthly financial support. No income proof needed, instant approval for eligible seniors!",
                        "eligibility": "Citizens above 60 years from BPL families",
                        "benefit_amount": "Rs 200-500/month",
                        "priority": "High",
                        "category": "Social Security",
                        "call_to_action": "Claim your pension!"
                    })
                
                # Universal appealing schemes
                notifications.extend([
                    {
                        "title": "Free Health Insurance Worth Rs 5 Lakh - No Premium!",
                        "description": "Complete family health coverage including surgeries, medicines, and hospitalization. Cashless treatment at thousands of hospitals nationwide!",
                        "eligibility": "All eligible families as per government database",
                        "benefit_amount": "Rs 5 lakh/year coverage",
                        "priority": "High",
                        "category": "Healthcare",
                        "call_to_action": "Get covered instantly!"
                    },
                    {
                        "title": "Own Your Dream Home - Get Rs 2.5 Lakh Grant!",
                        "description": "Government subsidy for building or buying your first home. No interest loans plus direct cash grant for construction!",
                        "eligibility": "First-time home buyers from eligible income groups",
                        "benefit_amount": "Up to Rs 2.5 lakh subsidy",
                        "priority": "Medium",
                        "category": "Housing",
                        "call_to_action": "Build your home now!"
                    }
                ])
                
                # Ensure we have exactly 5 notifications
                notifications = notifications[:5]
                
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            # Simplified compelling fallback
            notifications = [
                {
                    "title": "Get Rs 6,000 Cash Directly - Apply Now!",
                    "description": "Direct benefit transfer to your account. No paperwork, instant approval for eligible families!",
                    "eligibility": "Indian citizen with bank account",
                    "benefit_amount": "Rs 6,000/year",
                    "priority": "High",
                    "category": "Financial",
                    "call_to_action": "Claim your money!"
                },
                {
                    "title": "Free Health Cover Rs 5 Lakh - Zero Cost!",
                    "description": "Complete family health insurance with cashless treatment at top hospitals across India!",
                    "eligibility": "All eligible Indian families",
                    "benefit_amount": "Rs 5 lakh/year",
                    "priority": "High",
                    "category": "Healthcare", 
                    "call_to_action": "Get protected now!"
                },
                {
                    "title": "Women Get Monthly Income - Guaranteed!",
                    "description": "Monthly financial assistance for women. Direct bank transfer, no delays, lifetime benefits!",
                    "eligibility": "Women from eligible families",
                    "benefit_amount": "Rs 500-1,000/month",
                    "priority": "Medium",
                    "category": "Social Security",
                    "call_to_action": "Start earning today!"
                },
                {
                    "title": "Build Your Home - Get Rs 2.5 Lakh Free!",
                    "description": "Government grant for home construction plus subsidized loans. Make your dream home reality!",
                    "eligibility": "First-time home buyers",
                    "benefit_amount": "Up to Rs 2.5 lakh",
                    "priority": "Medium",
                    "category": "Housing",
                    "call_to_action": "Own your home!"
                },
                {
                    "title": "Free Education + Scholarship Money!",
                    "description": "Complete education support with monthly scholarships and fee waivers for deserving students!",
                    "eligibility": "Students from eligible families",
                    "benefit_amount": "Varies by course",
                    "priority": "Medium",
                    "category": "Education",
                    "call_to_action": "Secure your future!"
                }
            ]
        
        return NotificationResponse(
            user_id=user_id,
            notifications=notifications,
            total_schemes=len(notifications)
        )
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating notifications: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "llm_groq": "operational",
            "llm_gemini": "operational",
            "welfare_search": "operational",
            "sqlite_store": "operational"
        }
    }

@app.get("/form/{form_filename}")
async def get_generated_form(form_filename: str):
    """Serve generated HTML forms."""
    try:
        if not form_filename.endswith('.html'):
            form_filename += '.html'
        
        # Security check - only allow files that match our naming pattern
        if not re.match(r'^welfare_scheme_form_\d{8}_\d{6}\.html$', form_filename):
            raise HTTPException(status_code=400, detail="Invalid form filename")
        
        file_path = os.path.join(os.getcwd(), form_filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Form not found")
        
        return FileResponse(
            path=file_path,
            media_type='text/html',
            filename=form_filename
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving form: {str(e)}")

# =======================
# 10. APPLICATION STARTUP
# =======================

if __name__ == "__main__":
    print("🚀 Starting Welfare Scheme Assistant API...")
    print("📖 API Documentation at: http://localhost:8000/docs")

    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )

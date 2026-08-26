# ============================================================
# LANGGRAPH MULTI-AGENT REAL ESTATE AI
# GOOGLE COLAB - COMPLETE SINGLE CELL
# ============================================================

# ------------------------------------------------------------
# 1. INSTALL PACKAGES
# -----------------------------------------------------------


# ------------------------------------------------------------
# 2. IMPORTS
# ------------------------------------------------------------

import os
from typing import TypedDict

from getpass import getpass

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END


# ------------------------------------------------------------
# 3. OPENAI API KEY
# ------------------------------------------------------------

if not os.environ.get("OPENAI_API_KEY"):

    api_key = getpass("Enter your OpenAI API Key: ")

    os.environ["OPENAI_API_KEY"] = api_key


# ------------------------------------------------------------
# 4. TEST API KEY
# ------------------------------------------------------------

if not os.environ.get("OPENAI_API_KEY"):

    raise ValueError("OPENAI_API_KEY was not provided.")


print("API key loaded successfully.")


# ------------------------------------------------------------
# 5. CREATE LLM
# ------------------------------------------------------------

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

print("LLM initialized successfully.")


# ============================================================
# 6. SHARED LANGGRAPH STATE
# ============================================================

class RealEstateState(TypedDict, total=False):

    location: str
    budget: float
    property_type: str
    bedrooms: int
    down_payment: float

    property_analysis: str
    market_analysis: str
    location_analysis: str
    financial_analysis: str

    final_report: str


# ============================================================
# 7. PROPERTY DATABASE
# ============================================================

PROPERTY_DATABASE = """
PROPERTY 1
Address: 123 Oak Street, Austin, Texas
Price: $620,000
Type: Single Family Home
Bedrooms: 4
Bathrooms: 3
Area: 2350 sq ft
Year Built: 2018
Features: Garage, backyard, modern kitchen, solar panels

PROPERTY 2
Address: 456 Lake View Drive, Austin, Texas
Price: $645,000
Type: Single Family Home
Bedrooms: 4
Bathrooms: 2.5
Area: 2500 sq ft
Year Built: 2020
Features: Large backyard, garage, smart home system

PROPERTY 3
Address: 789 Green Valley Road, Austin, Texas
Price: $590,000
Type: Single Family Home
Bedrooms: 3
Bathrooms: 2
Area: 2100 sq ft
Year Built: 2017
Features: Garage, garden, renovated kitchen, quiet neighborhood

PROPERTY 4
Address: 321 Sunset Avenue, Austin, Texas
Price: $680,000
Type: Single Family Home
Bedrooms: 4
Bathrooms: 3
Area: 2700 sq ft
Year Built: 2021
Features: Swimming pool, garage, modern interior, large backyard
"""


# ============================================================
# 8. AGENT 1 - PROPERTY RESEARCH
# ============================================================

def property_agent(state: RealEstateState):

    prompt = f"""
You are the Property Research Agent.

CLIENT REQUIREMENTS:

Location:
{state["location"]}

Maximum Budget:
${state["budget"]:,.0f}

Property Type:
{state["property_type"]}

Minimum Bedrooms:
{state["bedrooms"]}


AVAILABLE PROPERTY DATA:

{PROPERTY_DATABASE}


TASK:

Analyze every property.

Select properties that:

1. Are within the client's budget.
2. Match the requested property type.
3. Have at least the required number of bedrooms.

For each suitable property provide:

- Address
- Price
- Bedrooms
- Bathrooms
- Area
- Price per square foot
- Year built
- Features
- Why it matches

Do not invent information.
Only use the supplied property data.
"""

    response = llm.invoke(prompt)

    return {
        "property_analysis": response.content
    }


# ============================================================
# 9. AGENT 2 - MARKET ANALYSIS
# ============================================================

def market_agent(state: RealEstateState):

    prompt = f"""
You are the Real Estate Market Analysis Agent.

CLIENT LOCATION:

{state["location"]}


PROPERTY RESEARCH AGENT RESULT:

{state["property_analysis"]}


ORIGINAL PROPERTY DATA:

{PROPERTY_DATABASE}


TASK:

Compare the selected properties.

Analyze:

1. Asking price
2. Price per square foot
3. Property size
4. Year built
5. Number of bedrooms
6. Number of bathrooms
7. Features
8. Overall value

Identify:

- Best value
- Most expensive
- Best price per square foot
- Strongest overall option

Do not invent real-time market statistics.

Use only the supplied information.
"""

    response = llm.invoke(prompt)

    return {
        "market_analysis": response.content
    }


# ============================================================
# 10. AGENT 3 - LOCATION ANALYSIS
# ============================================================

def location_agent(state: RealEstateState):

    prompt = f"""
You are the Location Analysis Agent.

LOCATION:

{state["location"]}


PROPERTY ANALYSIS:

{state["property_analysis"]}


TASK:

Evaluate the properties from a location and lifestyle
perspective.

Discuss:

- Neighborhood characteristics if provided
- Property surroundings if provided
- Lifestyle suitability
- Backyard/garden/pool/garage benefits
- Potential advantages
- Potential disadvantages

IMPORTANT:

Do not invent:

- Crime statistics
- School ratings
- Transport statistics
- Hospital distances
- Walkability scores

If information is unavailable, say:

"Information not provided."

Provide a clear comparison.
"""

    response = llm.invoke(prompt)

    return {
        "location_analysis": response.content
    }


# ============================================================
# 11. AGENT 4 - FINANCIAL ANALYSIS
# ============================================================

def finance_agent(state: RealEstateState):

    prompt = f"""
You are the Financial Analysis Agent.

CLIENT:

Budget:
${state["budget"]:,.0f}

Down Payment:
{state["down_payment"]}%


PROPERTY ANALYSIS:

{state["property_analysis"]}


TASK:

Calculate the following for each selected property:

1. Property price
2. Down payment amount
3. Loan amount
4. Price per square foot
5. Estimated monthly mortgage payment

Mortgage assumptions:

Annual interest rate = 6.5%

Loan period = 30 years

Use:

Monthly rate = annual rate / 12

Number of payments = 30 * 12

Mortgage formula:

M = P * r * (1+r)^n / ((1+r)^n - 1)

where:

P = loan amount
r = monthly interest rate
n = total number of payments

Clearly state that mortgage calculations
are estimates.

Do not include:

- Taxes
- Insurance
- HOA
- Maintenance
- Closing costs

unless provided.

Identify the strongest financial option.
"""

    response = llm.invoke(prompt)

    return {
        "financial_analysis": response.content
    }


# ============================================================
# 12. FINAL DECISION AGENT
# ============================================================

def final_agent(state: RealEstateState):

    prompt = f"""
You are the Final Decision Agent.

You are responsible for producing the final
real estate recommendation.

CLIENT REQUIREMENTS:

Location:
{state["location"]}

Budget:
${state["budget"]:,.0f}

Property Type:
{state["property_type"]}

Minimum Bedrooms:
{state["bedrooms"]}

Down Payment:
{state["down_payment"]}%


PROPERTY ANALYSIS:

{state["property_analysis"]}


MARKET ANALYSIS:

{state["market_analysis"]}


LOCATION ANALYSIS:

{state["location_analysis"]}


FINANCIAL ANALYSIS:

{state["financial_analysis"]}


CREATE THE FINAL REPORT.

Use this structure:

1. Executive Summary

2. Best Property

3. Second Best Property

4. Property Comparison

5. Market Analysis

6. Location Analysis

7. Financial Analysis

8. Advantages

9. Risks

10. Final Recommendation

Be concise but useful.

Do not invent facts.

Mention that financial values are estimates
and property information should be independently
verified.
"""

    response = llm.invoke(prompt)

    return {
        "final_report": response.content
    }


# ============================================================
# 13. CREATE LANGGRAPH
# ============================================================

workflow = StateGraph(RealEstateState)


# Add agents as graph nodes

workflow.add_node(
    "property_agent",
    property_agent
)

workflow.add_node(
    "market_agent",
    market_agent
)

workflow.add_node(
    "location_agent",
    location_agent
)

workflow.add_node(
    "finance_agent",
    finance_agent
)

workflow.add_node(
    "final_agent",
    final_agent
)


# ============================================================
# 14. CONNECT AGENTS
# ============================================================

workflow.add_edge(
    START,
    "property_agent"
)

workflow.add_edge(
    "property_agent",
    "market_agent"
)

workflow.add_edge(
    "market_agent",
    "location_agent"
)

workflow.add_edge(
    "location_agent",
    "finance_agent"
)

workflow.add_edge(
    "finance_agent",
    "final_agent"
)

workflow.add_edge(
    "final_agent",
    END
)


# ============================================================
# 15. COMPILE GRAPH
# ============================================================

graph = workflow.compile()

print("LangGraph compiled successfully.")


# ============================================================
# 16. USER INPUT
# ============================================================

initial_state = {

    "location": "Austin, Texas",

    "budget": 650000,

    "property_type": "Single Family Home",

    "bedrooms": 3,

    "down_payment": 20
}


# ============================================================
# 17. RUN MULTI-AGENT SYSTEM
# ============================================================

print()
print("=" * 70)
print("🏠 REAL ESTATE MULTI-AGENT AI")
print("=" * 70)

print()
print("Starting agents...")
print()


try:

    result = graph.invoke(initial_state)

    print()
    print("=" * 70)
    print("✅ MULTI-AGENT EXECUTION COMPLETED")
    print("=" * 70)

    print()
    print("PROPERTY AGENT")
    print("-" * 70)
    print(result["property_analysis"])

    print()
    print("MARKET AGENT")
    print("-" * 70)
    print(result["market_analysis"])

    print()
    print("LOCATION AGENT")
    print("-" * 70)
    print(result["location_analysis"])

    print()
    print("FINANCIAL AGENT")
    print("-" * 70)
    print(result["financial_analysis"])

    print()
    print("=" * 70)
    print("🏆 FINAL REAL ESTATE REPORT")
    print("=" * 70)
    print()

    print(result["final_report"])

except Exception as e:

    print()
    print("=" * 70)
    print("❌ EXECUTION ERROR")
    print("=" * 70)

    print()
    print("Error type:")
    print(type(e).__name__)

    print()
    print("Error message:")
    print(str(e))

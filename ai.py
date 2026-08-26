# ============================================================
# AI REAL ESTATE MULTI-AGENT SYSTEM
# ============================================================
#
# Framework : CrewAI
# Language  : Python
#
# Agents:
#   1. Property Research Agent
#   2. Market Analysis Agent
#   3. Location Analysis Agent
#   4. Financial Analysis Agent
#
# ============================================================

import os

from crewai import Agent, Task, Crew, Process, LLM


# ============================================================
# 1. API KEY
# ============================================================

# IMPORTANT:
# Replace this with your own OpenAI API key.
#
# DO NOT upload your API key to GitHub.

os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"


# ============================================================
# 2. LLM CONFIGURATION
# ============================================================

llm = LLM(
    model="openai/gpt-4o-mini",
    temperature=0.2
)


# ============================================================
# 3. SAMPLE PROPERTY DATA
# ============================================================
#
# For the first working version, we provide property data
# directly to the agents.
#
# Later we can replace this with:
#   - Serper
#   - Zillow
#   - Realtor APIs
#   - Web search
#   - RAG
#
# ============================================================

properties = """

PROPERTY 1
Address: 123 Oak Street, Austin, Texas
Price: $620,000
Type: Single Family Home
Bedrooms: 4
Bathrooms: 3
Area: 2,350 sq ft
Year Built: 2018
Features: Garage, backyard, modern kitchen, solar panels

PROPERTY 2
Address: 456 Lake View Drive, Austin, Texas
Price: $645,000
Type: Single Family Home
Bedrooms: 4
Bathrooms: 2.5
Area: 2,500 sq ft
Year Built: 2020
Features: Large backyard, garage, smart home system

PROPERTY 3
Address: 789 Green Valley Road, Austin, Texas
Price: $590,000
Type: Single Family Home
Bedrooms: 3
Bathrooms: 2
Area: 2,100 sq ft
Year Built: 2017
Features: Garage, garden, renovated kitchen, quiet neighborhood

PROPERTY 4
Address: 321 Sunset Avenue, Austin, Texas
Price: $680,000
Type: Single Family Home
Bedrooms: 4
Bathrooms: 3
Area: 2,700 sq ft
Year Built: 2021
Features: Swimming pool, garage, modern interior, large backyard

"""


# ============================================================
# 4. USER REQUIREMENTS
# ============================================================

user_requirements = """

Location: Austin, Texas

Maximum Budget: $650,000

Property Type: Single Family Home

Minimum Bedrooms: 3

Down Payment: 20%

Primary Goal:
Find the most suitable property for the client based on
price, property features, estimated financial viability,
and overall suitability.

"""


# ============================================================
# 5. AGENT 1
# PROPERTY RESEARCH AGENT
# ============================================================

property_agent = Agent(

    role="Property Research Specialist",

    goal=(
        "Analyze the available property listings and identify "
        "the properties that best match the client's requirements."
    ),

    backstory=(
        "You are an experienced real estate property researcher. "
        "You carefully compare property price, type, bedrooms, "
        "bathrooms, size, year built, and features. "
        "You eliminate properties that do not satisfy the "
        "client's requirements."
    ),

    llm=llm,

    verbose=True,

    allow_delegation=False
)


# ============================================================
# 6. AGENT 2
# MARKET ANALYSIS AGENT
# ============================================================

market_agent = Agent(

    role="Real Estate Market Analyst",

    goal=(
        "Analyze the selected properties and determine which "
        "properties provide better value for their asking price."
    ),

    backstory=(
        "You are a professional real estate market analyst. "
        "You compare price per square foot, property size, "
        "property age, features, and asking price. "
        "You identify properties that appear expensive, "
        "reasonably priced, or attractive based on the "
        "available information."
    ),

    llm=llm,

    verbose=True,

    allow_delegation=False
)


# ============================================================
# 7. AGENT 3
# LOCATION ANALYSIS AGENT
# ============================================================

location_agent = Agent(

    role="Neighborhood and Location Analyst",

    goal=(
        "Evaluate the location-related advantages and "
        "disadvantages of the selected properties."
    ),

    backstory=(
        "You are a real estate location specialist. "
        "You analyze the available location information, "
        "property characteristics, neighborhood descriptions, "
        "accessibility, lifestyle suitability, and potential "
        "advantages or disadvantages."
    ),

    llm=llm,

    verbose=True,

    allow_delegation=False
)


# ============================================================
# 8. AGENT 4
# FINANCIAL ANALYSIS AGENT
# ============================================================

finance_agent = Agent(

    role="Real Estate Financial Analyst",

    goal=(
        "Perform a financial comparison of the selected "
        "properties and identify the strongest financial option."
    ),

    backstory=(
        "You are an experienced real estate financial analyst. "
        "You calculate estimated down payment, loan amount, "
        "monthly mortgage payment, price per square foot, "
        "and basic investment considerations."
    ),

    llm=llm,

    verbose=True,

    allow_delegation=False
)


# ============================================================
# 9. TASK 1
# PROPERTY SELECTION
# ============================================================

property_task = Task(

    description=f"""
You are the first agent in a multi-agent real estate system.

Analyze the following client requirements:

{user_requirements}

Analyze these available properties:

{properties}

Your job is to:

1. Check every property.
2. Remove properties that exceed the maximum budget.
3. Check whether the property type matches.
4. Check the minimum bedroom requirement.
5. Compare the remaining properties.
6. Select the best 2 or 3 properties.

For every selected property provide:

- Address
- Price
- Bedrooms
- Bathrooms
- Area
- Price per square foot
- Important features
- Why it matches the client

Do not invent property information.
Only use information provided above.

""",

    expected_output="""
A clear shortlist of the best 2 or 3 properties.

For each property include:
address, price, bedrooms, bathrooms,
area, price per square foot, features,
and reason for selection.
""",

    agent=property_agent
)


# ============================================================
# 10. TASK 2
# MARKET ANALYSIS
# ============================================================

market_task = Task(

    description=f"""
You are the second agent.

The first agent has selected suitable properties.

Analyze the previous agent's output together with the
original property data:

{properties}

Perform a comparative market analysis.

For each selected property:

1. Calculate price per square foot if necessary.
2. Compare the properties against each other.
3. Identify which property provides better value.
4. Identify properties that appear overpriced.
5. Identify properties that appear attractive.
6. Explain your reasoning.

Do not claim that you have access to live market data.
Use only the information available in this workflow.

""",

    expected_output="""
A comparative market analysis containing:

- Property comparison
- Price per square foot
- Value assessment
- Advantages
- Disadvantages
- Best-value property
""",

    agent=market_agent
)


# ============================================================
# 11. TASK 3
# LOCATION ANALYSIS
# ============================================================

location_task = Task(

    description=f"""
You are the third agent.

Analyze the properties selected by the previous agents.

Original property information:

{properties}

Evaluate the location and lifestyle suitability using only
the information available in the property data.

For every selected property discuss:

- Neighborhood characteristics
- Lifestyle suitability
- Property features
- Accessibility considerations if provided
- Potential advantages
- Potential disadvantages

If information is unavailable, explicitly say:

"Information not provided."

Do not invent crime statistics, school rankings,
transportation statistics, or other facts.

""",

    expected_output="""
A location comparison for each selected property,
including advantages, disadvantages, and lifestyle suitability.
""",

    agent=location_agent
)


# ============================================================
# 12. TASK 4
# FINANCIAL ANALYSIS
# ============================================================

finance_task = Task(

    description=f"""
You are the fourth agent.

Analyze the properties using:

Client requirements:
{user_requirements}

Property data:
{properties}

Perform a basic financial analysis.

The client has a 20% down payment.

For each selected property calculate:

1. Property price
2. Down payment
3. Estimated loan amount
4. Price per square foot
5. Basic affordability assessment

For the estimated monthly mortgage payment,
assume:

Annual interest rate = 6.5%

Loan term = 30 years

Use the standard mortgage formula:

M = P * [r(1+r)^n] / [(1+r)^n - 1]

where:

P = loan principal
r = monthly interest rate
n = number of monthly payments

Clearly state that this is only an estimate.

Do not include property taxes, insurance,
maintenance, HOA fees, or closing costs unless
they are provided.

Finally identify the financially strongest property.

""",

    expected_output="""
A financial comparison table containing:

Property
Price
Down Payment
Loan Amount
Estimated Monthly Mortgage
Price per Square Foot
Financial Assessment

Also identify the strongest financial option.
""",

    agent=finance_agent
)


# ============================================================
# 13. CREATE THE CREW
# ============================================================

real_estate_crew = Crew(

    agents=[
        property_agent,
        market_agent,
        location_agent,
        finance_agent
    ],

    tasks=[
        property_task,
        market_task,
        location_task,
        finance_task
    ],

    process=Process.sequential,

    verbose=True
)


# ============================================================
# 14. RUN THE MULTI-AGENT SYSTEM
# ============================================================

print()
print("=" * 70)
print("🏠 AI REAL ESTATE MULTI-AGENT SYSTEM")
print("=" * 70)
print()

print("Starting multi-agent analysis...")
print()

try:

    result = real_estate_crew.kickoff()

    print()
    print("=" * 70)
    print("🏆 FINAL REAL ESTATE REPORT")
    print("=" * 70)
    print()

    print(result)

except Exception as error:

    print()
    print("=" * 70)
    print("❌ ERROR")
    print("=" * 70)
    print()

    print("Error Type:")
    print(type(error).__name__)

    print()
    print("Error Message:")
    print(error)

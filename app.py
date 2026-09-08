import os
import io
import json
import traceback
from typing import TypedDict, List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yfinance as yf

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

# ==========================================
# 1. INITIALIZATION & ENV CONFIG
# ==========================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY environment variable is not set.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=GEMINI_API_KEY,
    temperature=0.2
)

# ==========================================
# 2. STATE DEFINITION
# ==========================================
class FinancialState(TypedDict):
    ticker: str
    target_concern: Optional[str]
    financial_data: Optional[Dict[str, Any]]
    analysis_report: Optional[str]
    risk_assessment: Optional[str]
    final_verdict: Optional[str]

# ==========================================
# 3. TOOLS
# ==========================================
@tool
def fetch_ticker_fundamentals(ticker: str) -> Dict[str, Any]:
    """Fetches fundamental financial metrics and recent ratios using Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        extracted = {
            "symbol": ticker.upper(),
            "shortName": info.get("shortName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "currentPrice": info.get("currentPrice") or info.get("regularMarketPrice"),
            "marketCap": info.get("marketCap"),
            "peRatio": info.get("trailingPE"),
            "forwardPE": info.get("forwardPE"),
            "pegRatio": info.get("pegRatio"),
            "profitMargins": info.get("profitMargins"),
            "operatingMargins": info.get("operatingMargins"),
            "debtToEquity": info.get("debtToEquity"),
            "freeCashflow": info.get("freeCashflow"),
            "recommendationKey": info.get("recommendationKey", "N/A"),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
        }
        return extracted
    except Exception as e:
        return {"error": f"Failed to fetch data for {ticker}: {str(e)}"}

# ==========================================
# 4. GRAPH NODES
# ==========================================
def market_data_fetcher_node(state: FinancialState):
    ticker = state["ticker"]
    metrics = fetch_ticker_fundamentals.invoke(ticker)
    return {"financial_data": metrics}

def financial_analyst_node(state: FinancialState):
    ticker = state["ticker"]
    data = state.get("financial_data", {})
    concern = state.get("target_concern") or "General valuation and fundamentals"

    prompt = f"""
    You are a Senior Wall Street Equity Research Analyst.
    Evaluate the following financial fundamentals for {ticker}:
    Data: {json.dumps(data, indent=2)}

    Primary Investor Concern/Focus: {concern}

    Provide a concise fundamental analysis report covering:
    1. Valuation (P/E, Forward P/E, PEG)
    2. Capital Structure & Health (Debt to Equity, Free Cash Flow)
    3. Operational Efficiency (Margins)
    Format clearly in Markdown.
    """
    response = llm.invoke(prompt)
    report_text = response.content if hasattr(response, "content") else str(response)
    return {"analysis_report": report_text}

def risk_officer_node(state: FinancialState):
    data = state.get("financial_data", {})
    analysis = state.get("analysis_report", "")

    prompt = f"""
    You are a Chief Risk Officer (CRO). Review this financial analysis and metrics:
    Metrics: {json.dumps(data, indent=2)}
    Analyst Report: {analysis}

    Identify:
    1. Top 3 downside risk factors (liquidity, leverage, industry headwinds, valuation multiple contraction).
    2. Stress points or red flags.
    Return a structured risk assessment in Markdown.
    """
    response = llm.invoke(prompt)
    risk_text = response.content if hasattr(response, "content") else str(response)
    return {"risk_assessment": risk_text}

def portfolio_manager_node(state: FinancialState):
    analysis = state.get("analysis_report", "")
    risks = state.get("risk_assessment", "")

    prompt = f"""
    You are the Lead Portfolio Manager making the final capital allocation decision.
    Based on:
    - Fundamental Analysis: {analysis}
    - Risk Audit: {risks}

    Provide:
    1. Rating: [BULLISH / NEUTRAL / BEARISH]
    2. Conviction Level: [LOW / MEDIUM / HIGH]
    3. Final Allocation Verdict (2-3 sentences explaining rationale).
    """
    response = llm.invoke(prompt)
    verdict_text = response.content if hasattr(response, "content") else str(response)
    return {"final_verdict": verdict_text}

# ==========================================
# 5. GRAPH CONSTRUCTION
# ==========================================
workflow = StateGraph(FinancialState)

workflow.add_node("data_fetcher", market_data_fetcher_node)
workflow.add_node("equity_analyst", financial_analyst_node)
workflow.add_node("risk_officer", risk_officer_node)
workflow.add_node("portfolio_manager", portfolio_manager_node)

workflow.add_edge(START, "data_fetcher")
workflow.add_edge("data_fetcher", "equity_analyst")
workflow.add_edge("equity_analyst", "risk_officer")
workflow.add_edge("risk_officer", "portfolio_manager")
workflow.add_edge("portfolio_manager", END)

app_graph = workflow.compile()

# ==========================================
# 6. FASTAPI WEB SERVER (FOR RENDER)
# ==========================================
api = FastAPI(
    title="Financial Analyzer Agent API",
    description="Multi-agent financial assessment system powered by LangGraph and Gemini",
    version="1.0.0"
)

class AnalyzeRequest(BaseModel):
    ticker: str
    concern: Optional[str] = "Evaluate valuation and short-to-medium term risk."

@api.get("/")
def health_check():
    return {"status": "online", "message": "Financial Analyzer Agent is active."}

@api.post("/analyze")
def run_analysis(request: AnalyzeRequest):
    if not request.ticker:
        raise HTTPException(status_code=400, detail="Ticker symbol must be provided.")
    
    initial_state: FinancialState = {
        "ticker": request.ticker.strip().upper(),
        "target_concern": request.concern,
        "financial_data": None,
        "analysis_report": None,
        "risk_assessment": None,
        "final_verdict": None
    }

    try:
        final_state = app_graph.invoke(initial_state)
        return {
            "ticker": final_state["ticker"],
            "raw_fundamentals": final_state["financial_data"],
            "equity_analysis": final_state["analysis_report"],
            "risk_audit": final_state["risk_assessment"],
            "portfolio_verdict": final_state["final_verdict"]
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:api", host="0.0.0.0", port=port, reload=False)

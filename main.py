import os
import json
import traceback
from typing import TypedDict, List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import yfinance as yf

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

# ==========================================
# 1. INITIALIZATION
# ==========================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.3
)

# In-memory session store for chat history
session_histories: Dict[str, List[BaseMessage]] = {}

# ==========================================
# 2. STATE & TOOLS
# ==========================================
class FinancialChatState(TypedDict):
    messages: List[BaseMessage]
    ticker: Optional[str]
    financial_data: Optional[Dict[str, Any]]
    final_response: Optional[str]

@tool
def fetch_ticker_data(ticker: str) -> Dict[str, Any]:
    """Fetches key valuation, margin, and market metrics for a stock ticker."""
    try:
        stock = yf.Ticker(ticker.strip().upper())
        info = stock.info
        return {
            "symbol": ticker.upper(),
            "name": info.get("shortName", "N/A"),
            "price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "market_cap": info.get("marketCap"),
            "debt_to_equity": info.get("debtToEquity"),
            "free_cash_flow": info.get("freeCashflow"),
            "profit_margins": info.get("profitMargins"),
            "target_high": info.get("targetHighPrice"),
            "recommendation": info.get("recommendationKey", "N/A"),
        }
    except Exception as e:
        return {"error": f"Failed fetching data for {ticker}: {str(e)}"}

# ==========================================
# 3. GRAPH NODES
# ==========================================
def extract_intent_node(state: FinancialChatState):
    """Extracts whether a ticker symbol is mentioned in the user conversation."""
    last_user_msg = [m.content for m in state["messages"] if isinstance(m, HumanMessage)][-1]
    
    extract_prompt = f"""
    Given this user message: "{last_user_msg}"
    Determine if the user is asking about a specific publicly traded company/stock.
    If yes, return ONLY the ticker symbol (e.g. AAPL, TSLA, NVDA, RELIANCE.NS).
    If no specific stock is mentioned or it's a general question/greeting, return 'NONE'.
    Do not output any markdown or punctuation, just the ticker or NONE.
    """
    res = llm.invoke(extract_prompt).content.strip().upper()
    ticker = None if "NONE" in res or len(res) > 12 else res
    return {"ticker": ticker}

def data_retrieval_node(state: FinancialChatState):
    """Fetches real-time financial metrics if a ticker was detected."""
    ticker = state.get("ticker")
    if ticker:
        data = fetch_ticker_data.invoke(ticker)
        return {"financial_data": data}
    return {"financial_data": None}

def analyst_chat_node(state: FinancialChatState):
    """Generates the conversational financial answer."""
    data = state.get("financial_data")
    ticker = state.get("ticker")

    system_instruction = (
        "You are an expert AI Financial Analyst and Portfolio Assistant. "
        "Engage conversationally, clearly explaining fundamentals (PE ratio, debt, margins, cash flows), "
        "highlighting key financial risks, and answering user follow-up questions accurately. "
        "Keep answers crisp, insightful, and formatted cleanly in Markdown."
    )

    if data and "error" not in data:
        context = f"\n[Real-time fundamental metrics for {ticker}: {json.dumps(data)}]"
    else:
        context = ""

    augmented_messages = [SystemMessage(content=system_instruction + context)] + state["messages"]
    bot_reply = llm.invoke(augmented_messages)
    reply_text = bot_reply.content if hasattr(bot_reply, "content") else str(bot_reply)

    return {
        "final_response": reply_text,
        "messages": state["messages"] + [AIMessage(content=reply_text)]
    }

# ==========================================
# 4. WORKFLOW GRAPH
# ==========================================
workflow = StateGraph(FinancialChatState)
workflow.add_node("extract_intent", extract_intent_node)
workflow.add_node("data_retrieval", data_retrieval_node)
workflow.add_node("analyst_chat", analyst_chat_node)

workflow.add_edge(START, "extract_intent")
workflow.add_edge("extract_intent", "data_retrieval")
workflow.add_edge("data_retrieval", "analyst_chat")
workflow.add_edge("analyst_chat", END)

chat_graph = workflow.compile()

# ==========================================
# 5. FASTAPI APP & CHAT UI
# ==========================================
app = FastAPI(title="Financial Analyst Chatbot")

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default_user"

@app.post("/chat")
def chat_endpoint(payload: ChatRequest):
    sid = payload.session_id
    if sid not in session_histories:
        session_histories[sid] = []

    # Append current user message
    session_histories[sid].append(HumanMessage(content=payload.message))

    # Keep conversation history window manageable
    recent_messages = session_histories[sid][-8:]

    initial_state: FinancialChatState = {
        "messages": recent_messages,
        "ticker": None,
        "financial_data": None,
        "final_response": None
    }

    try:
        output = chat_graph.invoke(initial_state)
        # Persist assistant reply
        session_histories[sid].append(AIMessage(content=output["final_response"]))
        return {
            "reply": output["final_response"],
            "ticker_analyzed": output.get("ticker"),
            "data_snapshot": output.get("financial_data")
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
def serve_chat_ui():
    """Serves a clean, responsive chat web page directly from the root domain."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Financial Analyst Agent</title>
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; height: 100vh; display: flex; flex-direction: column; }
            header { background: #1e293b; padding: 16px 24px; border-bottom: 1px solid #334155; font-size: 1.1rem; font-weight: 600; }
            #chat-window { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 16px; }
            .msg { max-width: 80%; padding: 14px 18px; border-radius: 12px; line-height: 1.6; word-break: break-word; font-size: 0.95rem; }
            .user { align-self: flex-end; background: #2563eb; color: #ffffff; border-bottom-right-radius: 2px; }
            .bot { align-self: flex-start; background: #1e293b; border: 1px solid #334155; border-bottom-left-radius: 2px; }
            .bot p { margin-bottom: 8px; }
            .bot ul, .bot ol { margin-left: 20px; margin-bottom: 8px; }
            #input-container { padding: 16px 24px; background: #1e293b; border-top: 1px solid #334155; display: flex; gap: 12px; }
            input { flex: 1; padding: 12px 16px; background: #0f172a; border: 1px solid #475569; border-radius: 8px; color: #fff; outline: none; font-size: 1rem; }
            input:focus { border-color: #3b82f6; }
            button { padding: 12px 24px; background: #2563eb; border: none; border-radius: 8px; color: #fff; font-weight: 600; cursor: pointer; transition: 0.2s; }
            button:hover { background: #1d4ed8; }
            button:disabled { background: #475569; cursor: not-allowed; }
        </style>
    </head>
    <body>
        <header>📈 Financial Analyst Agent</header>
        <div id="chat-window">
            <div class="msg bot">Hello! I am your AI Financial Analyst. Ask me about any company fundamentals, valuation multiples, risk assessments, or stock tickers (e.g., <em>"Analyze NVDA's margins"</em> or <em>"Is AAPL valuation stretched?"</em>).</div>
        </div>
        <form id="input-container" onsubmit="sendMessage(event)">
            <input id="prompt" type="text" placeholder="Ask financial question or ticker..." autocomplete="off" />
            <button id="send-btn" type="submit">Send</button>
        </form>

        <script>
            const sessionId = "session_" + Math.random().toString(36).substring(7);
            const chatWin = document.getElementById('chat-window');
            const promptInput = document.getElementById('prompt');
            const sendBtn = document.getElementById('send-btn');

            async function sendMessage(e) {
                e.preventDefault();
                const text = promptInput.value.trim();
                if (!text) return;

                // Add User Message
                appendMessage(text, 'user');
                promptInput.value = '';
                promptInput.disabled = true;
                sendBtn.disabled = true;

                // Loading Indicator
                const loader = appendMessage("Analyzing market data...", 'bot');

                try {
                    const res = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: text, session_id: sessionId })
                    });
                    const data = await res.json();
                    loader.innerHTML = marked.parse(data.reply);
                } catch (err) {
                    loader.innerText = "Error: Failed to reach the analysis agent.";
                } finally {
                    promptInput.disabled = false;
                    sendBtn.disabled = false;
                    promptInput.focus();
                    chatWin.scrollTop = chatWin.scrollHeight;
                }
            }

            function appendMessage(content, role) {
                const div = document.createElement('div');
                div.className = 'msg ' + role;
                if (role === 'user') {
                    div.innerText = content;
                } else {
                    div.innerHTML = marked.parse(content);
                }
                chatWin.appendChild(div);
                chatWin.scrollTop = chatWin.scrollHeight;
                return div;
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

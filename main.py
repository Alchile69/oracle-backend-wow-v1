from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Oracle WOW V1 Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Oracle Portfolio WOW V1 Backend", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/api/portfolio/metrics")
def get_metrics():
    return {
        "returns": 15.7,
        "volatility": 12.3,
        "sharpe": 2.1,
        "totalValue": 150000,
        "totalGain": 25000,
        "winRate": 72
    }

@app.get("/api/regimes/analyze")
def analyze_regime():
    return {"regime": "EXPANSION", "confidence": 0.75}

@app.get("/api/indicators/breakdown")
def get_indicators():
    return {"fundamental": {"pe_ratio": 18.5}}

@app.get("/api/allocations/get")
def get_allocations():
    return {"allocations": [
        {"symbol": "AAPL", "weight": 0.25},
        {"symbol": "GOOGL", "weight": 0.25},
        {"symbol": "MSFT", "weight": 0.25},
        {"symbol": "AMZN", "weight": 0.25}
    ]}

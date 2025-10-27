"""
Futures Metrics Wrappers
-------------------------
Python wrapper untuk mengambil data Funding Rate dan Open Interest
dari 3 exchange: Binance, Bybit, dan OKX.

Author: ChatGPT (GPT-5)
"""

import requests
import time

# ===================== BINANCE =====================

def get_funding_rate_binance(symbol: str, limit: int = 24):
    """Ambil funding rate historis dari Binance Futures (USDT-M)."""
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    params = {"symbol": symbol, "limit": limit}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return None
    return [
        {
            "time": d["fundingTime"],
            "fundingRate": float(d["fundingRate"]),
            "symbol": d["symbol"]
        }
        for d in data
    ]


def get_open_interest_binance(symbol: str, period: str = "1h", limit: int = 50):
    """Ambil Open Interest historis dari Binance Futures (USDT-M)."""
    url = "https://fapi.binance.com/futures/data/openInterestHist"
    params = {"symbol": symbol, "period": period, "limit": limit}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return None
    return [
        {
            "time": d["timestamp"],
            "openInterest": float(d["sumOpenInterest"]),
            "value": float(d["sumOpenInterestValue"]),
        }
        for d in data
    ]


# ===================== BYBIT =====================

def get_funding_rate_bybit(symbol: str, limit: int = 20):
    """Ambil funding rate historis dari Bybit Futures (linear)."""
    url = "https://api.bybit.com/v5/market/funding/history"
    params = {"category": "linear", "symbol": symbol, "limit": limit}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data.get("retCode") != 0:
        return None
    return [
        {
            "time": d["fundingRateTimestamp"],
            "fundingRate": float(d["fundingRate"]),
            "symbol": symbol,
        }
        for d in data["result"]["list"]
    ]


def get_open_interest_bybit(symbol: str, interval: str = "1h", limit: int = 50):
    """Ambil Open Interest historis dari Bybit Futures."""
    url = "https://api.bybit.com/v5/market/open-interest"
    params = {"category": "linear", "symbol": symbol, "intervalTime": interval, "limit": limit}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data.get("retCode") != 0:
        return None
    return [
        {
            "time": d["timestamp"],
            "openInterest": float(d["openInterest"]),
            "symbol": symbol,
        }
        for d in data["result"]["list"]
    ]


# ===================== OKX =====================

def get_funding_rate_okx(instId: str, limit: int = 20):
    """Ambil funding rate historis dari OKX untuk kontrak swap (e.g., BTC-USDT-SWAP)."""
    url = "https://www.okx.com/api/v5/public/funding-rate-history"
    params = {"instId": instId, "limit": limit}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    if not data:
        return None
    return [
        {
            "time": d["fundingTime"],
            "fundingRate": float(d["fundingRate"]),
            "symbol": instId,
        }
        for d in data
    ]


def get_open_interest_okx(instId: str):
    """Ambil snapshot open interest dari OKX untuk kontrak swap."""
    url = "https://www.okx.com/api/v5/public/open-interest"
    params = {"instId": instId}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    if not data:
        return None
    return {
        "instId": instId,
        "oi": float(data[0]["oi"]),
        "oiCcy": float(data[0]["oiCcy"]),
        "ts": data[0]["ts"],
    }


# ===================== SIMPLE TESTS =====================

if __name__ == "__main__":
    print("🧪 Testing Futures Metrics Wrappers...\n")

    try:
        print("➡️ Binance Funding Rate BTCUSDT:")
        data = get_funding_rate_binance("BTCUSDT", limit=5)
        print(data[:2])
    except Exception as e:
        print("Binance Funding Rate error:", e)

    time.sleep(1)

    try:
        print("\n➡️ Bybit Funding Rate BTCUSDT:")
        data = get_funding_rate_bybit("BTCUSDT", limit=5)
        print(data[:2])
    except Exception as e:
        print("Bybit Funding Rate error:", e)

    time.sleep(1)

    try:
        print("\n➡️ OKX Funding Rate BTC-USDT-SWAP:")
        data = get_funding_rate_okx("BTC-USDT-SWAP", limit=5)
        print(data[:2])
    except Exception as e:
        print("OKX Funding Rate error:", e)

    print("\n✅ Test selesai.")

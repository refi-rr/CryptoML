"""
liquidation_wrappers.py
Author: Refito + ChatGPT
Description:
    Multi-source liquidation data wrapper with debug logging for Binance, CoinGlass, and fallback.
"""

import requests
import pandas as pd
import streamlit as st
from datetime import datetime

# ===============================
# 🔧 Utility: Debug Logger
# ===============================
def debug_log(message, enabled=True):
    if enabled:
        print(f"[DEBUG] {datetime.now().strftime('%H:%M:%S')} | {message}")


# ===============================
# 🟢 Binance Futures Liquidation API (Public, no key)
# ===============================
@st.cache_data(ttl=60)
def get_liquidation_binance(symbol: str, limit: int = 100, debug=True) -> pd.DataFrame | None:
    """
    Fetch liquidation data from Binance Futures (public API)
    https://binance-docs.github.io/apidocs/futures/en/#all-force-orders-market-data
    """
    try:
        url = f"https://fapi.binance.com/fapi/v1/allForceOrders"
        params = {"symbol": symbol.upper(), "limit": limit}
        debug_log(f"Fetching Binance liquidation for {symbol} ...", debug)
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            debug_log(f"No data returned from Binance for {symbol}.", debug)
            return None

        df = pd.DataFrame(data)
        if df.empty:
            return None

        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df.rename(columns={"price": "liq_price", "qty": "liq_qty"}, inplace=True)
        df["exchange"] = "binance"
        df["side"] = df["side"].str.upper()
        df["symbol"] = symbol.upper()

        debug_log(f"✅ Binance returned {len(df)} liquidation rows for {symbol}", debug)
        return df[["time", "symbol", "side", "liq_price", "liq_qty", "exchange"]]

    except Exception as e:
        debug_log(f"❌ Binance API error: {e}", debug)
        return None


# ===============================
# 🟣 CoinGlass Liquidation API (Optional API key)
# ===============================
@st.cache_data(ttl=120)
def get_liquidation_coinglass(symbol: str, api_key: str = None, debug=True) -> pd.DataFrame | None:
    """
    Fetch liquidation data from CoinGlass (requires API key for stable access)
    Endpoint: /public/v2/liquidation_history
    """
    try:
        base_symbol = symbol.replace("USDT", "")
        url = "https://open-api.coinglass.com/public/v2/liquidation_history"
        headers = {"accept": "application/json"}
        if api_key:
            headers["coinglassSecret"] = api_key
        params = {"symbol": base_symbol, "time_type": "1h"}

        debug_log(f"Fetching CoinGlass liquidation for {base_symbol} ...", debug)
        resp = requests.get(url, headers=headers, params=params, timeout=6)
        resp.raise_for_status()
        j = resp.json()

        if not j.get("success"):
            debug_log(f"❌ CoinGlass returned no success for {base_symbol}", debug)
            return None

        records = j.get("data", [])
        if not records:
            return None

        df = pd.DataFrame(records)
        df.rename(columns={"buyVolUsd": "buy_vol", "sellVolUsd": "sell_vol", "volUsd": "total_vol"}, inplace=True)
        df["time"] = pd.to_datetime(df["createTime"], unit="ms")
        df["symbol"] = symbol.upper()
        df["exchange"] = "coinglass"
        df["side"] = df.apply(lambda x: "BUY" if x["buy_vol"] > x["sell_vol"] else "SELL", axis=1)
        df["liq_price"] = None
        df["liq_qty"] = df["total_vol"]

        debug_log(f"✅ CoinGlass returned {len(df)} liquidation rows for {base_symbol}", debug)
        return df[["time", "symbol", "side", "liq_price", "liq_qty", "exchange"]]

    except Exception as e:
        debug_log(f"❌ CoinGlass API error: {e}", debug)
        return None


# ===============================
# ⚙️ Combined Wrapper
# ===============================
@st.cache_data(ttl=60)
def get_liquidation_data(symbol: str, exchange: str = "auto", limit: int = 100, api_key: str = None, debug=True):
    """
    Main wrapper to fetch liquidation data from available sources.
    Priority:
        1. Binance (default)
        2. CoinGlass (if Binance fails)
    """
    symbol = symbol.upper()
    result = None

    if exchange.lower() in ["binance", "auto"]:
        result = get_liquidation_binance(symbol, limit=limit, debug=debug)
        if result is not None and not result.empty:
            return result

    if exchange.lower() in ["coinglass", "auto"]:
        result = get_liquidation_coinglass(symbol, api_key=api_key, debug=debug)
        if result is not None and not result.empty:
            return result

    debug_log(f"⚠️ No liquidation data found for {symbol} (all sources)", debug)
    return None
# ✅ ULTIMATE MASTER OPTION ALGO WITH INSTITUTIONAL LAYERS + ALL INDICES PARALLEL
# ✅ NIFTY + BANKNIFTY + SENSEX + FINNIFTY + MIDCPNIFTY + EICHERMOT + TRENT + RELIANCE
# ✅ Institutional Flow + OI/Delta Layers + Liquidity Hunting + Multi-timeframe + Telegram
# ✅ All indices running simultaneously in parallel threads
# ✅ ADDED: Opening-range institutional play, Gamma Squeeze (expiry actionable), Delta imbalance,
# ✅ Smart-Money divergence, Stop-hunt detector, Institutional continuation, expiry bias, momentum amplifier

import os
import time
import requests
import pandas as pd
import yfinance as yf
import ta
import warnings
import pyotp
import math
from datetime import datetime, time as dtime, timedelta
from SmartApi.smartConnect import SmartConnect
import threading
import numpy as np
import json

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
OPENING_PLAY_ENABLED = True           # Enable/disable the pure opening-level institutional play
OPENING_START = dtime(9,15)
OPENING_END = dtime(9,45)

EXPIRY_ACTIONABLE = True              # If True: expiry gamma layer can ACTIVATE trade signals (with extra checks)
EXPIRY_INFO_ONLY = False              # If True, expiry layer only sends info messages (won't trigger trades)
EXPIRY_RELAX_FACTOR = 0.7            # How much to relax confirmation requirements on expiry (0-1, lower -> more relaxation)

GAMMA_VOL_SPIKE_THRESHOLD = 1.8      # multiplier for volume to consider gamma-like event
DELTA_OI_RATIO = 1.5                 # OI CE/PE dominance ratio threshold
MOMENTUM_VOL_AMPLIFIER = 1.3         # threshold for amplifying confidence when volatility & momentum high

# SAFE FETCH settings
YF_RETRIES = 3
YF_DELAY = 2
YF_TIMEOUT = 15

# LTP CACHE FILE
LTP_CACHE_FILE = "ltp_cache.json"
LTP_CACHE_LOCK = threading.Lock()

# ----------------------------------------

# --------- ANGEL ONE LOGIN ---------
# REPLACE with env or plaintext placeholders. For safety we use placeholders here.
API_KEY = os.getenv("API_KEY") or "<REDACTED_API_KEY>"
CLIENT_CODE = os.getenv("CLIENT_CODE") or "<REDACTED_CLIENT_CODE>"
PASSWORD = os.getenv("PASSWORD") or "<REDACTED_PASSWORD>"
TOTP_SECRET = os.getenv("TOTP_SECRET") or "<REDACTED_TOTP_SECRET>"

try:
    TOTP = pyotp.TOTP(TOTP_SECRET).now() if TOTP_SECRET and "<REDACTED" not in TOTP_SECRET else None
except Exception:
    TOTP = None

client = None
session = None
feedToken = None
try:
    if API_KEY and "<REDACTED" not in API_KEY:
        client = SmartConnect(api_key=API_KEY)
        if CLIENT_CODE and PASSWORD and TOTP:
            try:
                session = client.generateSession(CLIENT_CODE, PASSWORD, TOTP)
                feedToken = client.getfeedToken()
            except Exception as e:
                print("Angel One session init error (expected in dev):", e)
                client = client  # keep object but session may be None
except Exception as e:
    print("Angel One init error (expected in dev):", e)
    client = None
    session = None
    feedToken = None

# --------- TELEGRAM ---------
BOT_TOKEN = os.getenv("BOT_TOKEN") or "<REDACTED_TELEGRAM_TOKEN>"
CHAT_ID = os.getenv("CHAT_ID") or "<REDACTED_CHAT_ID>"

# --- Added: guards to avoid repeating the same Telegram message many times
STARTED_SENT = False
STOP_SENT = False

# small rate limiter for telegram to avoid flooding
_TELEGRAM_LOCK = threading.Lock()
_last_telegram_time = 0
_TELEGRAM_MIN_INTERVAL = 0.3  # seconds

def send_telegram(msg, reply_to=None, force=False):
    global _last_telegram_time
    try:
        if not BOT_TOKEN or not CHAT_ID or "<REDACTED" in BOT_TOKEN:
            # in dev if not configured just print
            print("Telegram (not configured) ->", msg)
            return None
        # rate-limit minor
        with _TELEGRAM_LOCK:
            now = time.time()
            if not force and now - _last_telegram_time < _TELEGRAM_MIN_INTERVAL:
                # small sleep to avoid spamming
                time.sleep(_TELEGRAM_MIN_INTERVAL)
            _last_telegram_time = time.time()
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": msg}
        if reply_to:
            payload["reply_to_message_id"] = reply_to
        r = requests.post(url, data=payload, timeout=6)
        try:
            j = r.json()
            return j.get("result", {}).get("message_id")
        except Exception:
            return None
    except Exception as e:
        print("send_telegram error:", e)
        return None

# --------- MARKET HOURS ---------
def is_market_open():
    utc_now = datetime.utcnow()
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    current_time_ist = ist_now.time()
    return dtime(9,15) <= current_time_ist <= dtime(15,30)

def should_stop_trading():
    utc_now = datetime.utcnow()
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    current_time_ist = ist_now.time()
    return current_time_ist >= dtime(15,30)

# --------- STRIKE ROUNDING FOR ALL INDICES ---------
def round_strike(index, price):
    try:
        if price is None: return None
        # handle pandas NaN or numpy NaN
        if isinstance(price, float) and math.isnan(price): return None
        price = float(price)
        if price <= 0: return None
        if index == "NIFTY": return int(round(price/50.0)*50)
        if index == "BANKNIFTY": return int(round(price/100.0)*100)
        if index == "SENSEX": return int(round(price/100.0)*100)
        if index == "FINNIFTY": return int(round(price/50.0)*50)
        if index == "MIDCPNIFTY": return int(round(price/25.0)*25)
        if index == "EICHERMOT": return int(round(price/50.0)*50)
        if index == "TRENT": return int(round(price/100.0)*100)
        if index == "RELIANCE": return int(round(price/10.0)*10)
        return int(round(price/50.0)*50)
    except Exception:
        return None

# --------- ENSURE SERIES ---------
def ensure_series(data):
    return data.iloc[:,0] if isinstance(data, pd.DataFrame) else data.squeeze()

# --------- SAFE YFINANCE FETCH (retries + logging) ---------
def safe_yf_download(symbol, period="1d", interval="5m", retries=YF_RETRIES, delay=YF_DELAY, timeout=YF_TIMEOUT):
    last_exc = None
    for attempt in range(retries):
        try:
            # yfinance doesn't accept timeout param directly; wrap requests via session if needed. keep simple
            df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
            if df is not None and not df.empty:
                return df
            last_exc = Exception("empty-data")
        except Exception as e:
            last_exc = e
            time.sleep(delay * (1 + attempt*0.5))
    # final fallback None
    print(f"safe_yf_download: failed for {symbol} after {retries} attempts: {last_exc}")
    return None

# --------- FETCH INDEX DATA FOR ALL INDICES (uses safe fetch + caching last ltp) ---------
def fetch_index_data(index, interval="5m", period="2d"):
    symbol_map = {
        "NIFTY": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "SENSEX": "^BSESN",
        "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
        "MIDCPNIFTY": "NIFTY_MID_SELECT.NS",
        "EICHERMOT": "EICHERMOT.NS",
        "TRENT": "TRENT.NS",
        "RELIANCE": "RELIANCE.NS"
    }
    sym = symbol_map.get(index)
    if not sym:
        return None
    df = safe_yf_download(sym, period=period, interval=interval)
    if df is None:
        # try one last time with smaller period
        df = safe_yf_download(sym, period="1d", interval=interval)
    if df is None:
        return None
    # store last close into LTP cache
    try:
        last_close = ensure_series(df['Close']).iloc[-1]
        with LTP_CACHE_LOCK:
            try:
                cache = {}
                if os.path.exists(LTP_CACHE_FILE):
                    with open(LTP_CACHE_FILE, "r") as f:
                        cache = json.load(f)
                cache[index] = float(last_close)
                with open(LTP_CACHE_FILE, "w") as f:
                    json.dump(cache, f)
            except Exception:
                pass
    except Exception:
        pass
    return df

# --------- LOAD TOKEN MAP (cached) ---------
_TOKEN_MAP_LOCK = threading.Lock()
def load_token_map():
    try:
        url="https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        df=pd.DataFrame(requests.get(url,timeout=10).json())
        df.columns=[c.lower() for c in df.columns]
        df=df[df['exch_seg'].str.upper().isin(["NFO", "BFO"])]
        df['symbol']=df['symbol'].str.upper()
        return df.set_index('symbol')['token'].to_dict()
    except Exception as e:
        print("load_token_map error:", e)
        return {}

token_map = {}
try:
    token_map = load_token_map()
except:
    token_map = {}

# --------- LTP CACHE helpers ---------
def read_ltp_cache(index):
    try:
        with LTP_CACHE_LOCK:
            if os.path.exists(LTP_CACHE_FILE):
                with open(LTP_CACHE_FILE, "r") as f:
                    cache = json.load(f)
                    return cache.get(index)
    except Exception:
        return None
    return None

def write_ltp_cache(index, val):
    try:
        with LTP_CACHE_LOCK:
            cache = {}
            if os.path.exists(LTP_CACHE_FILE):
                with open(LTP_CACHE_FILE, "r") as f:
                    cache = json.load(f)
            cache[index] = float(val)
            with open(LTP_CACHE_FILE, "w") as f:
                json.dump(cache, f)
    except Exception:
        pass

# --------- SAFE LTP FETCH (option price) ---------
def fetch_option_price(symbol, retries=3, delay=2):
    """
    Tries to fetch option LTP from Angel using token_map if available and client is active.
    If client not configured or token not found, returns None.
    """
    try:
        if not symbol:
            return None
        sym_upper = symbol.upper()
        token = token_map.get(sym_upper)
        # refresh token_map if missing token
        if token is None:
            with _TOKEN_MAP_LOCK:
                # try reload once
                try:
                    new_map = load_token_map()
                    token_map.update(new_map)
                    token = token_map.get(sym_upper)
                except Exception:
                    token = None
        if not token:
            # no token found
            return None
        # if client configured, use it
        if client:
            for attempt in range(retries):
                try:
                    exchange = "BFO" if "SENSEX" in sym_upper else "NFO"
                    data = client.ltpData(exchange, symbol, token)
                    if data and 'data' in data and data['data'] and 'ltp' in data['data']:
                        val = data['data']['ltp']
                        if val is not None:
                            return float(val)
                except Exception as e:
                    # network or client error -> wait and retry
                    time.sleep(delay * (1 + attempt*0.5))
            return None
        else:
            # client not configured: cannot fetch option LTP reliably
            return None
    except Exception:
        return None

# --------- DETECT LIQUIDITY ZONE ---------
def detect_liquidity_zone(df, lookback=20):
    high_series = ensure_series(df['High']).dropna()
    low_series = ensure_series(df['Low']).dropna()
    try:
        high_pool = float(high_series.rolling(lookback).max().iloc[-2]) if len(high_series)>lookback else float(high_series.max()) if len(high_series)>0 else float('nan')
    except Exception:
        high_pool = float(high_series.max()) if len(high_series)>0 else float('nan')
    try:
        low_pool = float(low_series.rolling(lookback).min().iloc[-2]) if len(low_series)>lookback else float(low_series.min()) if len(low_series)>0 else float('nan')
    except Exception:
        low_pool = float(low_series.min()) if len(low_series)>0 else float('nan')
    if math.isnan(high_pool) and len(high_series)>0: high_pool = float(high_series.max())
    if math.isnan(low_pool) and len(low_series)>0: low_pool = float(low_series.min())
    # safety rounding and None handling
    try:
        high_pool = None if math.isnan(high_pool) else round(high_pool,0)
    except:
        high_pool = None
    try:
        low_pool = None if math.isnan(low_pool) else round(low_pool,0)
    except:
        low_pool = None
    return high_pool, low_pool

# --------- INSTITUTIONAL LIQUIDITY HUNT (robust) ---------
def institutional_liquidity_hunt(index, df):
    prev_high = None; prev_low = None
    try:
        prev_high_val = ensure_series(df['High']).iloc[-2]
        prev_low_val = ensure_series(df['Low']).iloc[-2]
        prev_high = float(prev_high_val) if not (isinstance(prev_high_val,float) and math.isnan(prev_high_val)) else None
        prev_low = float(prev_low_val) if not (isinstance(prev_low_val,float) and math.isnan(prev_low_val)) else None
    except Exception:
        prev_high = None; prev_low = None
    high_zone, low_zone = detect_liquidity_zone(df, lookback=15)
    last_close_val = None
    try:
        lc = ensure_series(df['Close']).iloc[-1]
        last_close_val = None if (isinstance(lc,float) and math.isnan(lc)) else float(lc)
    except Exception:
        last_close_val = None
    if last_close_val is None:
        highest_ce_oi_strike = None; highest_pe_oi_strike = None
    else:
        highest_ce_oi_strike = round_strike(index, last_close_val + 50)
        highest_pe_oi_strike = round_strike(index, last_close_val - 50)
    bull_liquidity = []
    if prev_low is not None: bull_liquidity.append(prev_low)
    if low_zone is not None: bull_liquidity.append(low_zone)
    if highest_pe_oi_strike is not None: bull_liquidity.append(highest_pe_oi_strike)
    bear_liquidity = []
    if prev_high is not None: bear_liquidity.append(prev_high)
    if high_zone is not None: bear_liquidity.append(high_zone)
    if highest_ce_oi_strike is not None: bear_liquidity.append(highest_ce_oi_strike)
    return bull_liquidity, bear_liquidity

def liquidity_zone_entry_check(price, bull_liq, bear_liq):
    if price is None or (isinstance(price,float) and math.isnan(price)): return None
    for zone in bull_liq:
        if zone is None: continue
        try:
            if abs(price - zone) <= 5: return "CE"
        except:
            continue
    for zone in bear_liq:
        if zone is None: continue
        try:
            if abs(price - zone) <= 5: return "PE"
        except:
            continue
    valid_bear = [z for z in bear_liq if z is not None]; valid_bull = [z for z in bull_liq if z is not None]
    if valid_bear and valid_bull:
        try:
            if price > max(valid_bear) or price < min(valid_bull): return "BOTH"
        except:
            return None
    return None

# --------- OPENING-PLAY (PURE LEVELS) ---------
def institutional_opening_play(index, df):
    """
    Pure level-based opening logic (no indicators).
    Returns "CE"/"PE"/None
    """
    try:
        prev_high = float(ensure_series(df['High']).iloc[-2])
        prev_low = float(ensure_series(df['Low']).iloc[-2])
        prev_close = float(ensure_series(df['Close']).iloc[-2])
        current_price = float(ensure_series(df['Close']).iloc[-1])
    except Exception:
        return None
    # Thresholds can be tweaked per-index if desired
    if current_price > prev_high + 10: return "CE"
    if current_price < prev_low - 10: return "PE"
    if current_price > prev_close + 20: return "CE"
    if current_price < prev_close - 20: return "PE"
    return None

# --------- BOTTOM-FISHING / INSTITUTIONAL LIQUIDITY LAYER (existing) ---------
def detect_bottom_fishing(index, df):
    try:
        close = ensure_series(df['Close']); low = ensure_series(df['Low']); high = ensure_series(df['High']); volume = ensure_series(df['Volume'])
        if len(close) < 6: return None
        bull_liq, bear_liq = institutional_liquidity_hunt(index, df)
        last_close = float(close.iloc[-1])
        wick = last_close - low.iloc[-1]
        body = abs(close.iloc[-1] - close.iloc[-2])
        vol_avg = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else volume.mean()
        vol_ratio = volume.iloc[-1] / (vol_avg if vol_avg > 0 else 1)
        if wick > body * 1.5 and vol_ratio > 1.2:
            for zone in bull_liq:
                if zone is None: continue
                if abs(last_close - zone) <= 5: return "CE"
        bear_wick = high.iloc[-1] - last_close
        if bear_wick > body * 1.5 and vol_ratio > 1.2:
            for zone in bear_liq:
                if zone is None: continue
                if abs(last_close - zone) <= 5: return "PE"
    except:
        return None
    return None

# --------- PULLBACK/TRAP/ORDERFLOW/MIMIC (existing) ---------
def detect_pullback_reversal(df):
    try:
        close = ensure_series(df['Close'])
        ema9 = ta.trend.EMAIndicator(close, 9).ema_indicator()
        ema21 = ta.trend.EMAIndicator(close, 21).ema_indicator()
        rsi = ta.momentum.RSIIndicator(close, 14).rsi()
        if len(close) < 6: return None
        if close.iloc[-6] > ema21.iloc[-6] and close.iloc[-3] <= ema21.iloc[-3] and close.iloc[-1] > ema9.iloc[-1] and rsi.iloc[-1] > 50: return "CE"
        if close.iloc[-6] < ema21.iloc[-6] and close.iloc[-3] >= ema21.iloc[-3] and close.iloc[-1] < ema9.iloc[-1] and rsi.iloc[-1] < 50: return "PE"
    except Exception:
        return None
    return None

def detect_institutional_trap(df, lookback=10):
    try:
        high = ensure_series(df['High']); low = ensure_series(df['Low']); close = ensure_series(df['Close']); volume = ensure_series(df['Volume'])
        if len(close) < lookback + 2: return None
        recent_high = high.rolling(lookback).max().iloc[-2]; recent_low = low.rolling(lookback).min().iloc[-2]
        avg_vol = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else volume.mean()
        if high.iloc[-1] > recent_high and close.iloc[-1] < recent_high and (avg_vol is None or volume.iloc[-1] > (avg_vol * 1.2)): return "PE"
        if low.iloc[-1] < recent_low and close.iloc[-1] > recent_low and (avg_vol is None or volume.iloc[-1] > (avg_vol * 1.2)): return "CE"
    except Exception:
        return None
    return None

def mimic_orderflow_logic(df):
    try:
        close = ensure_series(df['Close']); high = ensure_series(df['High']); low = ensure_series(df['Low']); volume = ensure_series(df['Volume'])
        rsi = ta.momentum.RSIIndicator(close, 14).rsi()
        if len(close) < 4: return None
        body = (high - low).abs(); wick_top = (high - close).abs(); wick_bottom = (close - low).abs()
        body_last = body.iloc[-1] if body.iloc[-1] != 0 else 1.0
        wick_top_ratio = wick_top.iloc[-1] / body_last; wick_bottom_ratio = wick_bottom.iloc[-1] / body_last
        vol_avg = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else volume.mean(); vol_ratio = volume.iloc[-1] / (vol_avg if vol_avg and vol_avg > 0 else 1)
        if close.iloc[-1] > close.iloc[-3] and rsi.iloc[-1] < rsi.iloc[-3] and wick_top_ratio > 0.6 and vol_ratio > 1.2: return "PE"
        if close.iloc[-1] < close.iloc[-3] and rsi.iloc[-1] > rsi.iloc[-3] and wick_bottom_ratio > 0.6 and vol_ratio > 1.2: return "CE"
    except Exception:
        return None
    return None

# --------- SMART-MONEY DIVERGENCE (NEW) ---------
def smart_money_divergence(df):
    """
    Detect stealth accumulation/distribution: price falling but RSI rising with higher volume -> accumulation (CE)
    or price rising but RSI falling with higher volume -> distribution (PE)
    """
    try:
        close = ensure_series(df['Close']); volume = ensure_series(df['Volume'])
        rsi = ta.momentum.RSIIndicator(close, 14).rsi()
        if len(close) < 10: return None
        # lookback windows
        p_short = close.iloc[-5]; p_now = close.iloc[-1]
        rsi_short = rsi.iloc[-5]; rsi_now = rsi.iloc[-1]
        vol_avg = volume.rolling(20).mean().iloc[-1] if len(volume)>=20 else volume.mean()
        vol_now = volume.iloc[-1]
        if p_now < p_short and rsi_now > rsi_short and vol_now > vol_avg*1.1:
            return "CE"
        if p_now > p_short and rsi_now < rsi_short and vol_now > vol_avg*1.1:
            return "PE"
    except Exception:
        return None
    return None

# --------- STOP-HUNT / LIQUIDITY GRAB DETECTOR (NEW) ---------
def detect_stop_hunt(df):
    """
    Look for spike wick beyond recent levels followed by quick retrace into the range -> indicates stop-hunt.
    Returns "CE"/"PE"/None depending on direction.
    """
    try:
        high = ensure_series(df['High']); low = ensure_series(df['Low']); close = ensure_series(df['Close']); volume = ensure_series(df['Volume'])
        if len(close) < 6: return None
        recent_high = high.iloc[-6:-1].max(); recent_low = low.iloc[-6:-1].min()
        last_high = high.iloc[-1]; last_low = low.iloc[-1]; last_close = close.iloc[-1]
        vol_avg = volume.rolling(20).mean().iloc[-1] if len(volume)>=20 else volume.mean()
        if last_high > recent_high * 1.002 and last_close < recent_high and volume.iloc[-1] > vol_avg*1.2:
            # wick above recent high then close back => upside stop-hunt -> PE trigger
            return "PE"
        if last_low < recent_low * 0.998 and last_close > recent_low and volume.iloc[-1] > vol_avg*1.2:
            # wick below recent low then close back => downside stop-hunt -> CE trigger
            return "CE"
    except Exception:
        return None
    return None

# --------- INSTITUTIONAL CONTINUATION (NEW) ---------
def detect_institutional_continuation(df):
    """
    Confirms breakout follow-through: ATR expansion + rising volume + price speed
    Returns "CE"/"PE"/None
    """
    try:
        close = ensure_series(df['Close']); high = ensure_series(df['High']); low = ensure_series(df['Low']); volume = ensure_series(df['Volume'])
        if len(close) < 10: return None
        atr = ta.volatility.AverageTrueRange(high, low, close, 14).average_true_range().iloc[-1]
        vol_avg = volume.rolling(20).mean().iloc[-1] if len(volume)>=20 else volume.mean()
        # price speed (last N bars)
        speed = (close.iloc[-1] - close.iloc[-3]) / (abs(close.iloc[-3]) + 1e-6)
        if atr > close.std() * 0.8 and volume.iloc[-1] > vol_avg * 1.2 and speed > 0.004:
            return "CE"
        if atr > close.std() * 0.8 and volume.iloc[-1] > vol_avg * 1.2 and speed < -0.004:
            return "PE"
    except Exception:
        return None
    return None

# --------- OI + DELTA FLOW DETECTION (existing + used by expiry) ---------
def oi_delta_flow_signal(index):
    try:
        url=f"https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        df=pd.DataFrame(requests.get(url,timeout=10).json())
        df=df[df['exch_seg'].str.upper().isin(["NFO", "BFO"])]
        df['symbol']=df['symbol'].str.upper()
        df_index=df[df['symbol'].str.contains(index)]
        if df_index.empty: return None
        if 'oi' not in df_index.columns:
            return None
        df_index['oi'] = pd.to_numeric(df_index['oi'], errors='coerce').fillna(0)
        df_index['oi_change'] = df_index['oi'].diff().fillna(0)
        ce_sum = df_index[df_index['symbol'].str.endswith("CE")]['oi_change'].sum()
        pe_sum = df_index[df_index['symbol'].str.endswith("PE")]['oi_change'].sum()
        if ce_sum > pe_sum * DELTA_OI_RATIO: return "CE"
        if pe_sum > ce_sum * DELTA_OI_RATIO: return "PE"
        if ce_sum>0 and pe_sum>0: return "BOTH"
    except Exception:
        return None

# --------- GAMMA SQUEEZE / EXPIRY LAYER (NEW) ---------
def is_expiry_day_for_index(index):
    try:
        ex = EXPIRIES.get(index)
        if not ex: return False
        # make case-insensitive parse e.g., "16 OCT 2025"
        dt = datetime.strptime(ex.title(), "%d %b %Y")
        today = (datetime.utcnow() + timedelta(hours=5, minutes=30)).date()
        return dt.date() == today
    except Exception:
        return False

def detect_gamma_squeeze(index, df):
    """
    Proxy gamma squeeze detection (no options greeks availability):
    - Check if today is expiry-like (or close to expiry)
    - Look for sudden volume spike + price acceleration near strikes with OI concentration
    - Returns dict {'side': 'CE'/'PE', 'confidence': 0-1} or None
    """
    try:
        close = ensure_series(df['Close']); volume = ensure_series(df['Volume']); high = ensure_series(df['High']); low = ensure_series(df['Low'])
        if len(close) < 6: return None
        vol_avg = volume.rolling(20).mean().iloc[-1] if len(volume)>=20 else volume.mean()
        vol_ratio = volume.iloc[-1] / (vol_avg if vol_avg>0 else 1)
        speed = (close.iloc[-1] - close.iloc[-3]) / (abs(close.iloc[-3]) + 1e-6)
        # quick OI proxy using scrip master concentration by strike (best-effort)
        try:
            url=f"https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
            df_s = pd.DataFrame(requests.get(url,timeout=10).json())
            df_s['symbol'] = df_s['symbol'].str.upper()
            df_index = df_s[df_s['symbol'].str.contains(index)]
            df_index['oi'] = pd.to_numeric(df_index.get('oi',0), errors='coerce').fillna(0)
            ce_oi = df_index[df_index['symbol'].str.endswith("CE")]['oi'].sum()
            pe_oi = df_index[df_index['symbol'].str.endswith("PE")]['oi'].sum()
        except Exception:
            ce_oi = pe_oi = 0
        # decide side using speed and oi bias
        if vol_ratio > GAMMA_VOL_SPIKE_THRESHOLD and abs(speed) > 0.002:
            if speed > 0:
                conf = min(1.0, (vol_ratio - 1.0) / 3.0 + (ce_oi / (pe_oi+1e-6)) * 0.1)
                return {'side':'CE','confidence':conf}
            else:
                conf = min(1.0, (vol_ratio - 1.0) / 3.0 + (pe_oi / (ce_oi+1e-6)) * 0.1)
                return {'side':'PE','confidence':conf}
    except Exception:
        return None
    return None

# --------- INSTITUTIONAL FLOW CHECKS (existing) ---------
def institutional_flow_signal(index, df5):
    try:
        last_close = float(ensure_series(df5["Close"]).iloc[-1])
        prev_close = float(ensure_series(df5["Close"]).iloc[-2])
    except:
        return None
    vol5 = ensure_series(df5["Volume"])
    vol_latest = float(vol5.iloc[-1])
    vol_avg = float(vol5.rolling(20).mean().iloc[-1]) if len(vol5) >= 20 else float(vol5.mean())
    if vol_latest > vol_avg*1.5 and abs(last_close-prev_close)/prev_close>0.003:
        return "BOTH"
    elif last_close>prev_close and vol_latest>vol_avg:
        return "CE"
    elif last_close<prev_close and vol_latest>vol_avg:
        return "PE"
    high_zone, low_zone = detect_liquidity_zone(df5, lookback=15)
    try:
        if high_zone is not None and last_close>=high_zone: return "PE"
        elif low_zone is not None and last_close<=low_zone: return "CE"
    except:
        return None
    return None

# --------- INSTITUTIONAL CONFIRMATION LAYER (existing) ---------
def institutional_confirmation_layer(index, df5, base_signal):
    try:
        close = ensure_series(df5['Close']); high = ensure_series(df5['High']); low = ensure_series(df5['Low']); volume = ensure_series(df5['Volume'])
        atr = ta.volatility.AverageTrueRange(high, low, close, 14).average_true_range().iloc[-1]
        last_close = float(close.iloc[-1]); prev_close = float(close.iloc[-2])
        body_strength = abs(last_close - (high.iloc[-1] + low.iloc[-1]) / 2)
        high_zone, low_zone = detect_liquidity_zone(df5, lookback=20)
        if base_signal == 'CE' and high_zone is not None and last_close >= high_zone: return False
        if base_signal == 'PE' and low_zone is not None and last_close <= low_zone: return False
        vol_avg = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else volume.mean()
        if volume.iloc[-1] < vol_avg or body_strength < atr*0.25: return False
        if atr < (close.std() * 0.25): return False
        # Breadth check pair
        if index == 'NIFTY':
            b_df = fetch_index_data('BANKNIFTY', '5m', '2d')
        elif index == 'BANKNIFTY':
            b_df = fetch_index_data('NIFTY', '5m', '2d')
        else:
            b_df = None
        if b_df is not None:
            b_close = ensure_series(b_df['Close'])
            b_ema9 = ta.trend.EMAIndicator(b_close, 9).ema_indicator().iloc[-1]
            b_ema21 = ta.trend.EMAIndicator(b_close, 21).ema_indicator().iloc[-1]
            if base_signal == 'PE' and b_ema9 > b_ema21: return False
            if base_signal == 'CE' and b_ema9 < b_ema21: return False
        # allow if pullback aligns
        try:
            pull_sig = detect_pullback_reversal(df5)
            if pull_sig and pull_sig == base_signal: return True
        except:
            pass
        try:
            trap_sig = detect_institutional_trap(df5)
            if trap_sig and trap_sig != base_signal: return False
        except:
            pass
    except Exception:
        return False
    return True

def institutional_flow_confirm(index, base_signal, df5):
    # Standard checks
    flow = institutional_flow_signal(index, df5)
    oi_flow = oi_delta_flow_signal(index)
    if flow and flow != 'BOTH' and flow != base_signal: return False
    if oi_flow and oi_flow != 'BOTH' and oi_flow != base_signal: return False
    if not institutional_confirmation_layer(index, df5, base_signal): return False
    return True

# --------- ANALYZE SIGNAL (integrated priority order) ---------
def analyze_index_signal(index):
    df5 = fetch_index_data(index, "5m", "2d")
    df15 = fetch_index_data(index, "15m", "10d")
    if df5 is None or df15 is None: return None
    close5 = ensure_series(df5["Close"]); close15 = ensure_series(df15["Close"])
    if len(close5) < 6 or len(close15) < 2: return None
    if close5.isna().iloc[-1] or close5.isna().iloc[-2]: return None
    last_close = float(close5.iloc[-1]); prev_close = float(close5.iloc[-2])

    # 0) Opening-play priority during opening window
    try:
        utc_now = datetime.utcnow()
        ist_now = utc_now + timedelta(hours=5, minutes=30)
        t = ist_now.time()
        opening_range_bias = OPENING_PLAY_ENABLED and (OPENING_START <= t <= OPENING_END)
        if opening_range_bias:
            op_sig = institutional_opening_play(index, df5)
            if op_sig:
                # opening plays are high-priority: lightweight fakeout check afterwards
                fakeout = False
                # quick liquidity check
                high_zone, low_zone = detect_liquidity_zone(df5, lookback=10)
                try:
                    if op_sig == "CE" and high_zone is not None and last_close >= high_zone: fakeout = True
                    if op_sig == "PE" and low_zone is not None and last_close <= low_zone: fakeout = True
                except:
                    fakeout = False
                return op_sig, df5, fakeout
    except Exception:
        pass

    # 1) Expiry / Gamma detection (high priority but non-blocking)
    try:
        gamma = detect_gamma_squeeze(index, df5)
        if gamma:
            gamma_msg = f"⚡ GAMMA-LIKE EVENT DETECTED: {index} {gamma['side']} (conf {gamma['confidence']:.2f})"
            # send info always
            send_telegram(gamma_msg)
            # actionable: if expiry day AND configured to be actionable
            if is_expiry_day_for_index(index) and EXPIRY_ACTIONABLE and not EXPIRY_INFO_ONLY:
                # create an expiry-weighted candidate signal
                cand = gamma['side']
                # if confidence high, try to relax confirmation requirements (but not remove them)
                # We'll allow an expiry-enabled trade if either institutional_flow_confirm OR (gamma.conf high + oi_flow agrees)
                oi_flow = oi_delta_flow_signal(index)
                if institutional_flow_confirm(index, cand, df5):
                    return cand, df5, False
                if gamma['confidence'] > 0.45 and oi_flow == cand:
                    # partial relax: treat as valid trade
                    return cand, df5, False
                # else treat it informational only
    except Exception:
        pass

    # 2) Bottom fishing (high-priority)
    bottom_sig = detect_bottom_fishing(index, df5)
    if bottom_sig:
        return bottom_sig, df5, False

    # 3) Traps / Pullback / Orderflow / Stop-hunt / Smart-money divergence (priority stack)
    try:
        trap_sig = detect_institutional_trap(df5)
        if trap_sig:
            return trap_sig, df5, True  # trap = fakeout
    except:
        trap_sig = None
    try:
        pull_sig = detect_pullback_reversal(df5)
        if pull_sig:
            return pull_sig, df5, False
    except:
        pull_sig = None
    try:
        flow_sig = mimic_orderflow_logic(df5)
        if flow_sig:
            return flow_sig, df5, False
    except:
        flow_sig = None
    try:
        stop_sig = detect_stop_hunt(df5)
        if stop_sig:
            # stop-hunts are often good short-term signals
            return stop_sig, df5, True
    except:
        stop_sig = None
    try:
        sm_sig = smart_money_divergence(df5)
        if sm_sig:
            return sm_sig, df5, False
    except:
        sm_sig = None

    # 4) Institutional continuation (confirms breakouts)
    cont_sig = detect_institutional_continuation(df5)
    if cont_sig:
        # continuation should be confirmed by institutional_flow_confirm
        if institutional_flow_confirm(index, cont_sig, df5):
            return cont_sig, df5, False

    # 5) EMA/RSI multi-timeframe relaxed logic (existing)
    try:
        ema9_5 = float(ta.trend.EMAIndicator(close5,9).ema_indicator().iloc[-1])
        ema21_5 = float(ta.trend.EMAIndicator(close5,21).ema_indicator().iloc[-1])
        rsi5 = float(ta.momentum.RSIIndicator(close5,14).rsi().iloc[-1])
        ema9_15 = float(ta.trend.EMAIndicator(close15,9).ema_indicator().iloc[-1])
        ema21_15 = float(ta.trend.EMAIndicator(close15,21).ema_indicator().iloc[-1])
        rsi15 = float(ta.momentum.RSIIndicator(close15,14).rsi().iloc[-1])
    except Exception:
        return None

    bullish = ((ema9_5 > ema21_5 and rsi5 > 50 and last_close > prev_close) or (ema9_15 > ema21_15 and rsi15 > 51))
    bearish = ((ema9_5 < ema21_5 and rsi5 < 50 and last_close < prev_close) or (ema9_15 < ema21_15 and rsi15 < 49))

    # Opening-range mild boost (already handled earlier, but keep)
    try:
        if opening_range_bias:
            if last_close > close5.iloc[-5:].mean(): bullish = True
            elif last_close < close5.iloc[-5:].mean(): bearish = True
    except:
        pass

    # 6) Liquidity hunt integration
    high_zone, low_zone = detect_liquidity_zone(df5, lookback=15)
    fakeout = False
    try:
        if bullish and high_zone is not None and last_close < high_zone: fakeout = True
        if bearish and low_zone is not None and last_close > low_zone: fakeout = True
    except:
        fakeout = False

    bull_liq, bear_liq = institutional_liquidity_hunt(index, df5)
    liquidity_side = liquidity_zone_entry_check(last_close, bull_liq, bear_liq)
    if liquidity_side:
        # require confirmation layer
        if institutional_flow_confirm(index, liquidity_side, df5):
            return liquidity_side, df5, fakeout

    if bullish:
        if institutional_flow_confirm(index, "CE", df5):
            return "CE", df5, fakeout
    if bearish:
        if institutional_flow_confirm(index, "PE", df5):
            return "PE", df5, fakeout

    return None

# --------- SYMBOL FORMAT FOR ALL INDICES (existing) ---------
def get_option_symbol(index, expiry_str, strike, opttype):
    # expiry_str could be "16 OCT 2025" -> normalize
    try:
        dt = datetime.strptime(expiry_str.title(), "%d %b %Y")
    except Exception:
        # try variations
        try:
            dt = datetime.strptime(expiry_str, "%d %b %Y")
        except Exception:
            # fallback to today
            dt = datetime.utcnow()
    if index == "SENSEX":
        # keep older naming pattern but ensure month is uppercase 3-letter
        year_short = dt.strftime("%y")
        month_code = dt.strftime("%b").upper()
        day = dt.strftime("%d")
        return f"SENSEX{year_short}{month_code}{strike}{opttype}"
    elif index == "FINNIFTY":
        return f"FINNIFTY{dt.strftime('%d%b%y').upper()}{strike}{opttype}"
    elif index == "MIDCPNIFTY":
        return f"MIDCPNIFTY{dt.strftime('%d%b%y').upper()}{strike}{opttype}"
    else:
        return f"{index}{dt.strftime('%d%b%y').upper()}{strike}{opttype}"

# --------- MONITOR WITH THREAD UPDATES (existing) ---------
def monitor_price_live(symbol,entry,targets,sl,fakeout,thread_id):
    last_high = entry
    weakness_sent = False
    in_trade=False
    for idx, val in active_trades.items():
        if val and val.get("symbol") == symbol:
            active_trades[idx]["status"] = "OPEN"
            break
    while True:
        if should_stop_trading():
            global STOP_SENT
            if not STOP_SENT:
                send_telegram(f"🛑 Market closed - Stopping monitoring for {symbol}", reply_to=thread_id)
                STOP_SENT = True
            for idx, val in active_trades.items():
                if val and val.get("symbol") == symbol:
                    active_trades[idx]["status"] = "CLOSED"
            break
        price = fetch_option_price(symbol)
        if not price:
            # if option price fetch fails, wait but also check underlying cache to avoid indefinite wait
            time.sleep(10); continue
        price = round(price)
        if not in_trade:
            if price >= entry:
                send_telegram(f"✅ ENTRY TRIGGERED at {price}", reply_to=thread_id)
                in_trade=True; last_high=price
        else:
            if price > last_high:
                send_telegram(f"🚀 {symbol} making new high → {price}", reply_to=thread_id)
                last_high=price
            elif not weakness_sent and price < sl*1.05:
                send_telegram(f"⚡ {symbol} showing weakness near SL {sl}", reply_to=thread_id)
                weakness_sent=True
            if price>=targets[0]:
                send_telegram(f"🌟 {symbol}: First Target {targets[0]} hit", reply_to=thread_id)
                for idx, val in active_trades.items():
                    if val and val.get("symbol") == symbol:
                        active_trades[idx]["status"] = "CLOSED"
                break
            if price<=sl:
                send_telegram(f"🔗 {symbol}: Stop Loss {sl} hit. Exit trade.", reply_to=thread_id)
                for idx, val in active_trades.items():
                    if val and val.get("symbol") == symbol:
                        active_trades[idx]["status"] = "CLOSED"
                break
        time.sleep(10)

# --------- EXPIRY CONFIG FOR ALL INDICES (existing) ---------
EXPIRIES = {
    "NIFTY": "14 OCT 2025",
    "BANKNIFTY": "28 OCT 2025",
    "SENSEX": "16 OCT 2025",
    "FINNIFTY": "28 OCT 2025",
    "MIDCPNIFTY": "28 OCT 2025",
    "EICHERMOT": "28 OCT 2025",
    "TRENT": "28 OCT 2025",
    "RELIANCE": "28 OCT 2025"
}

# ACTIVE TRACKING FOR ALL INDICES
active_trades = {"NIFTY": None, "BANKNIFTY": None, "SENSEX": None, "FINNIFTY": None, "MIDCPNIFTY": None, "EICHERMOT": None, "TRENT": None, "RELIANCE": None}

# --------- SEND SIGNAL (extended msg includes layer info) ---------
def send_signal(index,side,df,fakeout,layer_hint=None,extra_note=None):
    # validate df and underlying ltp robustly
    try:
        ltp_val = None
        try:
            ltp_val = float(ensure_series(df["Close"]).iloc[-1])
        except Exception:
            ltp_val = None
        # fallback to cache if missing
        if ltp_val is None or (isinstance(ltp_val,float) and math.isnan(ltp_val)):
            cached = read_ltp_cache(index)
            if cached:
                ltp_val = float(cached)
        if ltp_val is None or (isinstance(ltp_val,float) and math.isnan(ltp_val)):
            send_telegram(f"⚠️ {index}: could not determine underlying LTP. Signal skipped.")
            return
        strike=round_strike(index,ltp_val)
        if strike is None:
            send_telegram(f"⚠️ {index}: could not determine strike (ltp missing or invalid). Signal skipped.")
            return
        # enforce expiry lock - use EXPIRIES mapping (do not auto-change expiry)
        expiry = EXPIRIES.get(index)
        if not expiry:
            send_telegram(f"⚠️ {index}: expiry not configured. Signal skipped.")
            return
        symbol=get_option_symbol(index,expiry,strike,side)
        price=fetch_option_price(symbol)
        if not price:
            send_telegram(f"⚠️ {index} {symbol}: option price fetch failed. Signal skipped.")
            return
        high=ensure_series(df["High"]); low=ensure_series(df["Low"]); close=ensure_series(df["Close"])
        atr=float(ta.volatility.AverageTrueRange(high,low,close,14).average_true_range().iloc[-1])
        entry=round(price+5)
        sl=round(price-atr) if price-atr>0 else max(1, round(price*0.8))
        targets=[round(price+atr*1.5),round(price+atr*2)]
        layer_text = f"Layer: {layer_hint}" if layer_hint else "Layer: Institutional Core"
        extra = f"\n📝 Note: {extra_note}" if extra_note else ""
        msg=(f" GIT🔊 {index} {side} VSSIGNAL CONFIRMED\n"
             f"{layer_text}\n"
             f"🔹 Strike: {strike}\n"
             f"🟩 Buy Above ₹{entry}\n"
             f"🔵 SL: ₹{sl}\n"
             f"🌟 Targets: {targets[0]} / {targets[1]}\n"
             f"⚡ Fakeout: {'YES' if fakeout else 'NO'}{extra}")
        thread_id = send_telegram(msg)
        active_trades[index] = {"symbol":symbol,"entry":entry,"sl":sl,"targets":targets,"thread":thread_id,"status":"OPEN"}
        # write last-known LTP to cache
        write_ltp_cache(index, ltp_val)
        # start monitoring in a thread so send_signal can return quickly if needed
        monitor_thread = threading.Thread(target=monitor_price_live, args=(symbol,entry,targets,sl,fakeout,thread_id))
        monitor_thread.daemon = True
        monitor_thread.start()
        # do not block the caller — monitor thread will manage closure
        return
    except Exception as e:
        send_telegram(f"⚠️ send_signal error for {index}: {e}")
        return

# --------- THREAD WORKER (existing but integrates new layers) ---------
def trade_thread(index):
    global active_trades
    try:
        # Do not run if open trade already present
        if active_trades[index] and isinstance(active_trades[index], dict) and active_trades[index].get("status") == "OPEN":
            return
        sig = analyze_index_signal(index)
        side=None; fakeout=False; df=None; layer_hint=None; extra_note=None
        if sig:
            if isinstance(sig, tuple) and len(sig) == 3:
                side, df, fakeout = sig
            elif isinstance(sig, tuple) and len(sig) == 2:
                side, df = sig; fakeout = False
            else:
                side = sig
        df5=fetch_index_data(index,"5m","2d")
        inst_signal = institutional_flow_signal(index, df5) if df5 is not None else None
        oi_signal = oi_delta_flow_signal(index)
        final_signal = oi_signal or inst_signal or side

        # If gamma exists and expiry and EXPIRY_ACTIONABLE, let it overwrite final_signal if confidently matching and not conflicting
        try:
            gamma = None
            if df5 is not None:
                gamma = detect_gamma_squeeze(index, df5)
            if gamma and is_expiry_day_for_index(index) and EXPIRY_ACTIONABLE and gamma['confidence']>0.45:
                # prefer gamma if agrees with oi or inst flow, else require confirmation
                if oi_signal == gamma['side'] or inst_signal == gamma['side'] or side == gamma['side']:
                    final_signal = gamma['side']
                    layer_hint = "Expiry Gamma"
                else:
                    # if not agreeing, allow only informational send and keep normal flow
                    send_telegram(f"⚡ Info: {index} gamma event detected -> {gamma['side']} conf {gamma['confidence']:.2f}")
        except Exception:
            pass

        if final_signal == "BOTH":
            for s in ["CE","PE"]:
                if institutional_flow_confirm(index, s, df5):
                    send_signal(index,s,df,fakeout,layer_hint=layer_hint)
            return
        elif final_signal:
            if df is None: df = df5
            # BEFORE sending a signal, validate underlying LTP: do not use NaN or zero
            valid_underlying = None
            try:
                valid_underlying = float(ensure_series(df['Close']).iloc[-1])
                if isinstance(valid_underlying,float) and math.isnan(valid_underlying):
                    valid_underlying = None
            except Exception:
                valid_underlying = None
            if valid_underlying is None:
                # try cached
                cached = read_ltp_cache(index)
                if cached:
                    valid_underlying = float(cached)
            if valid_underlying is None:
                send_telegram(f"⚠️ {index}: Invalid underlying LTP. Skipping signal to avoid wrong strike.")
                return
            # ensure strike valid
            strike_check = round_strike(index, valid_underlying)
            if strike_check is None:
                send_telegram(f"⚠️ {index}: Invalid strike calculation. LTP: {valid_underlying}. Signal skipped.")
                return
            # confirmation
            if institutional_flow_confirm(index, final_signal, df5):
                send_signal(index,final_signal,df,fakeout,layer_hint=layer_hint)
            else:
                # expiry special relaxation: if expiry actionable and gamma exists and partial confirmation okay
                if is_expiry_day_for_index(index) and EXPIRY_ACTIONABLE:
                    gamma2 = None
                    if df5 is not None:
                        gamma2 = detect_gamma_squeeze(index, df5)
                    if gamma2 and gamma2['side'] == final_signal and gamma2['confidence']>0.5:
                        # relaxed path
                        send_signal(index,final_signal,df,fakeout,layer_hint="Expiry-Gamma-Relaxed")
        else:
            return
    except Exception as e:
        # catch-all for the worker so threads won't die silently
        send_telegram(f"⚠️ trade_thread {index} error: {e}")

# --------- MAIN LOOP (ALL INDICES PARALLEL) ---------
def run_algo_parallel():
    if not is_market_open():
        print("❌ Market closed - skipping iteration")
        return
    if should_stop_trading():
        global STOP_SENT
        if not STOP_SENT:
            send_telegram("🛑 Market closed at 3:30 PM IST - Algorithm stopped")
            STOP_SENT = True
        return
    threads=[]
    all_indices = ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "MIDCPNIFTY", "EICHERMOT", "TRENT", "RELIANCE"]
    for index in all_indices:
        t=threading.Thread(target=trade_thread,args=(index,))
        t.daemon = True
        t.start(); threads.append(t)
    for t in threads: t.join(timeout=30)

# --------- START (main loop) ---------
if __name__ == "__main__":
    # Wait until 9:15 AM IST before starting the main algorithm
    while True:
        try:
            utc_now = datetime.utcnow()
            ist_now = utc_now + timedelta(hours=5, minutes=30)
            current_time_ist = ist_now.time()
            
            # Check if it's before 9:15 AM IST
            if current_time_ist < dtime(9, 15):
                # Calculate seconds until 9:15 AM
                target_time = datetime.combine(ist_now.date(), dtime(9, 15))
                wait_seconds = (target_time - ist_now).total_seconds()
                
                if wait_seconds > 0:
                    print(f"⏰ Waiting until 9:15 AM IST. Current time: {ist_now.strftime('%H:%M:%S')} IST")
                    # Wait in smaller chunks to allow for graceful interruption
                    for _ in range(int(wait_seconds)):
                        time.sleep(1)
                        # Check if we should break early (in case of system time changes)
                        utc_now = datetime.utcnow()
                        ist_now = utc_now + timedelta(hours=5, minutes=30)
                        if ist_now.time() >= dtime(9, 15):
                            break
            
            # Start the main algorithm once it's 9:15 AM or later
            if not STARTED_SENT and is_market_open():
                send_telegram("🚀 GIT ULTIMATE MASTER ALGO STARTED - All 8 Indices Running with Institutional Layers (Opening play + Gamma expiry handling enabled).")
                STARTED_SENT = True
                STOP_SENT = False
            
            if should_stop_trading():
                if not STOP_SENT:
                    send_telegram("🛑 Market closing time reached - Algorithm stopped automatically")
                    STOP_SENT = True
                    STARTED_SENT = False
                # Wait until next market day
                utc_now = datetime.utcnow()
                ist_now = utc_now + timedelta(hours=5, minutes=30)
                if ist_now.time() >= dtime(15, 30):
                    # Calculate seconds until next day 9:15 AM
                    next_day = ist_now.date() + timedelta(days=1)
                    target_time = datetime.combine(next_day, dtime(9, 15))
                    wait_seconds = (target_time - ist_now).total_seconds()
                    
                    if wait_seconds > 0:
                        print(f"🌙 Market closed. Waiting until next market day 9:15 AM IST.")
                        time.sleep(wait_seconds)
                continue
            
            if is_market_open():
                run_algo_parallel()
            
            time.sleep(30)
            
        except Exception as e:
            # log exception and notify but don't crash
            try:
                send_telegram(f"⚠️ Error in main loop: {e}")
            except:
                print("Main loop error:", e)
            time.sleep(60)

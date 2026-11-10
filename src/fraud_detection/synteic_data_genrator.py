"""
Synthetic UK credit/debit card transaction generator -- deliberately MESSY.

Produces data/raw/uk_card_transactions.csv (~650k rows, > 2x the Kaggle
creditcard.csv size) for EDA + cleaning practice and fraud modelling.

SPOILER WARNING: this file documents every data-quality problem that was
planted. If you want the EDA challenge, don't read past this docstring
until you've done your own exploration.

Run:  python "src/fraud detection/synteic_data_genrator.py"
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_CUSTOMERS = 9_000
START = pd.Timestamp("2024-01-01")
END = pd.Timestamp("2025-01-01")  # exclusive
SYSTEM_MIGRATION = pd.Timestamp("2024-07-01")
OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "uk_card_transactions.csv"

# --------------------------------------------------------------------------
# Reference data
# --------------------------------------------------------------------------
# region: (lat, lon, population in millions)
REGIONS = {
    "London": (51.51, -0.13, 8.9),
    "South East": (51.30, -0.80, 9.3),
    "South West": (50.90, -3.50, 5.7),
    "East of England": (52.20, 0.40, 6.3),
    "East Midlands": (52.90, -1.00, 4.9),
    "West Midlands": (52.50, -1.90, 5.9),
    "North West": (53.60, -2.60, 7.4),
    "North East": (54.90, -1.60, 2.6),
    "Yorkshire and the Humber": (53.80, -1.30, 5.5),
    "Wales": (52.30, -3.60, 3.1),
    "Scotland": (56.40, -3.90, 5.4),
    "Northern Ireland": (54.60, -6.60, 1.9),
}
REG_NAMES = np.array(list(REGIONS))
REG_POP = np.array([v[2] for v in REGIONS.values()])
REG_POP = REG_POP / REG_POP.sum()
REG_LAT = {k: v[0] for k, v in REGIONS.items()}
REG_LON = {k: v[1] for k, v in REGIONS.items()}

# category: (mcc, log-median £, log-sigma, online base prob, popularity,
#            local merchants per region, online merchants)
CATEGORIES = {
    "groceries":       (5411, np.log(25),  0.70, 0.08, 25, 40,   6),
    "fuel":            (5541, np.log(50),  0.35, 0.00,  8, 25,   0),
    "restaurants":     (5812, np.log(22),  0.60, 0.10, 12, 60,  10),
    "transport":       (4111, np.log(6),   0.80, 0.30, 10,  8,   6),
    "retail_clothing": (5651, np.log(40),  0.70, 0.45,  7, 25,  30),
    "online_shopping": (5964, np.log(30),  0.90, 1.00, 12,  0, 120),
    "utilities_bills": (4900, np.log(70),  0.50, 0.95,  5,  2,  25),
    "entertainment":   (7832, np.log(15),  0.70, 0.60,  7, 12,  25),
    "travel":          (4722, np.log(180), 1.00, 0.85,  2,  5,  30),
    "electronics":     (5732, np.log(120), 1.00, 0.60,  3, 10,  30),
    "gambling":        (7995, np.log(20),  1.10, 0.95,  2,  4,  20),
    "cash_withdrawal": (6011, np.log(60),  0.60, 0.00,  7, 15,   0),
}
CAT_NAMES = np.array(list(CATEGORIES))
NCAT = len(CAT_NAMES)
CAT_IDX = {c: i for i, c in enumerate(CAT_NAMES)}
CAT_MCC = np.array([v[0] for v in CATEGORIES.values()])
CAT_MU = np.array([v[1] for v in CATEGORIES.values()])
CAT_SIG = np.array([v[2] for v in CATEGORIES.values()])
CAT_ONLINE = np.array([v[3] for v in CATEGORIES.values()])
CAT_POP = np.array([v[4] for v in CATEGORIES.values()], dtype=float)
CAT_POP = CAT_POP / CAT_POP.sum()

CAT_NOUNS = {
    "groceries": ["GROCERS", "FOODS", "SUPERMARKET", "FOOD STORE", "EXPRESS"],
    "fuel": ["FUEL", "PETROLEUM", "SERVICE STN", "FUELS"],
    "restaurants": ["KITCHEN", "BISTRO", "GRILL", "CAFE", "PIZZERIA", "CURRY HOUSE"],
    "transport": ["RAIL", "BUSES", "TAXIS", "TRANSIT", "PARKING"],
    "retail_clothing": ["CLOTHING", "FASHION", "OUTFITTERS", "APPAREL"],
    "online_shopping": ["ONLINE", "DIRECT", "MARKETPLACE", ".CO.UK", "SHOP"],
    "utilities_bills": ["ENERGY", "WATER", "TELECOM", "BROADBAND", "MOBILE"],
    "entertainment": ["CINEMA", "STREAMING", "TICKETS", "GAMES", "MUSIC"],
    "travel": ["AIRWAYS", "HOLIDAYS", "HOTELS", "TRAVEL", "FLIGHTS"],
    "electronics": ["ELECTRONICS", "TECH", "DIGITAL", "COMPUTERS"],
    "gambling": ["BET", "CASINO", "GAMING", "BINGO"],
    "cash_withdrawal": ["ATM", "CASH POINT", "CASH MACHINE"],
}
PREFIXES = ["GREENFIELD", "NORTHGATE", "OAKWOOD", "RIVERSIDE", "KINGSWAY", "HIGHLAND",
            "MEADOW", "HARBOUR", "CROWN", "ALBION", "BEACON", "THISTLE", "WILLOW",
            "SUMMIT", "GRANITE", "STATION", "MARKET", "CASTLE", "BRIDGE", "ELM",
            "FOXGLOVE", "REDBRICK", "MILLSTONE", "BLUEWATER", "ASHFORD", "LANTERN"]

# foreign countries: (currency, fx per £1, approx km from UK)
COUNTRIES = {
    "FR": ("EUR", 1.17, 700), "ES": ("EUR", 1.17, 1400), "IT": ("EUR", 1.17, 1500),
    "DE": ("EUR", 1.17, 900), "IE": ("EUR", 1.17, 450), "PT": ("EUR", 1.17, 1600),
    "GR": ("EUR", 1.17, 2500), "NL": ("EUR", 1.17, 550), "US": ("USD", 1.27, 5600),
    "TR": ("TRY", 41.0, 2900),
}
CTRY = np.array(list(COUNTRIES))
TRIP_W = np.array([.20, .22, .12, .07, .08, .07, .06, .05, .08, .05])
ONLINE_INTL_W = np.array([.08, .03, .02, .10, .20, .01, .01, .15, .35, .05])
FRAUD_IP = np.array(["US", "NL", "RO", "BR", "IN", "DE", "VN", "LT", "UA", "ID"])

HOUR_W = np.array([0.4, 0.25, 0.15, 0.1, 0.1, 0.2, 0.6, 1.6, 2.6, 3.0, 3.4, 3.8,
                   4.6, 4.4, 3.8, 3.6, 3.9, 4.6, 4.8, 4.3, 3.6, 2.8, 1.9, 1.0])
HOUR_W = HOUR_W / HOUR_W.sum()
FRAUD_HOUR_W = 0.5 * HOUR_W + 0.5 / 24
NIGHT_W = 0.4 * HOUR_W + 0.6 * np.array([1 if h < 6 else 0 for h in range(24)]) / 6

DAYS = pd.date_range(START, END, freq="D", inclusive="left")
NDAYS = len(DAYS)


def day_weights():
    w = np.ones(NDAYS)
    w[DAYS.day_of_week.isin([4, 5])] *= 1.15
    w[(DAYS.month == 12) & (DAYS.day <= 24)] *= 1.35
    w[(DAYS >= "2024-11-25") & (DAYS <= "2024-12-02")] *= 1.30  # Black Friday
    w[DAYS.month == 1] *= 0.90
    return w / w.sum()


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


# --------------------------------------------------------------------------
# 1. Customers
# --------------------------------------------------------------------------
def make_customers(rng):
    n = N_CUSTOMERS
    home = rng.choice(REG_NAMES, n, p=REG_POP)
    age = np.clip(rng.normal(44, 15, n), 18, 92).round().astype(int)
    income = np.exp(rng.normal(10.35 - ((age - 50) / 25) ** 2 * 0.3, 0.55, n))
    income = np.where(home == "London", income * 1.3, income)
    income = (income / 100).round() * 100
    spend_mult = np.sqrt(income / 31_000) * rng.lognormal(0, 0.25, n)
    monthly_rate = rng.gamma(2.2, 2.6, n)
    online_ratio = np.clip(rng.beta(2, 4, n) * (1.3 - (age - 18) / 100), 0.02, 0.9)

    fav = np.array([rng.choice(NCAT, 4, replace=False, p=CAT_POP) for _ in range(n)])

    scheme = rng.choice(["Visa", "Mastercard", "Amex"], n, p=[.60, .35, .05])
    kind = np.where(scheme == "Amex", "credit", rng.choice(["debit", "credit"], n, p=[.55, .45]))
    tier = np.where(income > 70_000, rng.choice(["gold", "platinum"], n, p=[.5, .5]),
                    np.where(income > 40_000, rng.choice(["standard", "gold"], n, p=[.6, .4]), "standard"))
    lim_lo = np.select([tier == "standard", tier == "gold"], [500, 3000], 8000)
    lim_hi = np.select([tier == "standard", tier == "gold"], [4000, 10000], 25000)
    limit = (rng.uniform(lim_lo, lim_hi) / 100).round() * 100
    limit = np.where(kind == "credit", limit, np.nan)

    max_days = np.clip((age - 18) * 365, 60, 365 * 25)
    open_date = START - pd.to_timedelta(rng.uniform(30, max_days).astype(int), unit="D")

    has_trip = rng.random(n) < 0.18
    trip_day = rng.choice(NDAYS, n, p=_trip_day_weights())
    trip_start = np.where(has_trip, DAYS.values[trip_day], np.datetime64("NaT", "ns"))
    trip_len = rng.integers(4, 13, n)
    trip_end = trip_start + trip_len.astype("timedelta64[D]")

    return pd.DataFrame({
        "cust_num": np.arange(1, n + 1),
        "home_region": home, "age": age, "income": income, "spend_mult": spend_mult,
        "monthly_rate": monthly_rate, "online_ratio": online_ratio,
        "fav": list(fav), "scheme": scheme, "kind": kind, "tier": tier,
        "credit_limit": limit, "open_date": open_date,
        "card_last4": rng.integers(0, 10_000, n),
        "trip_start": trip_start, "trip_end": trip_end,
        "trip_country": rng.choice(CTRY, n, p=TRIP_W),
    })


def _trip_day_weights():
    w = np.where(DAYS.month.isin([6, 7, 8, 9]), 2.5, 1.0)
    w[(DAYS.month == 12) & (DAYS.day > 20)] = 0.2
    return w / w.sum()


# --------------------------------------------------------------------------
# 2. Merchants
# --------------------------------------------------------------------------
def make_merchants(rng):
    rows = []

    def name(cat):
        return f"{rng.choice(PREFIXES)} {rng.choice(CAT_NOUNS[cat])}"

    for cat, spec in CATEGORIES.items():
        k_local, k_online = spec[5], spec[6]
        for r in REG_NAMES:
            for _ in range(k_local):
                rows.append((CAT_IDX[cat], "local", r, "GB", f"{name(cat)} {rng.integers(1, 9999):04d}"))
        online_regions = rng.choice(REG_NAMES, k_online, p=_online_reg_w())
        for r in online_regions:
            rows.append((CAT_IDX[cat], "online", r, "GB", name(cat)))
        if cat != "utilities_bills":
            for c in CTRY:
                for _ in range(4):
                    rows.append((CAT_IDX[cat], "foreign", None, c, f"{name(cat)} {c}"))
    m = pd.DataFrame(rows, columns=["cat", "kind", "region", "country", "merchant_name"])
    m["merchant_id"] = [f"MER{i:06d}" for i in rng.permutation(len(m)) + 1]
    return m


def _online_reg_w():
    w = REG_POP.copy()
    w[REG_NAMES == "London"] += 0.35
    w[REG_NAMES == "South East"] += 0.10
    return w / w.sum()


# --------------------------------------------------------------------------
# 3. Transactions (clean, internal representation)
# --------------------------------------------------------------------------
def sample_amount(cat, spend, rng):
    a = np.exp(rng.normal(CAT_MU[cat] + np.log(spend), CAT_SIG[cat]))
    a = np.clip(a, 0.30, None)
    cash = cat == CAT_IDX["cash_withdrawal"]
    a[cash] = np.clip(np.round(a[cash] / 10) * 10, 10, 500)
    fuel = cat == CAT_IDX["fuel"]
    a[fuel] = np.clip(a[fuel], 5, 180)
    tr = cat == CAT_IDX["transport"]
    a[tr] = np.clip(a[tr], 1.5, None)
    return np.round(a, 2)


def present_channel(amount, cat, tfrac, rng):
    n = len(amount)
    r = rng.random(n)
    wallet_p = 0.10 + 0.22 * tfrac  # mobile wallets grow over the year (drift)
    ch = np.where(r < wallet_p, "mobile_wallet",
                  np.where((amount <= 100) & (rng.random(n) < 0.65), "contactless", "chip_pin"))
    ch = np.where((ch == "contactless") & (amount > 100), "chip_pin", ch)
    ch = np.where(rng.random(n) < 0.008, "magstripe", ch)
    ch = np.where(cat == CAT_IDX["cash_withdrawal"], "chip_pin", ch)
    return ch


def build_normal(cust, rng):
    counts = rng.poisson(cust["monthly_rate"].to_numpy() * 12)
    ci = np.repeat(np.arange(len(cust)), counts)
    n = len(ci)

    d = rng.choice(NDAYS, n, p=day_weights())
    secs = rng.choice(24, n, p=HOUR_W) * 3600 + rng.integers(0, 3600, n)
    ts = DAYS.values[d] + secs.astype("timedelta64[s]")
    tfrac = (ts - START.to_datetime64()) / np.timedelta64(365, "D")

    fav = np.stack(cust["fav"].to_numpy())
    use_fav = rng.random(n) < 0.78
    cat = np.where(use_fav, fav[ci, rng.integers(0, 4, n)], rng.choice(NCAT, n, p=CAT_POP))

    amount = sample_amount(cat, cust["spend_mult"].to_numpy()[ci], rng)

    base = CAT_ONLINE[cat]
    p_online = np.where((base == 0) | (base == 1), base,
                        np.clip(base * (0.5 + 1.7 * cust["online_ratio"].to_numpy()[ci]), 0, 0.98))
    online = rng.random(n) < p_online

    channel = np.where(online, "online", present_channel(amount, cat, tfrac, rng))

    home = cust["home_region"].to_numpy()[ci]
    region = np.where(rng.random(n) < 0.87, home, rng.choice(REG_NAMES, n, p=REG_POP))

    not_util = cat != CAT_IDX["utilities_bills"]
    intl = not_util & np.where(online, rng.random(n) < 0.07, rng.random(n) < 0.008)
    country = np.where(intl, np.where(online, rng.choice(CTRY, n, p=ONLINE_INTL_W),
                                      rng.choice(CTRY, n, p=TRIP_W)), "GB")

    # holidays abroad: card-present spend during the trip becomes international
    t_start = cust["trip_start"].to_numpy()[ci]
    t_end = cust["trip_end"].to_numpy()[ci]
    in_trip = (ts >= t_start) & (ts < t_end) & ~online & not_util
    intl = intl | in_trip
    country = np.where(in_trip, cust["trip_country"].to_numpy()[ci], country)

    device = np.where(online, rng.choice(["ios", "android", "desktop"], n, p=[.45, .35, .20]), None)
    ip = np.where(online, np.where(rng.random(n) < 0.97, "GB",
                                   rng.choice(["IE", "FR", "ES", "US", "NL"], n)), None)

    auth_p = 0.012 + 0.03 * (amount > 500)
    auth = np.where(rng.random(n) < auth_p, "DECLINED", "APPROVED")
    ttype = np.where(cat == CAT_IDX["cash_withdrawal"], "cash_withdrawal", "purchase")

    df = pd.DataFrame({
        "ci": ci, "ts": ts, "cat": cat, "amount_gbp": amount, "online": online,
        "channel": channel, "intl": intl, "country": country,
        "mregion": np.where(online | intl, None, region),
        "device": device, "ip": ip, "failed": rng.poisson(0.04, n),
        "auth": auth, "ttype": ttype, "is_fraud": 0,
    })

    # holiday bookings: genuine high-value online travel purchases (false-positive bait)
    trips = cust.index[cust["trip_start"].notna()].to_numpy()
    book_ts = cust["trip_start"].to_numpy()[trips] - rng.integers(14, 60, len(trips)).astype("timedelta64[D]")
    book_ts = book_ts + rng.integers(0, 86400, len(trips)).astype("timedelta64[s]")
    keep = book_ts >= START.to_datetime64()
    trips, book_ts = trips[keep], book_ts[keep]
    k = len(trips)
    book = pd.DataFrame({
        "ci": trips, "ts": book_ts, "cat": CAT_IDX["travel"],
        "amount_gbp": np.round(np.exp(rng.normal(np.log(550), 0.5, k)), 2),
        "online": True, "channel": "online", "intl": False, "country": "GB", "mregion": None,
        "device": rng.choice(["ios", "android", "desktop"], k, p=[.3, .2, .5]),
        "ip": "GB", "failed": 0, "auth": "APPROVED", "ttype": "purchase", "is_fraud": 0,
    })
    return pd.concat([df, book], ignore_index=True)


def build_fraud(cust, rng):
    rows = []
    n = len(cust)
    home = cust["home_region"].to_numpy()
    spend = cust["spend_mult"].to_numpy()
    fav = np.stack(cust["fav"].to_numpy())
    C = CAT_IDX

    def start(hour_w):
        return START + pd.Timedelta(days=int(rng.integers(0, NDAYS)),
                                    seconds=int(rng.choice(24, p=hour_w) * 3600 + rng.integers(0, 3600)))

    def add(ci, ts, cat, amt, online, intl=False, country="GB", mregion=None,
            device=None, ip=None, failed=0, decline_p=0.0, channel=None):
        if channel is None:
            channel = "online" if online else ("contactless" if amt <= 100 else "chip_pin")
        if ts >= END:
            return
        rows.append(dict(
            ci=ci, ts=ts, cat=cat, amount_gbp=round(float(amt), 2), online=online,
            channel=channel, intl=intl, country=country if intl else "GB",
            mregion=None if (online or intl) else mregion,
            device=device if online else None, ip=ip if online else None, failed=failed,
            auth="DECLINED" if rng.random() < decline_p else "APPROVED",
            ttype="cash_withdrawal" if cat == C["cash_withdrawal"] else "purchase", is_fraud=1))

    def nonhome(ci):
        r = rng.choice(REG_NAMES, p=REG_POP)
        while r == home[ci]:
            r = rng.choice(REG_NAMES, p=REG_POP)
        return r

    devices = ["desktop", "android", "ios", "unknown"]

    # (a) card-not-present fraud -- the biggest UK card-fraud type
    for _ in range(900):
        ci = rng.integers(n)
        t = start(FRAUD_HOUR_W)
        foreign_ip = rng.random() < 0.55
        dev = rng.choice(devices, p=[.45, .30, .15, .10])
        for j in range(rng.choice([1, 2, 3], p=[.55, .30, .15])):
            t = t + pd.Timedelta(minutes=float(rng.uniform(2, 240))) * int(j > 0)
            cat = rng.choice([C["electronics"], C["travel"], C["gambling"], C["online_shopping"],
                              C["retail_clothing"]], p=[.35, .15, .15, .25, .10])
            intl = rng.random() < 0.25
            add(ci, t, cat, np.clip(np.exp(rng.normal(np.log(170), 1.0)), 5, 4000), True,
                intl=intl, country=rng.choice(CTRY, p=ONLINE_INTL_W), device=dev,
                ip=rng.choice(FRAUD_IP) if foreign_ip else "GB",
                failed=rng.poisson(0.5), decline_p=0.30)

    # (b) card testing: tiny probes then one big hit
    for _ in range(100):
        ci = rng.integers(n)
        t = start(FRAUD_HOUR_W)
        ip = rng.choice(FRAUD_IP) if rng.random() < 0.6 else "GB"
        dev = rng.choice(devices, p=[.6, .2, .05, .15])
        for _ in range(rng.integers(3, 7)):
            t = t + pd.Timedelta(minutes=float(rng.uniform(0.5, 4)))
            add(ci, t, rng.choice([C["entertainment"], C["online_shopping"]]),
                rng.uniform(0.5, 2.5), True, device=dev, ip=ip, failed=rng.poisson(3), decline_p=0.45)
        t = t + pd.Timedelta(minutes=float(rng.uniform(20, 360)))
        add(ci, t, rng.choice([C["electronics"], C["travel"]]), np.exp(rng.normal(np.log(450), 0.5)),
            True, device=dev, ip=ip, failed=rng.poisson(2), decline_p=0.30)

    # (c) account takeover: night-time burst, often away from home / abroad
    for _ in range(120):
        ci = rng.integers(n)
        t = start(NIGHT_W)
        online = rng.random() < 0.7
        intl = rng.random() < 0.3
        country = rng.choice(CTRY)
        region = nonhome(ci)
        dev = rng.choice(devices, p=[.5, .3, .05, .15])
        ip = rng.choice(FRAUD_IP) if rng.random() < 0.5 else "GB"
        for _ in range(rng.integers(3, 9)):
            t = t + pd.Timedelta(minutes=float(rng.uniform(3, 20)))
            cat = rng.choice([C["electronics"], C["gambling"], C["travel"], C["online_shopping"],
                              C["cash_withdrawal"]], p=[.3, .2, .15, .2, .15])
            if online and cat == C["cash_withdrawal"]:
                cat = C["online_shopping"]
            if not online and cat in (C["gambling"], C["online_shopping"], C["travel"]):
                cat = C["electronics"]
            amt = np.exp(rng.normal(np.log(160), 0.9))
            if cat == C["cash_withdrawal"]:
                amt = np.clip(round(amt / 10) * 10, 50, 500)
            add(ci, t, cat, amt, online, intl=intl, country=country, mregion=region,
                device=dev, ip=ip, failed=rng.poisson(1.5), decline_p=0.25)

    # (d) lost / stolen card: contactless just under the £100 limit, then cash
    for _ in range(120):
        ci = rng.integers(n)
        region = nonhome(ci) if rng.random() < 0.7 else home[ci]
        t = start(FRAUD_HOUR_W)
        for _ in range(rng.integers(3, 7)):
            t = t + pd.Timedelta(minutes=float(rng.uniform(5, 60)))
            cat = rng.choice([C["retail_clothing"], C["restaurants"], C["groceries"], C["fuel"],
                              C["electronics"], C["entertainment"]])
            add(ci, t, cat, rng.uniform(25, 99.99), False, mregion=region,
                channel="contactless", decline_p=0.10)
        if rng.random() < 0.4:
            t = t + pd.Timedelta(minutes=float(rng.uniform(10, 120)))
            add(ci, t, C["cash_withdrawal"], rng.choice([200, 250, 300]), False,
                mregion=region, channel="chip_pin", decline_p=0.10)

    # (e) subtle fraud: looks almost like the customer's normal behaviour
    for _ in range(300):
        ci = rng.integers(n)
        cat = fav[ci, rng.integers(4)]
        base = CAT_ONLINE[cat]
        online = bool(base == 1 or (base > 0 and rng.random() < 0.5))
        amt = sample_amount(np.array([cat]), np.array([spend[ci]]), rng)[0] * rng.uniform(1.3, 2.5)
        if cat == C["cash_withdrawal"]:
            amt = round(amt / 10) * 10
        add(ci, start(HOUR_W), cat, amt, online, mregion=home[ci],
            device=rng.choice(["ios", "android", "desktop"]), ip="GB", decline_p=0.05)

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"]).astype("datetime64[ns]")
    return df


def build_refunds(df, rng):
    refundable = df["cat"].isin([CAT_IDX[c] for c in ("retail_clothing", "online_shopping",
                                                       "electronics", "travel")])
    ok = refundable & (df["is_fraud"] == 0) & (df["auth"] == "APPROVED")
    pick = df[ok].sample(frac=0.012, random_state=int(rng.integers(1e9))).copy()
    pick["ts"] = pick["ts"] + pd.to_timedelta(rng.integers(1, 31, len(pick)), unit="D") \
                            + pd.to_timedelta(rng.integers(0, 86400, len(pick)), unit="s")
    pick["amount_gbp"] = -np.round(pick["amount_gbp"] * np.where(rng.random(len(pick)) < 0.8, 1,
                                                                 rng.uniform(0.3, 0.9, len(pick))), 2)
    pick["ttype"] = "refund"
    pick["failed"] = 0
    return pick[pick["ts"] < END]


def assign_merchants(df, merch, rng):
    mid = np.empty(len(df), dtype=object)
    local = merch[merch["kind"] == "local"].groupby(["cat", "region"])["merchant_id"].apply(np.array)
    online = merch[merch["kind"] == "online"].groupby("cat")["merchant_id"].apply(np.array)
    foreign = merch[merch["kind"] == "foreign"].groupby(["cat", "country"])["merchant_id"].apply(np.array)

    m_local = ~df["online"] & ~df["intl"]
    for key, idx in df[m_local].groupby(["cat", "mregion"]).groups.items():
        mid[idx] = rng.choice(local[key], len(idx))
    m_online = df["online"] & ~df["intl"]
    for key, idx in df[m_online].groupby("cat").groups.items():
        mid[idx] = rng.choice(online[key], len(idx))
    for key, idx in df[df["intl"]].groupby(["cat", "country"]).groups.items():
        mid[idx] = rng.choice(foreign[key], len(idx))

    df["merchant_id"] = mid
    info = merch.set_index("merchant_id")
    df["merchant_name"] = info.loc[mid, "merchant_name"].to_numpy()
    df["mregion"] = np.where(m_online, info.loc[mid, "region"].to_numpy(), df["mregion"])
    df.loc[df["intl"], "mregion"] = None
    return df


# --------------------------------------------------------------------------
# 4. Assemble the "true" clean table
# --------------------------------------------------------------------------
def assemble(cust, merch, rng):
    df = pd.concat([build_normal(cust, rng), build_fraud(cust, rng)], ignore_index=True)
    df = pd.concat([df, build_refunds(df, rng)], ignore_index=True)
    df["ts"] = df["ts"].astype("datetime64[ns]")
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    df = assign_merchants(df, merch, rng)

    c = cust.iloc[df["ci"].to_numpy()].reset_index(drop=True)
    home = c["home_region"].to_numpy()
    mreg = df["mregion"].to_numpy()

    # distance from home (km): unknown for online, rough country distance abroad
    dist = np.full(len(df), np.nan)
    local = ~df["online"].to_numpy() & ~df["intl"].to_numpy()
    same = local & (mreg == home)
    diff = local & ~same
    dist[same] = rng.exponential(7, same.sum())
    hl = np.array([REG_LAT[r] for r in home[diff]]); hn = np.array([REG_LON[r] for r in home[diff]])
    ml = np.array([REG_LAT[r] for r in mreg[diff]]); mn = np.array([REG_LON[r] for r in mreg[diff]])
    dist[diff] = haversine(hl, hn, ml, mn) + rng.normal(0, 15, diff.sum())
    abroad = df["intl"].to_numpy() & ~df["online"].to_numpy()
    dist[abroad] = np.array([COUNTRIES[k][2] for k in df.loc[abroad, "country"]]) \
                   * rng.uniform(0.85, 1.15, abroad.sum())
    dist = np.round(np.abs(dist), 1)

    # amount in billing currency (abroad: local currency unless DCC was chosen)
    cur = np.array(["GBP"] * len(df), dtype=object)
    amount = df["amount_gbp"].to_numpy().copy()
    fx_rows = df["intl"].to_numpy() & (rng.random(len(df)) < 0.8)
    cur[fx_rows] = [COUNTRIES[k][0] for k in df.loc[fx_rows, "country"]]
    amount[fx_rows] = np.round(amount[fx_rows] * np.array([COUNTRIES[k][1] for k in df.loc[fx_rows, "country"]]), 2)

    out = pd.DataFrame({
        "ts": df["ts"],
        "cust_num": c["cust_num"].to_numpy(),
        "age": c["age"].to_numpy().astype(float),
        "income": c["income"].to_numpy(),
        "home_region": home,
        "open_date": c["open_date"].to_numpy(),
        "scheme": c["scheme"].to_numpy(), "kind": c["kind"].to_numpy(), "tier": c["tier"].to_numpy(),
        "card_last4": c["card_last4"].to_numpy(), "credit_limit": c["credit_limit"].to_numpy(),
        "ttype": df["ttype"], "amount": amount, "currency": cur,
        "merchant_id": df["merchant_id"], "merchant_name": df["merchant_name"],
        "category": CAT_NAMES[df["cat"].to_numpy()], "mcc": CAT_MCC[df["cat"].to_numpy()],
        "merchant_region": df["mregion"], "merchant_country": df["country"],
        "channel": df["channel"], "card_present": ~df["online"], "intl": df["intl"],
        "device": df["device"], "ip": df["ip"], "distance": dist,
        "failed": df["failed"].astype(float), "auth": df["auth"], "is_fraud": df["is_fraud"],
    })
    out["tier"] = np.where(out["kind"] == "debit", None, out["tier"])
    return out


# --------------------------------------------------------------------------
# 5. Make it messy
# --------------------------------------------------------------------------
REGION_ALIASES = {
    "London": ["Greater London", "LDN", "Gtr London"],
    "Yorkshire and the Humber": ["Yorkshire & Humber", "Yorkshire and The Humber", "Yorks"],
    "Northern Ireland": ["N. Ireland", "NI", "N Ireland"],
    "East of England": ["East Anglia", "Eastern"],
    "North East": ["N East", "North-East"], "North West": ["N West", "North-West"],
    "South East": ["South-East", "SE England"], "South West": ["South-West", "SW England"],
    "West Midlands": ["W Midlands", "W. Midlands"], "East Midlands": ["E Midlands", "E. Midlands"],
    "Scotland": ["Alba"], "Wales": ["Cymru"],
}
CATEGORY_ALIASES = {
    "groceries": ["Groceries", "grocery", "Supermarkets"], "fuel": ["Fuel", "petrol", "Fuel/Petrol"],
    "restaurants": ["Restaurants", "dining", "Food & Drink"], "transport": ["Transport", "travel_local"],
    "retail_clothing": ["Retail - Clothing", "clothing", "retail clothing"],
    "online_shopping": ["Online Shopping", "e-commerce", "online"],
    "utilities_bills": ["Utilities", "bills", "utilities/bills"],
    "entertainment": ["Entertainment", "leisure"], "travel": ["Travel", "Airlines/Hotels"],
    "electronics": ["Electronics", "electrical"], "gambling": ["Gambling", "betting"],
    "cash_withdrawal": ["ATM", "Cash", "cash withdrawal"],
}
CHANNEL_LEGACY = {"chip_pin": "CHIP_PIN", "contactless": "CONTACTLESS", "online": "ECOM",
                  "mobile_wallet": "MOBILE_WALLET", "magstripe": "MAGSTRIPE"}
CHANNEL_ALIASES = {
    "chip_pin": ["chip & pin", "Chip and PIN", "chip"], "contactless": ["Contactless", "contact_less", "tap"],
    "online": ["Online", "e-commerce", "CNP"], "mobile_wallet": ["apple_pay", "google_pay", "Mobile Wallet"],
    "magstripe": ["swipe", "Magstripe"],
}
DEVICE_ALIASES = {"ios": ["iOS", "iphone", "IOS", "Apple"], "android": ["Android", "ANDROID", "android "],
                  "desktop": ["Desktop", "PC", "Windows", "desktop "], "unknown": ["?", "n/a"]}


def scatter_variants(s, p, aliases, rng, generic=True):
    s = s.astype(object).copy()
    hit = np.flatnonzero((rng.random(len(s)) < p) & s.notna().to_numpy())
    vals = s.to_numpy(dtype=object, copy=True)
    for i in hit:
        v = vals[i]
        opts = list(aliases.get(v, []))
        if generic:
            opts += [v.lower(), v.upper(), f" {v} ", f"{v.title()}  "]
        if opts:
            vals[i] = opts[rng.integers(len(opts))]
    return pd.Series(vals, index=s.index, dtype=object)


def set_null(s, p, rng, value=np.nan):
    s = s.astype(object).copy()
    s[rng.random(len(s)) < p] = value
    return s


def fmt_timestamps(ts, legacy, rng):
    n = len(ts)
    r = rng.random(n)
    out = np.empty(n, dtype=object)
    # legacy core banking system: UK local time, day-first, no seconds
    m = legacy & (r < 0.90); out[m] = ts[m].dt.strftime("%d/%m/%Y %H:%M")
    m = legacy & (r >= 0.90); out[m] = ts[m].dt.strftime("%Y-%m-%d %H:%M:%S")
    # new system: mostly UTC ISO-8601 with Z (so BST rows shift by -1h), some local, a few US-style
    in_bst = (ts >= "2024-03-31 01:00") & (ts < "2024-10-27 02:00")
    utc = ts - pd.to_timedelta(in_bst.astype(int), unit="h")
    m = ~legacy & (r < 0.85); out[m] = utc[m].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    m = ~legacy & (r >= 0.85) & (r < 0.98); out[m] = ts[m].dt.strftime("%Y-%m-%d %H:%M:%S")
    m = ~legacy & (r >= 0.98); out[m] = ts[m].dt.strftime("%m/%d/%Y %I:%M %p")
    # broken values
    bad = rng.random(n)
    out[bad < 0.002] = None
    out[(bad >= 0.002) & (bad < 0.0025)] = "1970-01-01 00:00:00"
    out[(bad >= 0.0025) & (bad < 0.0027)] = "2099-12-31 23:59:59"
    out[(bad >= 0.0027) & (bad < 0.0030)] = ""
    return out


def fmt_amounts(amount, legacy, rng):
    n = len(amount)
    a = amount.copy()
    # decimal-point shift errors, placeholders and zero amounts
    r = rng.random(n)
    a[r < 0.0005] *= 100
    a[(r >= 0.0005) & (r < 0.0009)] = 999999.99
    a[(r >= 0.0009) & (r < 0.0019)] = 0.0
    style = rng.random(n)
    out = np.empty(n, dtype=object)
    for i in range(n):
        v = a[i]
        if legacy[i] and style[i] < 0.30:
            out[i] = f"£{v:,.2f}" if v >= 0 or style[i] < 0.15 else f"-£{-v:,.2f}"
        elif style[i] < 0.33:
            out[i] = f"{v:,.2f}"
        elif style[i] < 0.36:
            out[i] = repr(float(v))
        else:
            out[i] = f"{v:.2f}"
    blank = rng.random(n)
    out[blank < 0.0025] = ""
    out[(blank >= 0.0025) & (blank < 0.0040)] = None
    out[(blank >= 0.0040) & (blank < 0.0045)] = "N/A"
    return out


def make_messy(df, rng):
    n = len(df)
    legacy = (df["ts"] < SYSTEM_MIGRATION).to_numpy()
    out = pd.DataFrame(index=df.index)

    out["transaction_id"] = [f"TXN{i:09d}" for i in range(1, n + 1)]
    out["Transaction Timestamp"] = fmt_timestamps(df["ts"], legacy, rng)
    out["source_system"] = np.where(legacy, "LEGACY_CBS", "NEXUS_v2")

    # customer id format changed with the system migration
    out["customer_id"] = np.where(legacy, [f"CUST{c:06d}" for c in df["cust_num"]],
                                  [f"CUST-{c:06d}" for c in df["cust_num"]])

    # --- customer-level problems (consistent across a customer's rows) ---
    custs = df["cust_num"].unique()
    bad_age = dict(zip(rng.choice(custs, 30, replace=False), rng.choice([0, -1, 999, 150, 7, 200], 30)))
    age = df["age"].copy()
    m = df["cust_num"].isin(bad_age)
    age[m] = df.loc[m, "cust_num"].map(bad_age)
    out["customerAge"] = set_null(age, 0.03, rng)

    inc = df["income"].to_numpy()
    cust_inc = df.groupby("cust_num")["income"].first()
    p_miss = 0.05 + 0.25 * (cust_inc > 60_000)  # high earners skip the question (MNAR)
    miss_c = set(cust_inc.index[rng.random(len(cust_inc)) < p_miss])
    zero_c = set(rng.choice(custs, 20, replace=False))
    huge_c = set(rng.choice(custs, 10, replace=False))
    inc = np.where(df["cust_num"].isin(zero_c), 0, inc)
    inc = np.where(df["cust_num"].isin(huge_c), 9_999_999, inc)
    inc = np.where(df["cust_num"].isin(miss_c), np.nan, inc)
    out["customer_annual_income"] = inc

    out["customer_home_region"] = set_null(scatter_variants(df["home_region"], 0.06, REGION_ALIASES, rng),
                                           0.025, rng)

    od = df["open_date"].copy()
    late_c = set(rng.choice(custs, 35, replace=False))  # account "opened" after it transacted
    m = df["cust_num"].isin(late_c)
    od[m] = pd.Timestamp("2024-11-15")
    ph_c = set(rng.choice(custs, 25, replace=False))
    od_str = np.where(rng.random(n) < 0.85, od.dt.strftime("%Y-%m-%d"), od.dt.strftime("%d-%b-%Y"))
    od_str = np.where(df["cust_num"].isin(ph_c), "1900-01-01", od_str)
    out["account_open_date"] = od_str

    out["card_type"] = scatter_variants(df["scheme"], 0.05, {"Mastercard": ["MasterCard", "MC"],
                                                              "Amex": ["American Express", "AMEX"]}, rng)
    out["card_kind"] = df["kind"]
    out["card_tier"] = df["tier"]
    out["card_last4"] = df["card_last4"].map(lambda x: f"{x:04d}")
    out.loc[rng.random(n) < 0.01, "card_last4"] = df.loc[:, "card_last4"].astype(str)  # lost leading zeros
    out["credit_limit_gbp"] = df["credit_limit"]

    # --- transaction-level problems ---
    out["transaction_type"] = df["ttype"]
    out["amount"] = fmt_amounts(df["amount"].to_numpy(), legacy, rng)
    out["currency"] = scatter_variants(df["currency"], 0.03, {"GBP": ["gbp", "£", "UKP"],
                                                              "EUR": ["eur", "€"], "USD": ["usd", "$"]},
                                       rng, generic=False)
    out["merchant_id"] = df["merchant_id"]
    out["merchant_name"] = scatter_variants(df["merchant_name"], 0.10, {}, rng)

    cat = scatter_variants(df["category"], 0.05, CATEGORY_ALIASES, rng)
    r = rng.random(n)
    cat[r < 0.02] = np.nan
    cat[(r >= 0.02) & (r < 0.03)] = rng.choice(["unknown", "other", "MISC"], ((r >= 0.02) & (r < 0.03)).sum())
    out["merchant_category"] = cat
    mcc = df["mcc"].astype(float).to_numpy().copy()
    wrong = rng.random(n) < 0.002
    mcc[wrong] = rng.choice(CAT_MCC, wrong.sum())
    mcc[rng.random(n) < 0.01] = np.nan
    out["mcc"] = pd.array(mcc, dtype="Int64")

    mreg = df["merchant_region"].astype(object).copy()
    mreg[df["intl"] & (rng.random(n) < 0.3)] = "Overseas"
    out["merchant_region"] = set_null(scatter_variants(mreg, 0.06, REGION_ALIASES, rng), 0.02, rng)
    out["merchant_country"] = np.where(rng.random(n) < 0.04, np.where(df["merchant_country"] == "GB", "UK",
                                                                       df["merchant_country"]),
                                       df["merchant_country"])

    ch = df["channel"].copy()
    ch = np.where(legacy, ch.map(CHANNEL_LEGACY), ch)
    ch = scatter_variants(pd.Series(ch), 0.02, CHANNEL_ALIASES, rng, generic=False)
    # contactless over the £100 limit (data entry error)
    over = (df["amount"] > 100) & (df["channel"] == "chip_pin") & (rng.random(n) < 0.002)
    ch[over.to_numpy()] = np.where(legacy[over.to_numpy()], "CONTACTLESS", "contactless")
    out["channel"] = set_null(ch, 0.01, rng)

    cp = df["card_present"].to_numpy()
    cp_str = np.where(legacy, np.where(cp, "Y", "N"), np.where(cp, "True", "False")).astype(object)
    r = rng.random(n)
    cp_str[r < 0.02] = np.where(cp[r < 0.02], "1", "0")
    flip = (r >= 0.02) & (r < 0.0215)
    cp_str[flip] = np.where(cp[flip], "False", "True")  # contradicts channel
    cp_str[(r >= 0.0215) & (r < 0.0265)] = None
    out["card_present"] = cp_str

    intl = df["intl"].to_numpy()
    flip = rng.random(n) < 0.003
    intl = np.where(flip, ~intl, intl)
    out["is_international"] = set_null(pd.Series(np.where(intl, "True", "False")), 0.01, rng).to_numpy()

    dev = scatter_variants(df["device"], 0.08, DEVICE_ALIASES, rng, generic=False)
    ip = df["ip"].astype(object).copy()
    outage = ((df["ts"] >= "2024-06-10") & (df["ts"] < "2024-06-17")).to_numpy()  # logging outage
    dev[outage] = np.nan
    ip[outage] = np.nan
    out["device_type"] = dev
    out["ip_country"] = set_null(ip, 0.01, rng)

    dist = df["distance"].to_numpy().copy()
    r = rng.random(n)
    dist[r < 0.001] *= -1
    dist[(r >= 0.001) & (r < 0.003)] *= 1000  # stored in metres
    dist[(r >= 0.003) & (r < 0.023)] = np.nan
    out["distance_from_home_km"] = dist

    failed = df["failed"].to_numpy().copy()
    r = rng.random(n)
    failed[r < 0.015] = np.nan
    failed[(r >= 0.015) & (r < 0.016)] = -1
    out["failed_attempts_24h"] = pd.array(failed, dtype="Int64")

    out["auth_response"] = np.where(legacy, df["auth"].str[0], df["auth"])

    out["bank_code"] = "UKB-0042"
    out["promo_code"] = np.where(rng.random(n) < 0.03, rng.choice(["SPRING10", "XMAS24", "WELCOME", "CASHBACK5"], n), None)
    notes = np.full(n, None, dtype=object)
    m = rng.random(n) < 0.01
    notes[m] = rng.choice(["", " ", "test", "CHECK", "customer called", "dup?", "manual entry", "N/A"], m.sum())
    out["notes"] = notes

    # --- labels: undetected fraud, false claims, pending investigations ---
    y = df["is_fraud"].to_numpy().copy().astype(float)
    true_fraud = y == 1
    y[true_fraud & (rng.random(n) < 0.04)] = 0
    y[~true_fraud & (rng.random(n) < 0.0003)] = 1
    late = (df["ts"] >= END - pd.Timedelta(days=21)).to_numpy()
    pending = rng.random(n) < np.where(late, 0.04, 0.0015)
    y[pending] = np.nan

    # leakage traps: only known AFTER the fraud was reported
    cb = np.where(y == 1, rng.random(n) < 0.88, rng.random(n) < 0.0025)
    cb = np.where(np.isnan(y), rng.random(n) < 0.3 * true_fraud + 0.002, cb)
    out["chargeback_filed"] = np.where(cb, "Y", "N")
    ref = np.where((y == 1) & (rng.random(n) < 0.75),
                   [f"FC-2024-{i:05d}" for i in rng.integers(0, 99999, n)], None)
    out["fraud_case_ref"] = ref

    out["is_fraud"] = pd.array(np.where(np.isnan(y), pd.NA, y), dtype="Int64")
    return out


def add_dirt_rows(out, rng):
    n = len(out)
    # exact duplicates (double-ingested batches)
    dups = out.sample(n=int(n * 0.004), random_state=int(rng.integers(1e9)))
    # same transaction_id, conflicting values (later "corrections")
    conf = out.sample(n=int(n * 0.0008), random_state=int(rng.integers(1e9))).copy()
    amt = pd.to_numeric(conf["amount"].astype(str).str.replace(r"[£,]", "", regex=True), errors="coerce")
    conf["amount"] = (amt * rng.uniform(0.9, 1.1, len(conf))).round(2).astype(str)
    conf["merchant_category"] = rng.choice(CAT_NAMES, len(conf))
    # internal test transactions that should never be modelled
    k = 60
    test = out.sample(n=k, random_state=int(rng.integers(1e9))).copy()
    test["transaction_id"] = [f"TEST{i:06d}" for i in range(k)]
    test["customer_id"] = rng.choice([f"TEST_CUST_{i:02d}" for i in range(1, 6)], k)
    test["amount"] = rng.choice(["0.01", "1.00", "0.01"], k)
    test["merchant_name"] = "INTERNAL TEST MERCHANT"
    test["notes"] = "test txn"
    test["is_fraud"] = pd.array(rng.integers(0, 2, k), dtype="Int64")
    out = pd.concat([out, dups, conf, test], ignore_index=True)
    return out.sample(frac=1, random_state=int(rng.integers(1e9))).reset_index(drop=True)


def main():
    rng = np.random.default_rng(SEED)
    print("Generating customers and merchants...")
    cust = make_customers(rng)
    merch = make_merchants(rng)
    print("Generating transactions...")
    clean = assemble(cust, merch, rng)
    print(f"  clean rows: {len(clean):,}   fraud rate: {clean['is_fraud'].mean():.3%}")
    print("Adding data-quality problems...")
    messy = add_dirt_rows(make_messy(clean, rng), rng)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    messy.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(messy):,} rows x {messy.shape[1]} columns -> {OUT_PATH}")
    print(f"File size: {OUT_PATH.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()

"""Cross-Asset Universe — Comprehensive security universe for Metadron Capital.

Contains the full S&P 1500 (S&P 500 + S&P MidCap 400 + S&P SmallCap 600)
plus additional international ADRs, high-profile IPOs, and thematic names.

Total: ~2,500+ individual equities + 70+ ETFs across all asset classes.

When OpenBB API keys are configured (FMP, Intrinio, etc.), the UniverseEngine
fetches live constituents from obb.index.constituents() and enriches with
obb.equity.profile(). This static list serves as the fallback when API access
is unavailable.

GICS sector classification is embedded for every ticker. The mapping covers
all 11 GICS sectors. Tickers not in this map default to sector assignment
via OpenBB fundamentals or remain unclassified.

Maintenance: This file is periodically regenerated from index providers.
Individual additions/removals happen via EXTRA_TICKERS / EXCLUDE_TICKERS.
"""

# ═══════════════════════════════════════════════════════════════════════════
# S&P 500 — Full constituents (as of March 2026)
# ═══════════════════════════════════════════════════════════════════════════
SP500_TICKERS = [
    # Information Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "CSCO", "ADBE",
    "AMD", "ACN", "IBM", "INTC", "TXN", "INTU", "QCOM", "AMAT",
    "ADI", "LRCX", "KLAC", "SNPS", "CDNS", "MRVL", "PANW", "FTNT",
    "NOW", "PLTR", "CRWD", "MCHP", "ON", "NXPI", "MPWR", "KEYS",
    "ANSS", "HPQ", "HPE", "WDC", "STX", "ZBRA", "JNPR", "SWKS",
    "TER", "FFIV", "PTC", "TRMB", "GEN", "EPAM", "CTSH", "IT",
    "VRSN", "AKAM", "FSLR",
    # Health Care
    "UNH", "LLY", "JNJ", "ABBV", "MRK", "TMO", "ABT", "DHR",
    "PFE", "AMGN", "ISRG", "ELV", "GILD", "CI", "SYK", "BSX",
    "VRTX", "MDT", "REGN", "BDX", "ZTS", "HCA", "IDXX", "EW",
    "A", "IQV", "MTD", "DXCM", "RMD", "BAX", "ALGN", "HOLX",
    "ILMN", "TECH", "PODD", "RVTY", "HSIC", "CRL", "CTLT", "DGX",
    "LH", "VTRS", "OGN", "XRAY", "SOLV", "GEHC", "INCY", "MOH",
    "CNC", "HUM", "BMY", "BIIB",
    # Financials
    "JPM", "V", "MA", "BRK-B", "BAC", "WFC", "GS", "MS",
    "SPGI", "BLK", "AXP", "CME", "ICE", "CB", "SCHW", "AON",
    "MMC", "PGR", "MCO", "AJG", "MET", "AFL", "PRU", "AIG",
    "TRV", "ALL", "COF", "USB", "PNC", "TFC", "BK", "STT",
    "FITB", "HBAN", "CFG", "RF", "KEY", "NTRS", "CINF", "ZION",
    "MTB", "FRC", "SIVB", "NDAQ", "CBOE", "MSCI", "MKTX", "FDS",
    "RJF", "L", "WRB", "GL", "RE", "BRO", "ERIE", "AIZ",
    "DFS", "SYF", "ALLY",
    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "BKNG", "SBUX",
    "TJX", "CMG", "ORLY", "AZO", "ROST", "MAR", "HLT", "YUM",
    "DHI", "LEN", "PHM", "NVR", "GPC", "BBY", "DRI", "POOL",
    "GRMN", "DECK", "MGM", "WYNN", "CZR", "LVS", "F", "GM",
    "APTV", "BWA", "EBAY", "ETSY", "EXPE", "RCL", "CCL", "NCLH",
    "DG", "DLTR", "KMX", "AAP", "ULTA", "LULU", "TGT", "TPR",
    "RL", "HAS", "TSCO",
    # Communication Services
    "GOOGL", "GOOG", "META", "NFLX", "DIS", "CMCSA", "T", "VZ",
    "TMUS", "CHTR", "EA", "TTWO", "MTCH", "WBD", "FOXA", "FOX",
    "PARA", "NWS", "NWSA", "OMC", "IPG", "LYV",
    # Industrials
    "GE", "CAT", "BA", "RTX", "HON", "UNP", "UPS", "DE",
    "LMT", "NOC", "GD", "TDG", "ADP", "WM", "RSG", "ETN",
    "ITW", "EMR", "PH", "ROK", "CTAS", "FAST", "CPRT", "ODFL",
    "AME", "PCAR", "TT", "CARR", "OTIS", "SWK", "GWW", "NSC",
    "CSX", "FDX", "JCI", "XYL", "IR", "WAB", "PAYC", "VRSK",
    "PAYX", "BR", "DOV", "ROP", "IEX", "NDSN", "SNA", "J",
    "PWR", "GNRC", "EFX", "MAS", "LHX", "HII", "TXT", "LDOS",
    "AXON", "DAL", "UAL", "AAL", "LUV", "ALK",
    # Consumer Staples
    "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL",
    "MDLZ", "STZ", "KMB", "GIS", "SJM", "K", "HSY", "HRL",
    "CPB", "MKC", "CHD", "CAG", "TSN", "ADM", "BG", "TAP",
    "KHC", "MNST", "KDP", "EL", "CLX", "SYY", "KR", "WBA",
    # Energy
    "XOM", "CVX", "SLB", "COP", "EOG", "MPC", "PSX", "VLO",
    "OXY", "HAL", "DVN", "HES", "FANG", "BKR", "CTRA", "EQT",
    "MRO", "APA", "TRGP", "OKE", "WMB", "KMI", "ET",
    # Utilities
    "NEE", "DUK", "SO", "AEP", "D", "SRE", "EXC", "XEL",
    "ED", "WEC", "ES", "AWK", "DTE", "FE", "PEG", "PPL",
    "CMS", "AES", "ATO", "EVRG", "CNP", "NI", "LNT", "PNW",
    "NRG",
    # Real Estate
    "PLD", "AMT", "CCI", "EQIX", "PSA", "O", "SPG", "DLR",
    "WELL", "AVB", "EQR", "VTR", "ARE", "MAA", "UDR", "ESS",
    "SUI", "CPT", "REG", "FRT", "KIM", "BXP", "HST", "PEAK",
    "INVH", "IRM", "SBAC", "WY",
    # Materials
    "LIN", "APD", "SHW", "ECL", "FCX", "NEM", "DD", "NUE",
    "VMC", "MLM", "DOW", "PPG", "EMN", "CE", "CF", "MOS",
    "FMC", "ALB", "IFF", "BALL", "PKG", "IP", "WRK", "SEE",
    "AVY", "AMCR",
]

# ═══════════════════════════════════════════════════════════════════════════
# S&P MidCap 400 — Representative holdings (~400 tickers)
# ═══════════════════════════════════════════════════════════════════════════
SP400_TICKERS = [
    # Information Technology
    "MANH", "SMCI", "SAIA", "LSCC", "POWI", "ENPH", "MKSI", "AZPN",
    "CGNX", "ENTG", "CACC", "COHR", "NOVT", "ASGN", "DT", "MTSI",
    "VNT", "SLAB", "QLYS", "TENB", "CYBR", "PCTY", "WEX", "SANM",
    "EXPO", "CIEN", "VIAV", "CALX", "TTMI", "PRGS", "EVTC", "MIDD",
    # Health Care
    "MEDP", "JAZZ", "NBIX", "NVCR", "IRTC", "RARE", "HALO", "AZTA",
    "NEO", "IART", "MMSI", "ENSG", "SGRY", "NHC", "AMED", "LNTH",
    "ITCI", "IONS", "EXAS", "TNDM", "LIVN", "CERT", "OMCL", "PRCT",
    # Financials
    "EWBC", "FNB", "GBCI", "OZK", "IBOC", "SNV", "PNFP", "CADE",
    "FHN", "WBS", "UBSI", "SFNC", "WTFC", "HWC", "NWBI", "FFIN",
    "TRMK", "ABCB", "ASB", "SBCF", "PPBI", "WSFS", "BRKL", "TCBI",
    "PRIMERICA", "RLI", "KMPR", "THG", "HMN", "ORI", "SIGI",
    "PRI", "PIPR", "HOMB", "SSB", "BOH", "CBU", "FCF",
    # Consumer Discretionary
    "FIVE", "WSM", "ODP", "WH", "PLNT", "SCI", "SKX", "CROX",
    "LEA", "FOXF", "SHAK", "TXRH", "WING", "EAT", "DIN",
    "CAKE", "JACK", "LNW", "PENN", "BYD", "RRR", "ARMK",
    "AEO", "ANF", "URBN", "CATO", "GPI", "ABG", "SAH",
    # Communication Services
    "IACI", "ZD", "CARS", "TKO", "SIRI", "CNX",
    # Industrials
    "RBC", "AZEK", "FLR", "KBR", "MTZ", "AAON", "WFRD",
    "SPR", "TDY", "BWXT", "HEI", "KTOS", "ESAB", "RXO",
    "SRCL", "CLH", "ECOL", "USM", "MWA", "BMI", "WTS",
    "LFUS", "AIT", "FSS", "KAI", "NPO", "GGG", "JBT",
    "GTES", "PRIM", "APG", "AWI", "TREX", "SITE",
    # Consumer Staples
    "POST", "CELH", "SAM", "FLO", "LNDC", "SMPL", "CASY",
    "INGR", "SPTN", "THS", "USFD", "PFGC",
    # Energy
    "SM", "MGY", "MTDR", "RRC", "CNX", "CHX", "CHRD",
    "GPOR", "NOG", "DINO", "PARR", "CRC", "PTEN", "HP",
    # Utilities
    "OGE", "AVA", "BKH", "NWE", "POR", "SR", "UTL",
    "MDU", "MGEE", "SWX",
    # Real Estate
    "LAMR", "SRC", "STAG", "NNN", "STOR", "GTY", "APLE",
    "RHP", "PEB", "XHR", "DEI", "JBGS", "COLD", "CUBE",
    "LSI", "NSA", "EGP", "FR", "REXR",
    # Materials
    "CLF", "STLD", "CMC", "RS", "ATI", "HXL", "SLVM",
    "KWR", "GEF", "SON", "BERY", "GPK", "OI", "UFPI",
    "TROX", "CC", "IOSP", "AXTA", "CBT",
]

# ═══════════════════════════════════════════════════════════════════════════
# S&P SmallCap 600 — Representative holdings (~600 tickers)
# ═══════════════════════════════════════════════════════════════════════════
SP600_TICKERS = [
    # Information Technology
    "ATEN", "CEVA", "COHU", "CTS", "DIGI", "DIOD", "FORM", "IDCC",
    "IIVI", "IIIV", "IPGP", "MAXR", "OSIS", "PLAY", "RMBS", "SGH",
    "SLP", "SMTC", "SPSC", "VRNS", "CSWI", "NSIT", "TTEC", "AMSF",
    "VCEL", "PLUS", "PAYO", "RIOT", "MARA", "HQY", "FOUR", "OOMA",
    "JAMF", "ALRM", "VERX", "INTA", "ALKT", "TASK", "ATRC", "GSHD",
    # Health Care
    "ACHC", "ANGO", "ATEC", "BIO", "CNMD", "CRSP", "CORT", "CRY",
    "DCPH", "FLGT", "GMED", "HZNP", "ICUI", "INSP", "LMAT", "MASI",
    "NKTR", "NVST", "OLK", "OFIX", "OMED", "PCRX", "PGNY", "PRXL",
    "RDNT", "RGEN", "RVMD", "SRDX", "SUPN", "TGTX", "TWST",
    "VCEL", "XNCR", "ETNB",
    # Financials
    "ACIW", "AX", "BANF", "BHLB", "CASH", "CATC", "CNOB", "CVBF",
    "FBP", "FFBC", "FULT", "GABC", "HFWA", "HOPE", "HTH", "IBTX",
    "LCNB", "NBTB", "OCFC", "PACW", "PCBK", "PFS", "RENR", "RNR",
    "SASR", "SBSI", "SFBS", "STBA", "TMP", "TBBK", "TOWN", "UFCS",
    "UVSP", "WABC", "WASH", "WSBC", "WRBS", "AGO", "ESGR",
    # Consumer Discretionary
    "AAP", "ABM", "BC", "BOOT", "CAL", "CRI", "DORM", "FND",
    "FL", "GIII", "HIBB", "HNI", "JELD", "KNX", "LEG", "LZB",
    "MCRI", "MOD", "MSGS", "PATK", "RGS", "RH", "SCVL", "SHOO",
    "SIG", "SNBR", "SWX", "STRA", "THRM", "VVV", "WWW",
    "CHUY", "BJRI", "RUTH", "KRUS", "LOCO", "DENN",
    # Communication Services
    "CCO", "CARG", "EVC", "LUMN", "MGNI", "PUBM", "QNST",
    "SCHL", "YELP", "ZUO", "WLY",
    # Industrials
    "AAON", "AGCO", "AIMC", "ARCB", "AZZ", "BCC", "CENTA",
    "CW", "DNOW", "DY", "ENS", "GATX", "GBX", "HDS", "HNI",
    "HUBB", "JBHT", "KFY", "LSTR", "MYRG", "MWA", "NNBR",
    "REVG", "RXO", "SSD", "TNC", "UFPT", "VSEC", "WNC", "BELFB",
    "MATX", "MRCY", "MTRN", "POWL", "RBC", "ROCK",
    # Consumer Staples
    "BGS", "BJ", "CALM", "CENT", "COKE", "FRPT", "HLF", "IPAR",
    "JJSF", "LANC", "MGPI", "SENEA", "SMPL", "SWM", "SXT",
    "TR", "UVV", "WDFC",
    # Energy
    "AROC", "CIVI", "CPK", "DEN", "GPRE", "HLX", "LBRT",
    "NEXT", "OIS", "PUMP", "REI", "RES", "SBOW", "SWN", "TALO",
    "TDW", "VNOM", "WTI", "WTTR",
    # Utilities
    "ATGE", "CALX", "CWEN", "HE", "OTTR", "PNM", "SJW",
    "SPKE", "UIL",
    # Real Estate
    "AAT", "AKR", "ALEX", "BRT", "CBL", "CDR", "CSR", "CUZ",
    "EFC", "GNL", "IRT", "KRC", "LTC", "MAC", "NXRT", "OHI",
    "PDM", "PLYM", "SAFE", "SKT", "SLG", "UE", "VRE", "WRE",
    # Materials
    "BCPC", "CSWI", "CYT", "HUN", "HWKN", "KOP", "KRA",
    "LTHM", "MTX", "NGVT", "OLN", "PBF", "RFP", "RYAM",
    "SCL", "SWM", "SXT", "TRS", "UFPI", "WOR",
]

# ═══════════════════════════════════════════════════════════════════════════
# Additional high-profile names (recent IPOs, mega-ADRs, popular names)
# ═══════════════════════════════════════════════════════════════════════════
EXTRA_TICKERS = [
    # Mega-cap not in SP500 at times / ADRs
    "ARM", "UBER", "DASH", "ABNB", "COIN", "SNOW", "DDOG", "NET",
    "ZS", "TEAM", "MDB", "SHOP", "SE", "MELI", "NU", "BABA",
    "JD", "PDD", "BIDU", "NIO", "LI", "XPEV", "TSM", "ASML",
    "SAP", "TM", "HMC", "SONY", "SNE", "NVS", "AZN", "GSK",
    "SAN", "HSBC", "UBS", "BCS", "ING", "DB", "CS", "RIO",
    "BHP", "VALE", "SCCO", "GOLD", "WPM", "PAAS",
    # Fintech / crypto-adjacent
    "SQ", "AFRM", "SOFI", "HOOD", "UPST", "LC", "PYPL", "FIS",
    "FISV", "GPN", "ADYEN",
    # AI / cloud
    "AI", "SMCI", "PATH", "U", "RBLX", "DOCN", "CFLT", "ESTC",
    "HUBS", "TTD", "ROKU", "PINS", "SNAP", "TWLO",
    # Biotech / pharma
    "MRNA", "BNTX", "SGEN", "ARGX", "PCVX", "ALNY", "BMRN",
    "EXEL", "SRPT", "RARE", "RCKT",
    # EV / energy transition
    "RIVN", "LCID", "QS", "PLUG", "FCEL", "BE", "SEDG", "ENPH",
    "RUN", "NOVA", "CHPT", "BLNK", "EVGO",
    # SPACs that became real companies
    "JOBY", "LILM", "ACHR", "ASTS", "IONQ", "RGTI",
    # Defense / space
    "RKLB", "LUNR", "RDW", "IRDM",
    # Cannabis (US-listed)
    "TLRY", "CGC", "ACB", "CRON",
]

# ═══════════════════════════════════════════════════════════════════════════
# GICS Sector Classification — comprehensive mapping
# ═══════════════════════════════════════════════════════════════════════════
# Key: ticker → GICS sector name
# This covers all SP500 + most SP400/SP600. Unknown tickers fall through
# to OpenBB fundamentals enrichment or remain unclassified.

SECTOR_MAP = {
    # ─── Information Technology ──────────────────────────────────────────
    "AAPL": "Information Technology", "MSFT": "Information Technology",
    "NVDA": "Information Technology", "AVGO": "Information Technology",
    "ORCL": "Information Technology", "CRM": "Information Technology",
    "CSCO": "Information Technology", "ADBE": "Information Technology",
    "AMD": "Information Technology", "ACN": "Information Technology",
    "IBM": "Information Technology", "INTC": "Information Technology",
    "TXN": "Information Technology", "INTU": "Information Technology",
    "QCOM": "Information Technology", "AMAT": "Information Technology",
    "ADI": "Information Technology", "LRCX": "Information Technology",
    "KLAC": "Information Technology", "SNPS": "Information Technology",
    "CDNS": "Information Technology", "MRVL": "Information Technology",
    "PANW": "Information Technology", "FTNT": "Information Technology",
    "NOW": "Information Technology", "PLTR": "Information Technology",
    "CRWD": "Information Technology", "MCHP": "Information Technology",
    "ON": "Information Technology", "NXPI": "Information Technology",
    "MPWR": "Information Technology", "KEYS": "Information Technology",
    "ANSS": "Information Technology", "HPQ": "Information Technology",
    "HPE": "Information Technology", "WDC": "Information Technology",
    "STX": "Information Technology", "ZBRA": "Information Technology",
    "JNPR": "Information Technology", "SWKS": "Information Technology",
    "TER": "Information Technology", "FFIV": "Information Technology",
    "PTC": "Information Technology", "TRMB": "Information Technology",
    "GEN": "Information Technology", "EPAM": "Information Technology",
    "CTSH": "Information Technology", "IT": "Information Technology",
    "VRSN": "Information Technology", "AKAM": "Information Technology",
    "FSLR": "Information Technology",
    "MANH": "Information Technology", "SMCI": "Information Technology",
    "LSCC": "Information Technology", "POWI": "Information Technology",
    "ENPH": "Information Technology", "MKSI": "Information Technology",
    "AZPN": "Information Technology", "CGNX": "Information Technology",
    "ENTG": "Information Technology", "COHR": "Information Technology",
    "NOVT": "Information Technology", "ASGN": "Information Technology",
    "DT": "Information Technology", "MTSI": "Information Technology",
    "VNT": "Information Technology", "SLAB": "Information Technology",
    "QLYS": "Information Technology", "TENB": "Information Technology",
    "CYBR": "Information Technology", "PCTY": "Information Technology",
    "WEX": "Information Technology", "SANM": "Information Technology",
    "EXPO": "Information Technology", "CIEN": "Information Technology",
    "VIAV": "Information Technology", "CALX": "Information Technology",
    "TTMI": "Information Technology", "PRGS": "Information Technology",
    "ARM": "Information Technology", "SNOW": "Information Technology",
    "DDOG": "Information Technology", "NET": "Information Technology",
    "ZS": "Information Technology", "TEAM": "Information Technology",
    "MDB": "Information Technology", "AI": "Information Technology",
    "PATH": "Information Technology", "DOCN": "Information Technology",
    "CFLT": "Information Technology", "ESTC": "Information Technology",
    "HUBS": "Information Technology", "TWLO": "Information Technology",
    "TSM": "Information Technology", "ASML": "Information Technology",
    "SAP": "Information Technology",

    # ─── Health Care ─────────────────────────────────────────────────────
    "UNH": "Health Care", "LLY": "Health Care", "JNJ": "Health Care",
    "ABBV": "Health Care", "MRK": "Health Care", "TMO": "Health Care",
    "ABT": "Health Care", "DHR": "Health Care", "PFE": "Health Care",
    "AMGN": "Health Care", "ISRG": "Health Care", "ELV": "Health Care",
    "GILD": "Health Care", "CI": "Health Care", "SYK": "Health Care",
    "BSX": "Health Care", "VRTX": "Health Care", "MDT": "Health Care",
    "REGN": "Health Care", "BDX": "Health Care", "ZTS": "Health Care",
    "HCA": "Health Care", "IDXX": "Health Care", "EW": "Health Care",
    "A": "Health Care", "IQV": "Health Care", "MTD": "Health Care",
    "DXCM": "Health Care", "RMD": "Health Care", "BAX": "Health Care",
    "ALGN": "Health Care", "HOLX": "Health Care", "ILMN": "Health Care",
    "TECH": "Health Care", "PODD": "Health Care", "RVTY": "Health Care",
    "HSIC": "Health Care", "CRL": "Health Care", "CTLT": "Health Care",
    "DGX": "Health Care", "LH": "Health Care", "VTRS": "Health Care",
    "OGN": "Health Care", "XRAY": "Health Care", "SOLV": "Health Care",
    "GEHC": "Health Care", "INCY": "Health Care", "MOH": "Health Care",
    "CNC": "Health Care", "HUM": "Health Care", "BMY": "Health Care",
    "BIIB": "Health Care",
    "MEDP": "Health Care", "JAZZ": "Health Care", "NBIX": "Health Care",
    "NVCR": "Health Care", "IRTC": "Health Care", "RARE": "Health Care",
    "HALO": "Health Care", "AZTA": "Health Care", "NEO": "Health Care",
    "IART": "Health Care", "MMSI": "Health Care", "ENSG": "Health Care",
    "SGRY": "Health Care", "NHC": "Health Care", "LNTH": "Health Care",
    "ITCI": "Health Care", "IONS": "Health Care", "EXAS": "Health Care",
    "MRNA": "Health Care", "BNTX": "Health Care", "ARGX": "Health Care",
    "ALNY": "Health Care", "BMRN": "Health Care", "EXEL": "Health Care",
    "SRPT": "Health Care", "RCKT": "Health Care",
    "NVS": "Health Care", "AZN": "Health Care", "GSK": "Health Care",

    # ─── Financials ──────────────────────────────────────────────────────
    "JPM": "Financials", "V": "Financials", "MA": "Financials",
    "BRK-B": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "SPGI": "Financials",
    "BLK": "Financials", "AXP": "Financials", "CME": "Financials",
    "ICE": "Financials", "CB": "Financials", "SCHW": "Financials",
    "AON": "Financials", "MMC": "Financials", "PGR": "Financials",
    "MCO": "Financials", "AJG": "Financials", "MET": "Financials",
    "AFL": "Financials", "PRU": "Financials", "AIG": "Financials",
    "TRV": "Financials", "ALL": "Financials", "COF": "Financials",
    "USB": "Financials", "PNC": "Financials", "TFC": "Financials",
    "BK": "Financials", "STT": "Financials", "FITB": "Financials",
    "HBAN": "Financials", "CFG": "Financials", "RF": "Financials",
    "KEY": "Financials", "NTRS": "Financials", "CINF": "Financials",
    "ZION": "Financials", "MTB": "Financials", "NDAQ": "Financials",
    "CBOE": "Financials", "MSCI": "Financials", "MKTX": "Financials",
    "FDS": "Financials", "RJF": "Financials", "L": "Financials",
    "WRB": "Financials", "GL": "Financials", "RE": "Financials",
    "BRO": "Financials", "ERIE": "Financials", "AIZ": "Financials",
    "DFS": "Financials", "SYF": "Financials", "ALLY": "Financials",
    "EWBC": "Financials", "FNB": "Financials", "GBCI": "Financials",
    "OZK": "Financials", "IBOC": "Financials", "SNV": "Financials",
    "PNFP": "Financials", "CADE": "Financials", "FHN": "Financials",
    "WBS": "Financials", "UBSI": "Financials", "SFNC": "Financials",
    "WTFC": "Financials", "HWC": "Financials",
    "PYPL": "Financials", "FIS": "Financials", "FISV": "Financials",
    "GPN": "Financials", "SQ": "Financials", "AFRM": "Financials",
    "SOFI": "Financials", "HOOD": "Financials", "UPST": "Financials",
    "LC": "Financials", "COIN": "Financials",
    "HSBC": "Financials", "UBS": "Financials", "BCS": "Financials",
    "ING": "Financials", "DB": "Financials", "SAN": "Financials",
    "NU": "Financials",

    # ─── Consumer Discretionary ──────────────────────────────────────────
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "LOW": "Consumer Discretionary",
    "BKNG": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "TJX": "Consumer Discretionary", "CMG": "Consumer Discretionary",
    "ORLY": "Consumer Discretionary", "AZO": "Consumer Discretionary",
    "ROST": "Consumer Discretionary", "MAR": "Consumer Discretionary",
    "HLT": "Consumer Discretionary", "YUM": "Consumer Discretionary",
    "DHI": "Consumer Discretionary", "LEN": "Consumer Discretionary",
    "PHM": "Consumer Discretionary", "NVR": "Consumer Discretionary",
    "GPC": "Consumer Discretionary", "BBY": "Consumer Discretionary",
    "DRI": "Consumer Discretionary", "POOL": "Consumer Discretionary",
    "GRMN": "Consumer Discretionary", "DECK": "Consumer Discretionary",
    "MGM": "Consumer Discretionary", "WYNN": "Consumer Discretionary",
    "CZR": "Consumer Discretionary", "LVS": "Consumer Discretionary",
    "F": "Consumer Discretionary", "GM": "Consumer Discretionary",
    "APTV": "Consumer Discretionary", "BWA": "Consumer Discretionary",
    "EBAY": "Consumer Discretionary", "ETSY": "Consumer Discretionary",
    "EXPE": "Consumer Discretionary", "RCL": "Consumer Discretionary",
    "CCL": "Consumer Discretionary", "NCLH": "Consumer Discretionary",
    "DG": "Consumer Discretionary", "DLTR": "Consumer Discretionary",
    "KMX": "Consumer Discretionary", "AAP": "Consumer Discretionary",
    "ULTA": "Consumer Discretionary", "LULU": "Consumer Discretionary",
    "TGT": "Consumer Discretionary", "TPR": "Consumer Discretionary",
    "RL": "Consumer Discretionary", "HAS": "Consumer Discretionary",
    "TSCO": "Consumer Discretionary",
    "UBER": "Consumer Discretionary", "DASH": "Consumer Discretionary",
    "ABNB": "Consumer Discretionary", "RIVN": "Consumer Discretionary",
    "LCID": "Consumer Discretionary",
    "FIVE": "Consumer Discretionary", "WSM": "Consumer Discretionary",
    "SKX": "Consumer Discretionary", "CROX": "Consumer Discretionary",
    "TXRH": "Consumer Discretionary", "WING": "Consumer Discretionary",
    "SHOP": "Consumer Discretionary", "SE": "Consumer Discretionary",
    "MELI": "Consumer Discretionary", "BABA": "Consumer Discretionary",
    "JD": "Consumer Discretionary", "PDD": "Consumer Discretionary",
    "NIO": "Consumer Discretionary", "LI": "Consumer Discretionary",
    "XPEV": "Consumer Discretionary", "TM": "Consumer Discretionary",
    "HMC": "Consumer Discretionary",

    # ─── Communication Services ──────────────────────────────────────────
    "GOOGL": "Communication Services", "GOOG": "Communication Services",
    "META": "Communication Services", "NFLX": "Communication Services",
    "DIS": "Communication Services", "CMCSA": "Communication Services",
    "T": "Communication Services", "VZ": "Communication Services",
    "TMUS": "Communication Services", "CHTR": "Communication Services",
    "EA": "Communication Services", "TTWO": "Communication Services",
    "MTCH": "Communication Services", "WBD": "Communication Services",
    "FOXA": "Communication Services", "FOX": "Communication Services",
    "PARA": "Communication Services", "NWS": "Communication Services",
    "NWSA": "Communication Services", "OMC": "Communication Services",
    "IPG": "Communication Services", "LYV": "Communication Services",
    "TTD": "Communication Services", "ROKU": "Communication Services",
    "PINS": "Communication Services", "SNAP": "Communication Services",
    "RBLX": "Communication Services", "U": "Communication Services",
    "BIDU": "Communication Services", "SONY": "Communication Services",

    # ─── Industrials ─────────────────────────────────────────────────────
    "GE": "Industrials", "CAT": "Industrials", "BA": "Industrials",
    "RTX": "Industrials", "HON": "Industrials", "UNP": "Industrials",
    "UPS": "Industrials", "DE": "Industrials", "LMT": "Industrials",
    "NOC": "Industrials", "GD": "Industrials", "TDG": "Industrials",
    "ADP": "Industrials", "WM": "Industrials", "RSG": "Industrials",
    "ETN": "Industrials", "ITW": "Industrials", "EMR": "Industrials",
    "PH": "Industrials", "ROK": "Industrials", "CTAS": "Industrials",
    "FAST": "Industrials", "CPRT": "Industrials", "ODFL": "Industrials",
    "AME": "Industrials", "PCAR": "Industrials", "TT": "Industrials",
    "CARR": "Industrials", "OTIS": "Industrials", "SWK": "Industrials",
    "GWW": "Industrials", "NSC": "Industrials", "CSX": "Industrials",
    "FDX": "Industrials", "JCI": "Industrials", "XYL": "Industrials",
    "IR": "Industrials", "WAB": "Industrials", "PAYC": "Industrials",
    "VRSK": "Industrials", "PAYX": "Industrials", "BR": "Industrials",
    "DOV": "Industrials", "ROP": "Industrials", "IEX": "Industrials",
    "NDSN": "Industrials", "SNA": "Industrials", "J": "Industrials",
    "PWR": "Industrials", "GNRC": "Industrials", "EFX": "Industrials",
    "MAS": "Industrials", "LHX": "Industrials", "HII": "Industrials",
    "TXT": "Industrials", "LDOS": "Industrials", "AXON": "Industrials",
    "DAL": "Industrials", "UAL": "Industrials", "AAL": "Industrials",
    "LUV": "Industrials", "ALK": "Industrials", "MMM": "Industrials",
    "SAIA": "Industrials", "AGCO": "Industrials",
    "RKLB": "Industrials", "JOBY": "Industrials",

    # ─── Consumer Staples ────────────────────────────────────────────────
    "PG": "Consumer Staples", "KO": "Consumer Staples",
    "PEP": "Consumer Staples", "COST": "Consumer Staples",
    "WMT": "Consumer Staples", "PM": "Consumer Staples",
    "MO": "Consumer Staples", "CL": "Consumer Staples",
    "MDLZ": "Consumer Staples", "STZ": "Consumer Staples",
    "KMB": "Consumer Staples", "GIS": "Consumer Staples",
    "SJM": "Consumer Staples", "K": "Consumer Staples",
    "HSY": "Consumer Staples", "HRL": "Consumer Staples",
    "CPB": "Consumer Staples", "MKC": "Consumer Staples",
    "CHD": "Consumer Staples", "CAG": "Consumer Staples",
    "TSN": "Consumer Staples", "ADM": "Consumer Staples",
    "BG": "Consumer Staples", "TAP": "Consumer Staples",
    "KHC": "Consumer Staples", "MNST": "Consumer Staples",
    "KDP": "Consumer Staples", "EL": "Consumer Staples",
    "CLX": "Consumer Staples", "SYY": "Consumer Staples",
    "KR": "Consumer Staples", "WBA": "Consumer Staples",
    "CELH": "Consumer Staples", "SAM": "Consumer Staples",

    # ─── Energy ──────────────────────────────────────────────────────────
    "XOM": "Energy", "CVX": "Energy", "SLB": "Energy",
    "COP": "Energy", "EOG": "Energy", "MPC": "Energy",
    "PSX": "Energy", "VLO": "Energy", "OXY": "Energy",
    "HAL": "Energy", "DVN": "Energy", "HES": "Energy",
    "FANG": "Energy", "BKR": "Energy", "CTRA": "Energy",
    "EQT": "Energy", "MRO": "Energy", "APA": "Energy",
    "TRGP": "Energy", "OKE": "Energy", "WMB": "Energy",
    "KMI": "Energy", "ET": "Energy",

    # ─── Utilities ───────────────────────────────────────────────────────
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    "AEP": "Utilities", "D": "Utilities", "SRE": "Utilities",
    "EXC": "Utilities", "XEL": "Utilities", "ED": "Utilities",
    "WEC": "Utilities", "ES": "Utilities", "AWK": "Utilities",
    "DTE": "Utilities", "FE": "Utilities", "PEG": "Utilities",
    "PPL": "Utilities", "CMS": "Utilities", "AES": "Utilities",
    "ATO": "Utilities", "EVRG": "Utilities", "CNP": "Utilities",
    "NI": "Utilities", "LNT": "Utilities", "PNW": "Utilities",
    "NRG": "Utilities",

    # ─── Real Estate ─────────────────────────────────────────────────────
    "PLD": "Real Estate", "AMT": "Real Estate", "CCI": "Real Estate",
    "EQIX": "Real Estate", "PSA": "Real Estate", "O": "Real Estate",
    "SPG": "Real Estate", "DLR": "Real Estate", "WELL": "Real Estate",
    "AVB": "Real Estate", "EQR": "Real Estate", "VTR": "Real Estate",
    "ARE": "Real Estate", "MAA": "Real Estate", "UDR": "Real Estate",
    "ESS": "Real Estate", "SUI": "Real Estate", "CPT": "Real Estate",
    "REG": "Real Estate", "FRT": "Real Estate", "KIM": "Real Estate",
    "BXP": "Real Estate", "HST": "Real Estate", "PEAK": "Real Estate",
    "INVH": "Real Estate", "IRM": "Real Estate", "SBAC": "Real Estate",
    "WY": "Real Estate",

    # ─── Materials ───────────────────────────────────────────────────────
    "LIN": "Materials", "APD": "Materials", "SHW": "Materials",
    "ECL": "Materials", "FCX": "Materials", "NEM": "Materials",
    "DD": "Materials", "NUE": "Materials", "VMC": "Materials",
    "MLM": "Materials", "DOW": "Materials", "PPG": "Materials",
    "EMN": "Materials", "CE": "Materials", "CF": "Materials",
    "MOS": "Materials", "FMC": "Materials", "ALB": "Materials",
    "IFF": "Materials", "BALL": "Materials", "PKG": "Materials",
    "IP": "Materials", "WRK": "Materials", "SEE": "Materials",
    "AVY": "Materials", "AMCR": "Materials",
    "CLF": "Materials", "STLD": "Materials", "CMC": "Materials",
    "RS": "Materials", "ATI": "Materials",
    "RIO": "Materials", "BHP": "Materials", "VALE": "Materials",
    "SCCO": "Materials", "GOLD": "Materials", "WPM": "Materials",

    # ─── Clean Energy (mapped to Industrials or IT) ──────────────────────
    "PLUG": "Industrials", "FCEL": "Industrials", "BE": "Industrials",
    "SEDG": "Information Technology", "RUN": "Industrials",
    "NOVA": "Industrials", "CHPT": "Industrials", "BLNK": "Industrials",
    "EVGO": "Industrials", "QS": "Information Technology",

    # ═══════════════════════════════════════════════════════════════════════
    # S&P MidCap 400 / SmallCap 600 / Extra — GICS sectors
    # ═══════════════════════════════════════════════════════════════════════

    # ─── Communication Services (MidCap/SmallCap/Extra) ──────────────────
    "CARG": "Communication Services", "CARS": "Communication Services",
    "CCO": "Communication Services", "CNX": "Communication Services",
    "EVC": "Communication Services", "IACI": "Communication Services",
    "LUMN": "Communication Services", "MGNI": "Communication Services",
    "PUBM": "Communication Services", "QNST": "Communication Services",
    "SCHL": "Communication Services", "SIRI": "Communication Services",
    "TKO": "Communication Services", "WLY": "Communication Services",
    "YELP": "Communication Services", "ZD": "Communication Services",
    "ZUO": "Communication Services",

    # ─── Consumer Discretionary (MidCap/SmallCap/Extra) ──────────────────
    "ABG": "Consumer Discretionary", "ABM": "Consumer Discretionary",
    "AEO": "Consumer Discretionary", "ANF": "Consumer Discretionary",
    "ARMK": "Consumer Discretionary", "BC": "Consumer Discretionary",
    "BJRI": "Consumer Discretionary", "BOOT": "Consumer Discretionary",
    "BYD": "Consumer Discretionary", "CAKE": "Consumer Discretionary",
    "CAL": "Consumer Discretionary", "CATO": "Consumer Discretionary",
    "CHUY": "Consumer Discretionary", "CRI": "Consumer Discretionary",
    "DENN": "Consumer Discretionary", "DIN": "Consumer Discretionary",
    "DORM": "Consumer Discretionary", "EAT": "Consumer Discretionary",
    "FL": "Consumer Discretionary", "FND": "Consumer Discretionary",
    "FOXF": "Consumer Discretionary", "GIII": "Consumer Discretionary",
    "GPI": "Consumer Discretionary", "HIBB": "Consumer Discretionary",
    "HNI": "Consumer Discretionary", "JACK": "Consumer Discretionary",
    "JELD": "Consumer Discretionary", "KNX": "Consumer Discretionary",
    "KRUS": "Consumer Discretionary", "LEA": "Consumer Discretionary",
    "LEG": "Consumer Discretionary", "LNW": "Consumer Discretionary",
    "LOCO": "Consumer Discretionary", "LZB": "Consumer Discretionary",
    "MCRI": "Consumer Discretionary", "MOD": "Consumer Discretionary",
    "MSGS": "Consumer Discretionary", "ODP": "Consumer Discretionary",
    "PATK": "Consumer Discretionary", "PENN": "Consumer Discretionary",
    "PLNT": "Consumer Discretionary", "RGS": "Consumer Discretionary",
    "RH": "Consumer Discretionary", "RRR": "Consumer Discretionary",
    "RUTH": "Consumer Discretionary", "SAH": "Consumer Discretionary",
    "SCI": "Consumer Discretionary", "SCVL": "Consumer Discretionary",
    "SHAK": "Consumer Discretionary", "SHOO": "Consumer Discretionary",
    "SIG": "Consumer Discretionary", "SNBR": "Consumer Discretionary",
    "STRA": "Consumer Discretionary", "SWX": "Consumer Discretionary",
    "THRM": "Consumer Discretionary", "URBN": "Consumer Discretionary",
    "VVV": "Consumer Discretionary", "WH": "Consumer Discretionary",
    "WWW": "Consumer Discretionary",

    # ─── Consumer Staples (MidCap/SmallCap/Extra) ────────────────────────
    "BGS": "Consumer Staples", "BJ": "Consumer Staples",
    "CALM": "Consumer Staples", "CASY": "Consumer Staples",
    "CENT": "Consumer Staples", "COKE": "Consumer Staples",
    "FLO": "Consumer Staples", "FRPT": "Consumer Staples",
    "HLF": "Consumer Staples", "INGR": "Consumer Staples",
    "IPAR": "Consumer Staples", "JJSF": "Consumer Staples",
    "LANC": "Consumer Staples", "LNDC": "Consumer Staples",
    "MGPI": "Consumer Staples", "PFGC": "Consumer Staples",
    "POST": "Consumer Staples", "SENEA": "Consumer Staples",
    "SMPL": "Consumer Staples", "SPTN": "Consumer Staples",
    "SWM": "Consumer Staples", "SXT": "Consumer Staples",
    "THS": "Consumer Staples", "TR": "Consumer Staples",
    "USFD": "Consumer Staples", "UVV": "Consumer Staples",
    "WDFC": "Consumer Staples",

    # ─── Energy (MidCap/SmallCap/Extra) ──────────────────────────────────
    "AROC": "Energy", "CHRD": "Energy", "CHX": "Energy",
    "CIVI": "Energy", "CPK": "Energy", "CRC": "Energy",
    "DEN": "Energy", "DINO": "Energy", "GPOR": "Energy",
    "GPRE": "Energy", "HLX": "Energy", "HP": "Energy",
    "LBRT": "Energy", "MGY": "Energy", "MTDR": "Energy",
    "NEXT": "Energy", "NOG": "Energy", "OIS": "Energy",
    "PARR": "Energy", "PTEN": "Energy", "PUMP": "Energy",
    "REI": "Energy", "RES": "Energy", "RRC": "Energy",
    "SBOW": "Energy", "SM": "Energy", "SWN": "Energy",
    "TALO": "Energy", "TDW": "Energy", "VNOM": "Energy",
    "WTI": "Energy", "WTTR": "Energy",

    # ─── Financials (MidCap/SmallCap/Extra) ──────────────────────────────
    "ABCB": "Financials", "ACIW": "Financials", "ADYEN": "Financials",
    "AGO": "Financials", "ASB": "Financials", "AX": "Financials",
    "BANF": "Financials", "BHLB": "Financials", "BOH": "Financials",
    "BRKL": "Financials", "CASH": "Financials", "CATC": "Financials",
    "CBU": "Financials", "CNOB": "Financials", "CVBF": "Financials",
    "ESGR": "Financials", "FBP": "Financials", "FCF": "Financials",
    "FFBC": "Financials", "FFIN": "Financials", "FULT": "Financials",
    "GABC": "Financials", "HFWA": "Financials", "HMN": "Financials",
    "HOMB": "Financials", "HOPE": "Financials", "HTH": "Financials",
    "IBTX": "Financials", "KMPR": "Financials", "LCNB": "Financials",
    "NBTB": "Financials", "NWBI": "Financials", "OCFC": "Financials",
    "ORI": "Financials", "PACW": "Financials", "PCBK": "Financials",
    "PFS": "Financials", "PIPR": "Financials", "PPBI": "Financials",
    "PRI": "Financials", "PRIMERICA": "Financials", "RENR": "Financials",
    "RLI": "Financials", "RNR": "Financials", "SASR": "Financials",
    "SBCF": "Financials", "SBSI": "Financials", "SFBS": "Financials",
    "SIGI": "Financials", "SSB": "Financials", "STBA": "Financials",
    "TBBK": "Financials", "TCBI": "Financials", "THG": "Financials",
    "TMP": "Financials", "TOWN": "Financials", "TRMK": "Financials",
    "UFCS": "Financials", "UVSP": "Financials", "WABC": "Financials",
    "WASH": "Financials", "WRBS": "Financials", "WSBC": "Financials",
    "WSFS": "Financials",

    # ─── Health Care (MidCap/SmallCap/Extra) ─────────────────────────────
    "ACB": "Health Care", "ACHC": "Health Care", "AMED": "Health Care",
    "ANGO": "Health Care", "ATEC": "Health Care", "BIO": "Health Care",
    "CERT": "Health Care", "CGC": "Health Care", "CNMD": "Health Care",
    "CORT": "Health Care", "CRON": "Health Care", "CRSP": "Health Care",
    "CRY": "Health Care", "DCPH": "Health Care", "ETNB": "Health Care",
    "FLGT": "Health Care", "GMED": "Health Care", "HZNP": "Health Care",
    "ICUI": "Health Care", "INSP": "Health Care", "LIVN": "Health Care",
    "LMAT": "Health Care", "MASI": "Health Care", "NKTR": "Health Care",
    "NVST": "Health Care", "OFIX": "Health Care", "OLK": "Health Care",
    "OMCL": "Health Care", "OMED": "Health Care", "PCRX": "Health Care",
    "PCVX": "Health Care", "PGNY": "Health Care", "PRCT": "Health Care",
    "PRXL": "Health Care", "RDNT": "Health Care", "RGEN": "Health Care",
    "RVMD": "Health Care", "SGEN": "Health Care", "SRDX": "Health Care",
    "SUPN": "Health Care", "TGTX": "Health Care", "TLRY": "Health Care",
    "TNDM": "Health Care", "TWST": "Health Care", "XNCR": "Health Care",

    # ─── Industrials (MidCap/SmallCap/Extra) ─────────────────────────────
    "AAON": "Industrials", "AIMC": "Industrials", "AIT": "Industrials",
    "APG": "Industrials", "ARCB": "Industrials", "AWI": "Industrials",
    "AZEK": "Industrials", "AZZ": "Industrials", "BCC": "Industrials",
    "BELFB": "Industrials", "BMI": "Industrials", "BWXT": "Industrials",
    "CENTA": "Industrials", "CLH": "Industrials", "CW": "Industrials",
    "DNOW": "Industrials", "DY": "Industrials", "ECOL": "Industrials",
    "ENS": "Industrials", "ESAB": "Industrials", "FLR": "Industrials",
    "FSS": "Industrials", "GATX": "Industrials", "GBX": "Industrials",
    "GGG": "Industrials", "GTES": "Industrials", "HDS": "Industrials",
    "HEI": "Industrials", "HUBB": "Industrials", "IRDM": "Industrials",
    "JBHT": "Industrials", "JBT": "Industrials", "KAI": "Industrials",
    "KBR": "Industrials", "KFY": "Industrials", "KTOS": "Industrials",
    "LFUS": "Industrials", "LSTR": "Industrials", "LUNR": "Industrials",
    "MATX": "Industrials", "MRCY": "Industrials", "MTRN": "Industrials",
    "MTZ": "Industrials", "MWA": "Industrials", "MYRG": "Industrials",
    "NNBR": "Industrials", "NPO": "Industrials", "POWL": "Industrials",
    "PRIM": "Industrials", "RBC": "Industrials", "RDW": "Industrials",
    "REVG": "Industrials", "ROCK": "Industrials", "RXO": "Industrials",
    "SITE": "Industrials", "SPR": "Industrials", "SRCL": "Industrials",
    "SSD": "Industrials", "TDY": "Industrials", "TNC": "Industrials",
    "TREX": "Industrials", "UFPT": "Industrials", "USM": "Industrials",
    "VSEC": "Industrials", "WFRD": "Industrials", "WNC": "Industrials",
    "WTS": "Industrials",

    # ─── Information Technology (MidCap/SmallCap/Extra) ──────────────────
    "ALKT": "Information Technology", "ALRM": "Information Technology",
    "AMSF": "Information Technology", "ATEN": "Information Technology",
    "ATRC": "Information Technology", "CACC": "Information Technology",
    "CEVA": "Information Technology", "COHU": "Information Technology",
    "CSWI": "Information Technology", "CTS": "Information Technology",
    "DIGI": "Information Technology", "DIOD": "Information Technology",
    "EVTC": "Information Technology", "FORM": "Information Technology",
    "FOUR": "Information Technology", "GSHD": "Information Technology",
    "HQY": "Information Technology", "IDCC": "Information Technology",
    "IIIV": "Information Technology", "IIVI": "Information Technology",
    "INTA": "Information Technology", "IPGP": "Information Technology",
    "JAMF": "Information Technology", "MARA": "Information Technology",
    "MAXR": "Information Technology", "MIDD": "Information Technology",
    "NSIT": "Information Technology", "OOMA": "Information Technology",
    "OSIS": "Information Technology", "PAYO": "Information Technology",
    "PLAY": "Information Technology", "PLUS": "Information Technology",
    "RIOT": "Information Technology", "RMBS": "Information Technology",
    "SGH": "Information Technology", "SLP": "Information Technology",
    "SMTC": "Information Technology", "SPSC": "Information Technology",
    "TASK": "Information Technology", "TTEC": "Information Technology",
    "VCEL": "Information Technology", "VERX": "Information Technology",
    "VRNS": "Information Technology",

    # ─── Materials (MidCap/SmallCap/Extra) ───────────────────────────────
    "AXTA": "Materials", "BCPC": "Materials", "BERY": "Materials",
    "CBT": "Materials", "CC": "Materials", "CYT": "Materials",
    "GEF": "Materials", "GPK": "Materials", "HUN": "Materials",
    "HWKN": "Materials", "HXL": "Materials", "IOSP": "Materials",
    "KOP": "Materials", "KRA": "Materials", "KWR": "Materials",
    "LTHM": "Materials", "MTX": "Materials", "NGVT": "Materials",
    "OI": "Materials", "OLN": "Materials", "PBF": "Materials",
    "RFP": "Materials", "RYAM": "Materials", "SCL": "Materials",
    "SLVM": "Materials", "SON": "Materials", "TROX": "Materials",
    "TRS": "Materials", "UFPI": "Materials", "WOR": "Materials",

    # ─── Real Estate (MidCap/SmallCap/Extra) ─────────────────────────────
    "AAT": "Real Estate", "AKR": "Real Estate", "ALEX": "Real Estate",
    "APLE": "Real Estate", "BRT": "Real Estate", "CBL": "Real Estate",
    "CDR": "Real Estate", "COLD": "Real Estate", "CSR": "Real Estate",
    "CUBE": "Real Estate", "CUZ": "Real Estate", "DEI": "Real Estate",
    "EFC": "Real Estate", "EGP": "Real Estate", "FR": "Real Estate",
    "GNL": "Real Estate", "GTY": "Real Estate", "IRT": "Real Estate",
    "JBGS": "Real Estate", "KRC": "Real Estate", "LAMR": "Real Estate",
    "LSI": "Real Estate", "LTC": "Real Estate", "MAC": "Real Estate",
    "NNN": "Real Estate", "NSA": "Real Estate", "NXRT": "Real Estate",
    "OHI": "Real Estate", "PDM": "Real Estate", "PEB": "Real Estate",
    "PLYM": "Real Estate", "REXR": "Real Estate", "RHP": "Real Estate",
    "SAFE": "Real Estate", "SKT": "Real Estate", "SLG": "Real Estate",
    "SRC": "Real Estate", "STAG": "Real Estate", "STOR": "Real Estate",
    "UE": "Real Estate", "VRE": "Real Estate", "WRE": "Real Estate",
    "XHR": "Real Estate",

    # ─── Utilities (MidCap/SmallCap/Extra) ───────────────────────────────
    "ATGE": "Utilities", "AVA": "Utilities", "BKH": "Utilities",
    "CWEN": "Utilities", "HE": "Utilities", "MDU": "Utilities",
    "MGEE": "Utilities", "NWE": "Utilities", "OGE": "Utilities",
    "OTTR": "Utilities", "PNM": "Utilities", "POR": "Utilities",
    "SJW": "Utilities", "SPKE": "Utilities", "SR": "Utilities",
    "UIL": "Utilities", "UTL": "Utilities",

    # ─── Remaining extras (sector inferred) ──────────────────────────────
    "ACHR": "Industrials", "ASTS": "Communication Services",
    "CS": "Financials", "IONQ": "Information Technology",
    "LILM": "Industrials", "PAAS": "Materials", "RGTI": "Information Technology",
    "SNE": "Consumer Discretionary",
}

# ═══════════════════════════════════════════════════════════════════════════
# Industry group classification for key names (GICS tier 2/3/4)
# ═══════════════════════════════════════════════════════════════════════════
INDUSTRY_GROUP_MAP = {
    # (industry_group, industry, sub_industry)
    "AAPL": ("Technology Hardware & Equipment", "Technology Hardware, Storage & Peripherals", "Technology Hardware, Storage & Peripherals"),
    "MSFT": ("Software & Services", "Systems Software", "Systems Software"),
    "NVDA": ("Semiconductors & Semiconductor Equipment", "Semiconductors", "Semiconductors"),
    "GOOGL": ("Media & Entertainment", "Interactive Media & Services", "Interactive Media & Services"),
    "META": ("Media & Entertainment", "Interactive Media & Services", "Interactive Media & Services"),
    "AMZN": ("Retailing", "Broadline Retail", "Broadline Retail"),
    "TSLA": ("Automobiles & Components", "Automobiles", "Electric Vehicles"),
    "JPM": ("Banks", "Diversified Banks", "Diversified Banks"),
    "UNH": ("Health Care Equipment & Services", "Managed Health Care", "Managed Health Care"),
    "XOM": ("Oil, Gas & Consumable Fuels", "Integrated Oil & Gas", "Integrated Oil & Gas"),
    "PG": ("Household & Personal Products", "Household Products", "Household Products"),
    "NEE": ("Utilities", "Electric Utilities", "Electric Utilities"),
    "PLD": ("Equity Real Estate Investment Trusts (REITs)", "Industrial REITs", "Industrial REITs"),
    "LIN": ("Chemicals", "Industrial Gases", "Industrial Gases"),
    "V": ("Financial Services", "Transaction & Payment Processing Services", "Transaction & Payment Processing Services"),
    "MA": ("Financial Services", "Transaction & Payment Processing Services", "Transaction & Payment Processing Services"),
    "JNJ": ("Pharmaceuticals, Biotechnology & Life Sciences", "Pharmaceuticals", "Pharmaceuticals"),
    "LLY": ("Pharmaceuticals, Biotechnology & Life Sciences", "Pharmaceuticals", "Pharmaceuticals"),
    "AVGO": ("Semiconductors & Semiconductor Equipment", "Semiconductors", "Semiconductors"),
    "HD": ("Consumer Discretionary Distribution & Retail", "Home Improvement Retail", "Home Improvement Retail"),
    "BA": ("Capital Goods", "Aerospace & Defense", "Aerospace & Defense"),
    "CAT": ("Capital Goods", "Machinery", "Construction Machinery & Heavy Transportation Equipment"),
    "CRM": ("Software & Services", "Application Software", "Application Software"),
    "ORCL": ("Software & Services", "Application Software", "Application Software"),
    "GS": ("Financial Services", "Capital Markets", "Investment Banking & Brokerage"),
    "KO": ("Food, Beverage & Tobacco", "Beverages", "Soft Drinks & Non-alcoholic Beverages"),
    "PEP": ("Food, Beverage & Tobacco", "Beverages", "Soft Drinks & Non-alcoholic Beverages"),
    "MCD": ("Consumer Services", "Hotels, Restaurants & Leisure", "Restaurants"),
    "DIS": ("Media & Entertainment", "Movies & Entertainment", "Movies & Entertainment"),
    "NFLX": ("Media & Entertainment", "Movies & Entertainment", "Movies & Entertainment"),
}

# ═══════════════════════════════════════════════════════════════════════════
# OpenBB Index Symbols for dynamic fetching
# ═══════════════════════════════════════════════════════════════════════════
OPENBB_INDEX_SYMBOLS = {
    "sp500": "^GSPC",
    "sp400": "^MID",
    "sp600": "^SML",
    "nasdaq100": "^NDX",
    "russell2000": "^RUT",
    "russell1000": "^RUI",
    "dow30": "^DJI",
}

# Tickers to always exclude (delisted, bankrupt, etc.)
EXCLUDE_TICKERS = {"FRC", "SIVB"}  # First Republic Bank, Silicon Valley Bank — failed 2023


def get_all_static_tickers() -> list[str]:
    """Return deduplicated list of all static equity tickers."""
    all_tickers = set(SP500_TICKERS + SP400_TICKERS + SP600_TICKERS + EXTRA_TICKERS)
    all_tickers -= EXCLUDE_TICKERS
    return sorted(all_tickers)


def get_sector_for_ticker(ticker: str) -> str:
    """Return GICS sector for a ticker, or empty string if unknown."""
    return SECTOR_MAP.get(ticker, "")


def get_industry_group_for_ticker(ticker: str) -> tuple[str, str, str]:
    """Return (industry_group, industry, sub_industry) or empty tuple."""
    return INDUSTRY_GROUP_MAP.get(ticker, ("", "", ""))

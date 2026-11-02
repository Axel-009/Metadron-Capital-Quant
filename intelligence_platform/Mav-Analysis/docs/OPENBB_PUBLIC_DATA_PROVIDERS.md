# OpenBB Public Data Providers

All macroeconomic and FRED data in Maverick-MCP is routed through the **OpenBB Platform API** (port 6900) using the `openbb-fred` provider. This centralizes data access through a unified gateway.

## Architecture

```
Maverick-MCP (macro_data.py)
    └── OpenBBFredClient (HTTP)
         └── OpenBB Platform API (localhost:6900)
              └── openbb-fred provider
                   └── FRED API (api.stlouisfed.org)
```

## Public Data Providers

These OpenBB extensions provide free public data. Sources are public agencies with open APIs. Some require registration, but all are free.

| Extension | Description | Install Command | API Key Name |
|-----------|-------------|----------------|--------------|
| openbb-bls | **Bureau of Labor Statistics** data connector | `pip install openbb-bls` | `bls_api_key` |
| openbb-congress-gov | **United States Congress** data connector | `pip install openbb-congress-gov` | `congress_gov_api_key` |
| openbb-cftc | **Commodity Futures Trading Commission** data connector | `pip install openbb-cftc` | `cftc_app_token` |
| openbb-ecb | **ECB** data connector | `pip install openbb-ecb` | - |
| openbb-imf | **IMF** data connector | `pip install openbb-imf` | - |
| openbb-federal-reserve | **Federal Reserve** data connector | `pip install openbb-federal-reserve` | - |
| openbb-fred | **FRED** data connector | `pip install openbb-fred` | `fred_api_key` |
| openbb-government-us | **US Government** data connector | `pip install openbb-us-government` | - |
| openbb-oecd | **OECD** data connector | `pip install openbb-oecd` | - |
| openbb-polygon | **Polygon** data connector | `pip install openbb-polygon` | `polygon_api_key` |
| openbb-sec | **SEC** data connector | `pip install openbb-sec` | - |
| openbb-us-eia | **U.S. Energy Information Administration (EIA)** data connector | `pip install openbb-us-eia` | Free |

## FRED Series Used by Maverick-MCP

The macro data engine uses the following FRED series routed through OpenBB:

| Series ID | Description | Used For |
|-----------|-------------|----------|
| `SP500` | S&P 500 Index | Market performance, momentum |
| `NASDAQ100` | NASDAQ-100 Index | Market performance, momentum |
| `VIXCLS` | CBOE Volatility Index (VIX) | Risk/fear gauge |
| `A191RL1Q225SBEA` | Real GDP Growth Rate (quarterly) | GDP growth |
| `UNRATE` | Unemployment Rate | Labor market |
| `CPILFESL` | Core CPI (All Items Less Food & Energy) | Inflation rate |
| `DTWEXBGS` | Broad USD Index | USD momentum |
| `NASDAQCOM` | NASDAQ Composite | Historical bounds |

## Configuration

### Environment Variables

```bash
# FRED API key (passed to OpenBB for FRED authentication)
FRED_API_KEY=your_fred_api_key

# OpenBB Platform API connection (defaults shown)
OPENBB_API_HOST=127.0.0.1
OPENBB_API_PORT=6900

# Override the full base URL if needed
OPENBB_BASE_URL=http://127.0.0.1:6900

# Optional: OpenBB Platform authentication
OPENBB_API_USERNAME=
OPENBB_API_PASSWORD=
```

### OpenBB Platform API Endpoints Used

| Endpoint | Description |
|----------|-------------|
| `GET /api/v1/economy/fred_series` | Fetch FRED time series data |
| `GET /api/v1/economy/fred_search` | Search FRED series metadata |

### Example API Call

```bash
# Fetch S&P 500 data via OpenBB Platform
curl "http://localhost:6900/api/v1/economy/fred_series?symbol=SP500&start_date=2024-01-01&provider=fred"
```

## OpenBB FRED Provider Features

The OpenBB FRED provider supports 28 specialized fetchers including:

- **Interest Rates**: Ameribor, SOFR, SONIA, Federal Funds Rate, IORB, DWPCR
- **Treasury**: Constant maturity yields, T-bills, TIPS
- **Corporate Bonds**: ICE BofA indices, Moody's, high-quality market
- **Consumer Prices**: CPI, PCE, retail prices
- **Labor**: Non-farm payrolls, unemployment claims
- **Other**: Yield curves, commodity spot prices, mortgage indices, equity indices

All accessible through the generic `fred_series` endpoint with the appropriate series ID.

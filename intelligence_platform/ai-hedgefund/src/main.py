import sys
import time
import logging

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from colorama import Fore, Style, init
import questionary
from src.agents.portfolio_manager import portfolio_management_agent
from src.agents.risk_manager import risk_management_agent
from src.graph.state import AgentState
from src.utils.display import print_trading_output
from src.utils.analysts import ANALYST_ORDER, get_analyst_nodes
from src.utils.progress import progress
from src.utils.visualize import save_graph_as_png
from src.cli.input import (
    parse_cli_inputs,
)

# Unified data & strategy imports
from src.data.openbb_universe import (
    get_full_universe,
    get_macro_data,
    get_news_sentiment,
    UniverseData,
)
from src.strategy.multi_horizon import (
    MultiHorizonEngine,
    TradeHorizon,
    TradeThesis,
)
from src.execution.hft_engine import HFTExecutionEngine
from src.reporting.daily_report import DailyReportGenerator

import argparse
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import json

# Load environment variables from .env file
load_dotenv()

init(autoreset=True)

logger = logging.getLogger(__name__)


def parse_hedge_fund_response(response):
    """Parses a JSON string and returns a dictionary."""
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}\nResponse: {repr(response)}")
        return None
    except TypeError as e:
        print(f"Invalid response type (expected string, got {type(response).__name__}): {e}")
        return None
    except Exception as e:
        print(f"Unexpected error while parsing response: {e}\nResponse: {repr(response)}")
        return None


##### Run the Hedge Fund #####
def run_hedge_fund(
    tickers: list[str],
    start_date: str,
    end_date: str,
    portfolio: dict,
    show_reasoning: bool = False,
    selected_analysts: list[str] = [],
    model_name: str = "gpt-4.1",
    model_provider: str = "OpenAI",
):
    # Start progress tracking
    progress.start()

    try:
        # Build workflow (default to all analysts when none provided)
        workflow = create_workflow(selected_analysts if selected_analysts else None)
        agent = workflow.compile()

        final_state = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Make trading decisions based on the provided data.",
                    )
                ],
                "data": {
                    "tickers": tickers,
                    "portfolio": portfolio,
                    "start_date": start_date,
                    "end_date": end_date,
                    "analyst_signals": {},
                },
                "metadata": {
                    "show_reasoning": show_reasoning,
                    "model_name": model_name,
                    "model_provider": model_provider,
                },
            },
        )

        return {
            "decisions": parse_hedge_fund_response(final_state["messages"][-1].content),
            "analyst_signals": final_state["data"]["analyst_signals"],
        }
    finally:
        # Stop progress tracking
        progress.stop()


def start(state: AgentState):
    """Initialize the workflow with the input message."""
    return state


def create_workflow(selected_analysts=None):
    """Create the workflow with selected analysts."""
    workflow = StateGraph(AgentState)
    workflow.add_node("start_node", start)

    # Get analyst nodes from the configuration
    analyst_nodes = get_analyst_nodes()

    # Default to all analysts if none selected
    if selected_analysts is None:
        selected_analysts = list(analyst_nodes.keys())
    # Add selected analyst nodes
    for analyst_key in selected_analysts:
        node_name, node_func = analyst_nodes[analyst_key]
        workflow.add_node(node_name, node_func)
        workflow.add_edge("start_node", node_name)

    # Always add risk and portfolio management
    workflow.add_node("risk_management_agent", risk_management_agent)
    workflow.add_node("portfolio_manager", portfolio_management_agent)

    # Connect selected analysts to risk management
    for analyst_key in selected_analysts:
        node_name = analyst_nodes[analyst_key][0]
        workflow.add_edge(node_name, "risk_management_agent")

    workflow.add_edge("risk_management_agent", "portfolio_manager")
    workflow.add_edge("portfolio_manager", END)

    workflow.set_entry_point("start_node")
    return workflow


# ===================================================================
# Unified Multi-Horizon Trading System
# ===================================================================
def run_unified_system(
    start_date: str,
    end_date: str,
    initial_capital: float = 1_000_000.0,
    run_hft: bool = True,
    generate_reports: bool = True,
    hft_scan_interval_seconds: int = 60,
    max_hft_iterations: int = 10,
):
    """
    Run the unified investment strategy system.

    Workflow:
    1. Pull full universe via OpenBB (sole data source)
    2. Fetch macro data and news sentiment
    3. Run multi-horizon scanning (HFT, swing, medium, long term)
    4. Execute HFT trades continuously
    5. Generate daily reports
    6. Separate buy-and-hold recommendations for medium/long term

    Parameters
    ----------
    start_date : str
        Start date 'YYYY-MM-DD'.
    end_date : str
        End date 'YYYY-MM-DD'.
    initial_capital : float
        Starting portfolio value.
    run_hft : bool
        Whether to run HFT execution loop.
    generate_reports : bool
        Whether to generate daily reports.
    hft_scan_interval_seconds : int
        Seconds between HFT scans.
    max_hft_iterations : int
        Maximum number of HFT scan iterations.

    Returns
    -------
    dict
        Complete system output with all reports and recommendations.
    """
    print(f"{Fore.CYAN}{'='*60}")
    print(f"  UNIFIED MULTI-HORIZON TRADING SYSTEM")
    print(f"  Capital: ${initial_capital:,.2f}")
    print(f"  Period: {start_date} to {end_date}")
    print(f"{'='*60}{Style.RESET_ALL}")

    # ---- Step 1: Pull full universe (OpenBB as sole data source) ----
    print(f"\n{Fore.YELLOW}[1/6] Fetching full universe via OpenBB...{Style.RESET_ALL}")
    try:
        universe = get_full_universe(start_date, end_date)
        print(
            f"  Fetched: {len(universe.equities)} equities, "
            f"{len(universe.bonds)} bonds, {len(universe.commodities)} commodities, "
            f"{len(universe.crypto)} crypto, {len(universe.fx)} FX pairs"
        )
    except Exception as e:
        print(f"{Fore.RED}  Failed to fetch universe: {e}{Style.RESET_ALL}")
        print(f"  Continuing with empty universe for demonstration...")
        universe = UniverseData()

    # ---- Step 2: Fetch macro data and news ----
    print(f"\n{Fore.YELLOW}[2/6] Fetching macro data and news sentiment...{Style.RESET_ALL}")
    try:
        macro_data = get_macro_data()
        print(f"  Fetched {len(macro_data)} macro indicators")
    except Exception as e:
        print(f"  Macro data fetch skipped: {e}")
        macro_data = {}

    try:
        news = get_news_sentiment(limit=50)
        print(f"  Fetched {len(news)} news articles")
    except Exception as e:
        print(f"  News fetch skipped: {e}")
        import pandas as pd
        news = pd.DataFrame()

    # ---- Step 3: Initialize engines ----
    print(f"\n{Fore.YELLOW}[3/6] Initializing strategy and execution engines...{Style.RESET_ALL}")
    strategy_engine = MultiHorizonEngine(portfolio_value=initial_capital)
    execution_engine = HFTExecutionEngine(initial_capital=initial_capital)
    report_generator = DailyReportGenerator(execution_engine, strategy_engine)

    # ---- Step 4: Run multi-horizon scanning ----
    print(f"\n{Fore.YELLOW}[4/6] Running multi-horizon opportunity scan...{Style.RESET_ALL}")
    all_opportunities = strategy_engine.scan_all_horizons(universe, macro_data, news)

    for horizon, theses in all_opportunities.items():
        print(f"  {horizon.value}: {len(theses)} opportunities found")

    # ---- Step 5: Execute HFT trades ----
    if run_hft:
        print(f"\n{Fore.YELLOW}[5/6] Executing HFT trades...{Style.RESET_ALL}")

        # Execute top HFT opportunities
        hft_trades = all_opportunities.get(TradeHorizon.HFT_INTRADAY, [])
        executed_count = 0
        for thesis in hft_trades[:20]:  # limit to top 20
            result = execution_engine.execute_trade(thesis)
            if result.success:
                executed_count += 1
                print(
                    f"  {Fore.GREEN}EXECUTED: {thesis.direction} {thesis.symbol} "
                    f"@ {result.order.fill_price:.2f} | "
                    f"Score: {thesis.composite_score:.2f}{Style.RESET_ALL}"
                )
            else:
                print(
                    f"  {Fore.RED}REJECTED: {thesis.symbol} - {result.message}{Style.RESET_ALL}"
                )

        # Execute top swing trades
        swing_trades = all_opportunities.get(TradeHorizon.SWING, [])
        for thesis in swing_trades[:10]:
            result = execution_engine.execute_trade(thesis)
            if result.success:
                executed_count += 1

        print(f"  Total trades executed: {executed_count}")

        # Run position management cycle
        if universe.equities:
            current_prices = {
                sym: float(df["Close"].iloc[-1])
                for sym, df in universe.all_dataframes.items()
                if "Close" in df.columns and len(df) > 0
            }

            # Check stop losses
            stops = execution_engine.check_stop_losses(current_prices)
            if stops:
                print(f"  {Fore.RED}Stop losses triggered: {len(stops)}{Style.RESET_ALL}")

            # Manage positions
            updates = execution_engine.manage_open_positions(current_prices)
            closes = [u for u in updates if u.action == "close"]
            if closes:
                print(f"  Positions closed by management rules: {len(closes)}")

    else:
        print(f"\n{Fore.YELLOW}[5/6] HFT execution skipped (disabled){Style.RESET_ALL}")

    # ---- Step 6: Generate reports & buy-and-hold recommendations ----
    report_output = {}
    if generate_reports:
        print(f"\n{Fore.YELLOW}[6/6] Generating daily reports...{Style.RESET_ALL}")

        reports = report_generator.generate_full_daily_report(universe, macro_data, news)
        report_output = reports

        # Print P&L summary
        pnl_report = reports["pnl"]
        print(f"\n{Fore.CYAN}{pnl_report.summary_text}{Style.RESET_ALL}")

        # Print risk summary
        risk_report = reports["risk"]
        print(f"\n{Fore.CYAN}{risk_report.summary_text}{Style.RESET_ALL}")

        # Print missed opportunities
        missed = reports["missed_opportunities"]
        print(f"\n{Fore.CYAN}{missed.summary_text}{Style.RESET_ALL}")

        # Buy-and-hold recommendations
        buy_hold = strategy_engine.get_buy_and_hold_recommendations()
        if buy_hold:
            print(f"\n{Fore.GREEN}{'='*60}")
            print(f"  BUY-AND-HOLD RECOMMENDATIONS (Medium/Long Term)")
            print(f"{'='*60}{Style.RESET_ALL}")
            for rec in buy_hold[:15]:
                print(
                    f"  {Fore.GREEN}{rec.symbol:<6s}{Style.RESET_ALL} "
                    f"{rec.horizon.value:<12s} | "
                    f"Entry: ${rec.entry_price:>8.2f} | "
                    f"Target: ${rec.target_price:>8.2f} | "
                    f"Stop: ${rec.stop_loss:>8.2f} | "
                    f"R:R {rec.risk_reward_ratio:.1f} | "
                    f"Score: {rec.composite_score:.2f}"
                )
                print(f"    {rec.catalyst}")
    else:
        print(f"\n{Fore.YELLOW}[6/6] Report generation skipped{Style.RESET_ALL}")

    # ---- Final Performance Summary ----
    perf = execution_engine.get_performance_summary()
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"  PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"  Portfolio Value:   ${perf['portfolio_value']:>14,.2f}")
    print(f"  Total Return:      {perf['total_return_pct']:>8.2f}%")
    print(f"  Sharpe Ratio:      {perf['sharpe_ratio']:>8.2f}")
    print(f"  Win Rate:          {perf['win_rate_pct']:>8.1f}%")
    print(f"  Open Positions:    {perf['num_open_positions']:>8d}")
    print(f"  Total Trades:      {perf['num_trades']:>8d}")
    print(f"{'='*60}{Style.RESET_ALL}")

    return {
        "opportunities": all_opportunities,
        "performance": perf,
        "reports": report_output,
        "buy_and_hold": strategy_engine.get_buy_and_hold_recommendations(),
        "execution_engine": execution_engine,
        "strategy_engine": strategy_engine,
    }


if __name__ == "__main__":
    inputs = parse_cli_inputs(
        description="Run the hedge fund trading system",
        require_tickers=True,
        default_months_back=None,
        include_graph_flag=True,
        include_reasoning_flag=True,
    )

    tickers = inputs.tickers
    selected_analysts = inputs.selected_analysts

    # Check if --unified flag is present (for the new system)
    # Default: run the original agent-based system.
    # With --unified: run the new multi-horizon system.
    run_unified = getattr(inputs, "unified", False) or "--unified" in sys.argv

    if run_unified:
        # Run the new unified multi-horizon system
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        result = run_unified_system(
            start_date=inputs.start_date,
            end_date=inputs.end_date,
            initial_capital=inputs.initial_cash,
            run_hft=True,
            generate_reports=True,
        )
    else:
        # Original agent-based system
        portfolio = {
            "cash": inputs.initial_cash,
            "margin_requirement": inputs.margin_requirement,
            "margin_used": 0.0,
            "positions": {
                ticker: {
                    "long": 0,
                    "short": 0,
                    "long_cost_basis": 0.0,
                    "short_cost_basis": 0.0,
                    "short_margin_used": 0.0,
                }
                for ticker in tickers
            },
            "realized_gains": {
                ticker: {
                    "long": 0.0,
                    "short": 0.0,
                }
                for ticker in tickers
            },
        }

        result = run_hedge_fund(
            tickers=tickers,
            start_date=inputs.start_date,
            end_date=inputs.end_date,
            portfolio=portfolio,
            show_reasoning=inputs.show_reasoning,
            selected_analysts=inputs.selected_analysts,
            model_name=inputs.model_name,
            model_provider=inputs.model_provider,
        )
        print_trading_output(result)

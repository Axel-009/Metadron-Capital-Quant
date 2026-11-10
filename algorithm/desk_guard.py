# region imports
from AlgorithmImports import *
# endregion
"""Feed guard, collared orders, broker-side resting stops, raw prices for option underlyings, splits, minute
subscriptions — mixed into MetadronCorridor (split out of main.py: QuantConnect caps files at 64,000 characters)."""

import feed_guard as fg
import schedule as hb
import options as op
import decision as dm
import protection as pr
from rotation import realised_vol as name_vol
from sectors import SECTOR_ETF_FALLBACK
from datetime import timedelta

MIN_PROTECT_TRADE = 0.005        # NAV fraction below which a trim remainder is not re-issued


class GuardMixin:   # plain holder of methods (not a base class — see main.py)
    # ------------------------------------------------------------------ feed guard (feed_guard.py)
    def _minutes(self, t):
        return t.timestamp() / 60.0 if t is not None and hasattr(t, "timestamp") else None

    def _last_bar(self, sym):
        try:
            d = self.securities[sym].get_last_data()
            return self._minutes(d.end_time) if d is not None else None
        except Exception:
            return None

    def _is_raw(self, sym) -> bool:
        try:
            subs = list(self.securities[sym].subscriptions)
            return bool(subs) and all(c.data_normalization_mode == DataNormalizationMode.RAW for c in subs)
        except Exception:
            return False

    def _ensure_raw(self, sym) -> bool:
        """Safety net (everything is subscribed RAW): QuantConnect trades options only on RAW-priced underlyings.
        Checks the real subscriptions; returns True when a switch was needed now (strikes picked next window)."""
        if self._is_raw(sym):
            return False
        try:
            self.securities[sym].set_data_normalization_mode(DataNormalizationMode.RAW)
        except Exception as exc:
            self.debug(f"raw prices unavailable for {sym.value}: {exc}")
        self.raw_mode.add(sym.value)
        return True

    def on_dividends(self, dividends):
        """RAW prices drop by the dividend on the ex-date; rescale protection levels so it is not read as a loss."""
        for sym, div in dividends.items():
            try:
                ref, dist = float(div.reference_price), float(div.distribution)
            except Exception:
                continue
            if ref <= 0 or dist <= 0 or dist >= ref:
                continue
            self.adj_factor[sym.value] = self.adj_factor.get(sym.value, 1.0) * (ref - dist) / ref
            st = self.protector.states.get(sym.value)
            if st is not None:
                f = (ref - dist) / ref
                st.peak *= f
                st.entry *= f
                if st.trimmed_at_peak is not None:
                    st.trimmed_at_peak *= f

    def on_splits(self, splits):
        """Rescale position-protection levels when a split moves the traded price (raw-priced option underlyings
        in backtests; every name live). Without this a 4:1 split reads as a −75% crash and trips the stop."""
        for sym, split in splits.items():
            try:
                if split.type != SplitType.SPLIT_OCCURRED:
                    continue
                factor = float(split.split_factor)
            except Exception:
                continue
            if factor <= 0:
                continue                                   # all equities trade on RAW prices
            self.adj_factor[sym.value] = self.adj_factor.get(sym.value, 1.0) * factor
            if sym.value in self.protect:
                self.protect[sym.value]["qty"] = int(self.protect[sym.value]["qty"] / factor)
            st = self.protector.states.get(sym.value)
            if st is not None:
                st.peak *= factor
                st.entry *= factor
                if st.trimmed_at_peak is not None:
                    st.trimmed_at_peak *= factor
            self._drop_stop(sym)                           # re-placed at the new level by _sync_stops
            self.debug(f"SPLIT {sym.value} ×{factor:g}: protection levels rescaled")

    # ------------------------------------------------------------------ protection cooldown
    def _protect_note(self, ticker, keep_qty, why):
        """A trim / stop holds for the cooldown: at most keep_qty shares until it expires."""
        self.protect[ticker] = {"qty": max(int(keep_qty), 0), "why": why,
                                "until": self.time.date() + timedelta(days=pr.PROTECT_COOLDOWN_DAYS)}
        sym = self._symbol(ticker)
        if sym is not None and self.securities.contains_key(sym):
            cap = keep_qty * self.securities[sym].price / self._nav()
            self.targets[ticker] = min(self.targets.get(ticker, cap), cap)

    def on_order_event(self, order_event):
        """A filled resting stop starts the same cooldown as a −10% close (no buy-back the next morning)."""
        try:
            if order_event.status != OrderStatus.FILLED:
                return
            m = self.stop_tickets.get(order_event.symbol)
            if m is not None and m["ticket"].order_id == order_event.order_id \
                    and order_event.symbol.security_type == SecurityType.EQUITY:
                self._protect_note(order_event.symbol.value, 0, "resting stop filled")
        except Exception:
            pass

    def _apply_protect(self, targets):
        """Cap research targets at the shares a trim / stop left, until the cooldown expires."""
        nav, today = self._nav(), self.time.date()
        for t, p in list(self.protect.items()):
            if today > p["until"]:
                self.protect.pop(t, None)
                continue
            sym = self._symbol(t)
            px = self.securities[sym].price if sym is not None and self.securities.contains_key(sym) else 0.0
            targets[t] = min(targets.get(t, 0.0), p["qty"] * px / nav)
        return targets

    def _reprice_protect(self):
        """Protective sells are collared limits; one left unfilled by a falling price is cancelled and re-issued
        at the current price on every risk check (and the resting stop returns once nothing is working)."""
        nav = self._nav()
        for t, p in list(self.protect.items()):
            sym = self._symbol(t)
            h = self._holding(sym)
            if h is None or h.quantity <= p["qty"]:
                continue
            price = self.securities[sym].price
            qty = int(h.quantity) - p["qty"]
            if price <= 0 or (p["qty"] > 0 and qty * price / nav < MIN_PROTECT_TRADE):
                continue
            self._limit(sym, -qty, "EQUITY", f"PROTECT {t}: {p['why']}")

    def _ten_year(self, macro) -> float:
        """10-year Treasury yield: macro agent (FRED) → QuantConnect US Treasury curve → 4.2% default."""
        v = macro.get("treasury_10y")
        if v is None and getattr(self, "ust", None) is not None:
            try:
                d = self.securities[self.ust].get_last_data()
                y = getattr(d, "ten_year", None) if d is not None else None
                v = float(y) / 100.0 if y else None
            except Exception:
                v = None
        return v if v is not None and 0.0 < v < 0.25 else 0.042

    def _ensure_minute(self):
        """'mixed' mode: minute data for the names we may trade today (shortlist + held + option underlyings);
        minute-only extras unused for 10 days are released (the stock stays subscribed daily via the universe)."""
        if self.data_mode != "mixed":
            return
        need = {s.value for s in self.selected}
        need |= {kv.key.value for kv in self.portfolio if kv.value.invested and kv.key.security_type == SecurityType.EQUITY}
        need |= {p["ticker"] for p in self.option_plans}
        today = self.time.date()
        for t in need:
            if t not in self.minute_subs:
                try:
                    self.add_equity(t, Resolution.MINUTE,
                                    data_normalization_mode=DataNormalizationMode.RAW).set_leverage(2.0)
                except Exception as exc:
                    self.debug(f"minute subscription failed for {t}: {exc}")
                    continue
            self.minute_subs[t] = today
        for t, last in list(self.minute_subs.items()):
            if t not in need and (today - last).days > 10 and t not in self.rot_symbols:
                sym = self.sym_of.get(t)
                h = self._holding(sym)
                if sym is not None and (h is None or not h.invested):
                    try:
                        self.remove_security(sym)          # universe re-adds it at daily resolution
                    except Exception:
                        pass
                    self.minute_subs.pop(t, None)

    def _feed_check(self) -> bool:
        """Staleness + SPY/MES cross-feed + jump checks; flips self.feed_ok and logs transitions."""
        keys = {"SPY": self.spy}
        mes_sym = self.future.mapped if self.future is not None else self.ref_index
        ref_name = "MES" if self.future is not None else "SPX"
        if mes_sym is not None and self.future is None and self._last_bar(mes_sym) is None:
            self.ref_missing = getattr(self, "ref_missing", 0) + 1
            if self.ref_missing >= 3:                      # no index data at this resolution: drop the cross-check
                self.debug("FEED: SPX index data unavailable — cross-feed check disabled, staleness checks continue")
                self.ref_index, mes_sym = None, None
        if mes_sym is not None:
            keys[ref_name] = mes_sym
        for kv in self.portfolio:                          # held names: only those with live minute data
            if kv.value.invested and kv.value.type == SecurityType.EQUITY and self._has_minute(kv.key):
                keys[kv.key.value] = kv.key
        last = {k: self._last_bar(s) for k, s in keys.items()}
        spy_px = self.securities[self.spy].price
        mes_px = self.securities[mes_sym].price if mes_sym is not None else None
        was = self.feed_ok
        st = self.feed.check(self._minutes(self.time), last, spy_px, mes_px, ref_name)
        self.feed_ok = st.ok
        if was and not st.ok:
            self.debug("FEED HALT: " + "; ".join(st.reasons[:5]) + " — no new entries, resting stops stay active")
        elif not was and st.ok:
            self.debug("FEED OK: halt cleared")
        self.plot("Feed", "ok", 1.0 if st.ok else 0.0)
        return st.ok

    def _has_minute(self, sym) -> bool:
        try:
            return any(c.resolution == Resolution.MINUTE for c in self.securities[sym].subscriptions)
        except Exception:
            return False

    def _drop_stop(self, sym):
        """Cancel a name's resting stop before trading it (re-placed by _sync_stops once no order is working),
        so a stop and a sell can never both execute and flip the position short."""
        m = self.stop_tickets.pop(sym, None)
        if m is not None:
            try:
                m["ticket"].cancel()
            except Exception:
                pass

    def _working(self, sym) -> bool:
        keep = {m["ticket"].order_id for m in self.stop_tickets.values()}
        try:
            return any(o.id not in keep for o in self.transactions.get_open_orders(sym))
        except Exception:
            return False

    def _cancel_working(self, sym):
        """Cancel working (non-stop) orders for a symbol so cancel-and-replace never stacks orders."""
        keep = {m["ticket"].order_id for m in self.stop_tickets.values()}
        try:
            for o in self.transactions.get_open_orders(sym):
                if o.id not in keep:
                    self.transactions.cancel_order(o.id)
        except Exception:
            pass

    def _limit(self, sym, qty, kind, tag):
        """Collared limit order around the last trusted price (never a raw market order)."""
        px = self.securities[sym].price
        if px <= 0 or qty == 0:
            return
        self._cancel_working(sym)
        self._drop_stop(sym)
        self.limit_order(sym, qty, fg.collar(px, qty > 0, kind), tag=tag)

    def _sync_stops(self):
        """Resting broker-side stops: long stocks at peak −10%, MES overlay at the ~8%-NAV adverse move."""
        want = {}
        for kv in self.portfolio:
            sym, h = kv.key, kv.value
            if not h.invested:
                continue
            if h.type == SecurityType.EQUITY and h.quantity > 0:
                st = self.protector.states.get(sym.value)
                peak = st.peak if st is not None else self.securities[sym].price
                want[sym] = (-int(h.quantity), fg.position_stop(peak))
            elif h.type == SecurityType.FUTURE and h.quantity != 0:
                stop = fg.futures_stop(self.securities[sym].price, self.current_leverage)
                if stop is not None:
                    want[sym] = (-int(h.quantity), stop)
        for sym in list(self.stop_tickets):
            m = self.stop_tickets[sym]
            done = str(getattr(m["ticket"], "status", "")).upper().endswith(("FILLED", "CANCELED", "INVALID"))
            w = want.get(sym)
            if done or w is None or w[0] != m["qty"] or abs(w[1] / m["stop"] - 1) > fg.STOP_REPLACE:
                if not done:
                    try:
                        m["ticket"].cancel()
                    except Exception:
                        pass
                self.stop_tickets.pop(sym, None)
        for sym, (qty, stop) in want.items():
            if sym not in self.stop_tickets and qty != 0 and not self._working(sym):
                t = self.stop_market_order(sym, qty, stop, tag=f"RESTING STOP {sym.value} @ {stop}")
                self.stop_tickets[sym] = {"ticket": t, "qty": qty, "stop": stop}


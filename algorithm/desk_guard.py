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


class GuardMixin:
    # ------------------------------------------------------------------ feed guard (feed_guard.py)
    def _minutes(self, t):
        return t.timestamp() / 60.0 if t is not None and hasattr(t, "timestamp") else None

    def _last_bar(self, sym):
        try:
            d = self.securities[sym].get_last_data()
            return self._minutes(d.end_time) if d is not None else None
        except Exception:
            return None

    def _ensure_raw(self, sym) -> bool:
        """QuantConnect trades options only on underlyings with RAW (unadjusted) prices, so strikes match the real
        price. Returns True when the switch happened now (prices become raw from the next bar)."""
        if sym.value in self.raw_mode:
            return False
        try:
            self.securities[sym].set_data_normalization_mode(DataNormalizationMode.RAW)
        except Exception as exc:
            self.debug(f"raw prices unavailable for {sym.value}: {exc}")
            return True
        self.raw_mode.add(sym.value)
        return True

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
            if factor <= 0 or not (self.live_mode or sym.value in self.raw_mode):
                continue
            st = self.protector.states.get(sym.value)
            if st is not None:
                st.peak *= factor
                st.entry *= factor
                if st.trimmed_at_peak is not None:
                    st.trimmed_at_peak *= factor
            self._drop_stop(sym)                           # re-placed at the new level by _sync_stops
            self.debug(f"SPLIT {sym.value} ×{factor:g}: protection levels rescaled")

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
                    self.add_equity(t, Resolution.MINUTE).set_leverage(2.0)
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
        if mes_sym is not None:
            keys[ref_name] = mes_sym
        for kv in self.portfolio:
            if kv.value.invested and kv.value.type == SecurityType.EQUITY:
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


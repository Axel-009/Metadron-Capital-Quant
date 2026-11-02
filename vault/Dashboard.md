# Dashboard

## Last 10 days
```dataview
TABLE regime, zone, rm AS "Rm", leverage AS "corridor L", overlay AS "MES", nav AS "NAV", positions
FROM "Daily" SORT date DESC LIMIT 10
```

## Holdings
```dataview
TABLE weight, target, subsector, credit_tier, option, unrealised_pct AS "P&L %"
FROM "Positions" WHERE type = "position" SORT weight DESC
```

## Alerts
```dataview
TABLE alerts FROM "Risk" WHERE alerts > 0 SORT date DESC LIMIT 10
```

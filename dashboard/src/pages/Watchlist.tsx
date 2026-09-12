/**
 * Watchlist: products flagged `watch_volatile` — forecast volatility high
 * enough that neither the stockout nor overstock call is confident, so they
 * get their own page rather than being folded into either split.
 */

import { RiskActionPage } from '../components/RiskActionPage';

export function Watchlist() {
  return (
    <RiskActionPage
      action="watch_volatile"
      eyebrow="Watchlist"
      title="Where the forecast itself is shaky"
      note="Demand volatility is high enough here that neither the stockout nor overstock call can be made with confidence — worth a human look before acting on the number."
      emptyTitle="No risk scoring trained yet"
      emptyBody="This instance has no forecast or inventory data to plot. Once real data has been run through the pipeline, volatile products will show up here."
    />
  );
}

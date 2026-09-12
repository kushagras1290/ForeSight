/** Stockout Risk Dashboard: products likely to run out over their lead time. */

import { RiskActionPage } from '../components/RiskActionPage';

export function StockoutRisk() {
  return (
    <RiskActionPage
      action="reorder_now"
      eyebrow="Stockout risk"
      title="What's about to run out"
      note="Products where demand over the replenishment lead time is likely to exceed available stock."
      emptyTitle="No stockout scoring trained yet"
      emptyBody="This instance has no forecast or inventory data to plot. Once real data has been run through the pipeline, at-risk products will show up here."
    />
  );
}

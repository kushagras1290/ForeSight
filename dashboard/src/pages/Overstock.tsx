/** Overstock Dashboard: products likely to be left holding excess stock. */

import { RiskActionPage } from '../components/RiskActionPage';

export function Overstock() {
  return (
    <RiskActionPage
      action="markdown_clear"
      eyebrow="Overstock risk"
      title="What's likely to be left over"
      note="Products where stock on hand is likely to exceed demand over the cover window — candidates to markdown or clear."
      emptyTitle="No overstock scoring trained yet"
      emptyBody="This instance has no forecast or inventory data to plot. Once real data has been run through the pipeline, overstocked products will show up here."
    />
  );
}

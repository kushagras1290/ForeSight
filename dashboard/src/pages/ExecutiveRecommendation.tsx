/**
 * Executive Recommendation Dashboard: what this engagement does not do, and
 * where to take it next. Split out of the old combined Executive Summary
 * page so a stakeholder looking for "what do we do now" doesn't have to
 * scroll past eight sections of engagement background to find it.
 */

import { useCoreData } from '../components/Layout';
import { ExecSection } from '../components/ExecSection';
import { NoDataYet } from '../components/NoDataYet';
import { formatPercent, formatSignedPercent, formatWape } from '../lib/format';

export function ExecutiveRecommendation() {
  const { core, ready } = useCoreData();

  if (!ready?.ready || !core) {
    return (
      <section className="section">
        <NoDataYet
          title="Nothing to recommend yet"
          body="This instance has no trained model or scored data, so there is nothing to base a recommendation on. Once real data has been run through the pipeline, this page covers what the model does not do and what to do next."
          missing={ready?.artifacts_missing}
        />
      </section>
    );
  }

  const { summary, accuracy } = core;

  return (
    <>
      <section className="section">
        <p className="eyebrow">Executive presentation</p>
        <h1 className="section__title" style={{ fontSize: 28 }}>
          What this does not do, and where to go next
        </h1>
        <p className="section__note">
          The honest limits of the current model, and the recommendations that follow from them.
        </p>
      </section>

      <ExecSection index={1} eyebrow="Being straight with you" title="What this does not do">
        <ul className="exec-list">
          <li>
            The forecast is wrong by about {formatWape(accuracy.selected_wape)} of demand on
            average — good for weekly product-level forecasting, not precision. Treat it as a
            prioritised starting point, not an instruction.
          </li>
          <li>
            Slow-moving products forecast far worse than fast ones; they're flagged separately
            rather than hidden inside the average.
          </li>
          <li>
            We cannot see demand that occurred while a product was out of stock — recorded demand
            is a floor, and stockout risk is, if anything, understated for the SKUs that run out
            most often.
          </li>
          <li>
            Stock positions are weekly snapshots, not live — use the Inventory page's what-if
            scorer for anything that has moved since.
          </li>
          <li>
            The stated 80% prediction interval assumes weeks are independent; in practice errors
            persist somewhat week to week, so the true range is a little wider than shown.
          </li>
        </ul>
      </ExecSection>

      <ExecSection index={2} eyebrow="Where to go next" title="Our recommendations">
        <ol className="exec-list exec-list--numbered">
          <li>
            <strong>Now:</strong> work the reorder list top-down —{' '}
            {formatPercent(summary.top_10_share_of_revenue_at_risk, 0)} of the sales at risk sits
            in the top 10 exposures.
          </li>
          <li>
            <strong>Now:</strong> agree a markdown plan for the dead stock before it costs another
            month of warehouse space and cash.
          </li>
          <li>
            <strong>This month:</strong> fix the data problems at source — zero-sale-day exports,
            de-duplicated re-runs, validated lead times on entry — instead of repairing them every
            pipeline run.
          </li>
          <li>
            <strong>This quarter:</strong> re-run monthly and track drift; retrain if average
            error climbs past ~35%.
          </li>
          <li>
            <strong>Later:</strong> feed confirmed promotions in earlier — the further ahead
            they're known, the more accurate the festive forecast becomes.
          </li>
        </ol>
        <p className="callout" style={{ maxWidth: 'none', marginTop: 'var(--space-4)' }}>
          The dashboard and the scoring service are live and NorthBay's to use without us — model{' '}
          {summary.model}, {formatSignedPercent(-accuracy.improvement_vs_baseline)} vs. the
          seasonal-naive baseline it has to beat.
        </p>
      </ExecSection>
    </>
  );
}

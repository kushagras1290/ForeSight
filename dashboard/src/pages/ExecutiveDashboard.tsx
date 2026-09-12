/**
 * Executive Dashboard: the engagement told end to end, in the order a
 * stakeholder presentation would use — problem, client, data, cleaning, EDA,
 * model, performance, risk scoring. Numbers that can change with the data
 * (accuracy, rupee exposure) are read live from the API so this page can
 * never drift from what the rest of the dashboard shows; numbers that don't
 * change with the data (the model's architecture) are condensed from
 * `reports/model_architecture.md` and `model_suite.md`.
 *
 * Recommendations, limitations, and next steps live on their own page,
 * Executive Recommendation — split out because they are a different kind of
 * content (a call to action, not a record of what was built) and a
 * stakeholder skimming this page shouldn't have to scroll past eight
 * sections of background to find them.
 */

import { Link } from 'react-router-dom';

import { useCoreData } from '../components/Layout';
import { AccuracyPanel } from '../components/AccuracyPanel';
import { ExecSection } from '../components/ExecSection';
import { NoDataYet } from '../components/NoDataYet';
import { formatDate, formatInr, formatPercent } from '../lib/format';

const FIGURES = [
  { file: '01_demand_over_time.png', caption: 'Demand over time — the seasonal swing' },
  { file: '02_seasonality_by_category.png', caption: 'Seasonality by category' },
  { file: '03_revenue_concentration.png', caption: 'Revenue concentration across the catalogue' },
  { file: '04_dead_stock.png', caption: 'Dead stock — lines that have stopped selling' },
  { file: '05_promo_uplift.png', caption: 'Promotional uplift by category' },
  { file: '06_data_quality.png', caption: 'Data quality issues found and resolved' },
];

export function ExecutiveDashboard() {
  const { core, ready } = useCoreData();

  if (!ready?.ready || !core) {
    return (
      <section className="section">
        <NoDataYet
          title="Nothing to present yet"
          body="This instance has no trained model or scored data, so there is no engagement to summarise. Once real data has been run through the pipeline, this page tells the full story end to end."
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
          Project FORESIGHT — the engagement, end to end
        </h1>
        <p className="section__note">
          Demand &amp; inventory intelligence for NorthBay Living. Plan cut for the week of{' '}
          {formatDate(summary.origin_week)}.
        </p>
      </section>

      <ExecSection
        index={1}
        eyebrow="Business problem"
        title="Guessing how much to stock, twice a month"
      >
        <p className="callout" style={{ maxWidth: 'none' }}>
          NorthBay Living plans inventory on gut feel and spreadsheets. Every month they stock
          out of things people want — a lost sale they can never recover — and sit on things
          nobody is buying, which locks up cash that later gets marked down to clear. The Head of
          Operations asked for one thing: for every product, how much will we likely sell over
          the next few weeks, which products are about to run out, and which are overstocked so
          we can clear them — in something the team can read without a data scientist in the
          room.
        </p>
      </ExecSection>

      <ExecSection index={2} eyebrow="Client background" title="NorthBay Living">
        <div className="stat-grid">
          <div className="stat">
            <div className="stat__label">Business</div>
            <div className="stat__value" style={{ fontSize: 14 }}>
              Direct-to-consumer home &amp; lifestyle
            </div>
            <div className="stat__hint">Furnishings, décor, small appliances</div>
          </div>
          <div className="stat">
            <div className="stat__label">Scale</div>
            <div className="stat__value tabular">{summary.total_skus} active SKUs</div>
            <div className="stat__hint">Online-only, ships from one warehouse</div>
          </div>
          <div className="stat">
            <div className="stat__label">Data maturity</div>
            <div className="stat__value" style={{ fontSize: 14 }}>
              Exports &amp; spreadsheets
            </div>
            <div className="stat__hint">No forecasting in place before this engagement</div>
          </div>
          <div className="stat">
            <div className="stat__label">Stakeholders</div>
            <div className="stat__value" style={{ fontSize: 14 }}>
              Ops, Merchandising, Finance
            </div>
            <div className="stat__hint">Head of Operations is the primary client</div>
          </div>
        </div>
      </ExecSection>

      <ExecSection index={3} eyebrow="Dataset" title="What we worked from">
        <p className="section__note" style={{ maxWidth: 'none' }}>
          Four extracts, matching the brief's data dictionary: a daily sales fact table, a
          product master, a promotion &amp; holiday calendar, and periodic inventory snapshots.
          Both the client-provided extract and a larger self-generated stress-test dataset were
          run through the identical pipeline and compared head-to-head before either was trusted
          — full reasoning and numbers in{' '}
          <span className="mono">reports/data_source_comparison.md</span>. The provided extract
          is what this dashboard currently serves.
        </p>
      </ExecSection>

      <ExecSection index={4} eyebrow="Data cleaning" title="It was not clean on arrival">
        <p className="section__note" style={{ maxWidth: 'none' }}>
          An automated pipeline profiles every table on arrival, resolves each issue by a
          documented rule, and records why — so the memo can never drift from the code. Recurring
          problem classes: products missing from the master (quarantined, not guessed at),
          part-week records at the edges of the extract (dropped — they read as a demand collapse
          otherwise), a handful of unclassifiable product categories (kept and labelled
          <em> Unclassified</em> rather than hidden), and calendar dates with no promotion
          (explicitly zero-filled rather than left null). Full detail in{' '}
          <span className="mono">reports/eda_memo.md</span> §1.
        </p>
      </ExecSection>

      <ExecSection index={5} eyebrow="EDA insights" title="What the demand data says">
        <div className="card-grid card-grid--figures">
          {FIGURES.map((figure) => (
            <figure key={figure.file} className="exec-figure">
              <img src={`/figures/${figure.file}`} alt={figure.caption} loading="lazy" />
              <figcaption>{figure.caption}</figcaption>
            </figure>
          ))}
        </div>
      </ExecSection>

      <ExecSection index={6} eyebrow="Forecast model" title="The Adaptive Demand Ensemble">
        <p className="section__note" style={{ maxWidth: 'none' }}>
          Five purpose-built models, each targeting a specific failure mode a single global
          forecaster can't cover on its own: a LightGBM gradient booster for steady demand, a
          Croston/TSB-style estimator for intermittent lines that sell nothing most weeks, and a
          seasonal profile for brand-new SKUs with no history. A regime router decides, per
          SKU-week, which of the three to trust and blends them — steady demand leans on the
          GBM, intermittent lines lean on TSB, cold-start SKUs lean on the seasonal profile. Two
          further models (hierarchical reconciliation, censored-demand recovery) were built,
          measured, and honestly reported as not earning their place on this dataset rather than
          kept for appearances — detail in <span className="mono">reports/model_suite.md</span>{' '}
          and <span className="mono">reports/model_architecture.md</span>.
        </p>
      </ExecSection>

      <ExecSection index={7} eyebrow="Model performance" title="Can you trust it">
        <AccuracyPanel accuracy={accuracy} />
      </ExecSection>

      <ExecSection index={8} eyebrow="Risk scoring" title="Turning a forecast into a decision">
        <p className="section__note" style={{ maxWidth: 'none' }}>
          Every SKU is scored for stockout risk (forecast demand over lead time against on-hand +
          on-order stock) and overstock risk (weeks of cover against a threshold), then given one
          of four plain-language actions: reorder now, markdown &amp; clear, watch — volatile, or
          healthy. The arithmetic is deliberately transparent rather than a second model, so an
          operations manager can challenge a reorder and get a straight answer.
        </p>
        <div className="stat-grid" style={{ marginTop: 'var(--space-4)' }}>
          <div className="stat">
            <div className="stat__label">Reorder now</div>
            <div className="stat__value tabular" style={{ color: 'var(--signal-critical)' }}>
              {summary.reorder_now_skus} SKUs
            </div>
            <div className="stat__hint">{formatInr(summary.revenue_at_risk_total)} sales at risk</div>
          </div>
          <div className="stat">
            <div className="stat__label">Markdown &amp; clear</div>
            <div className="stat__value tabular" style={{ color: 'var(--signal-warn-ink)' }}>
              {summary.markdown_skus} SKUs
            </div>
            <div className="stat__hint">{formatInr(summary.locked_capital_total)} capital locked</div>
          </div>
          <div className="stat">
            <div className="stat__label">Concentration</div>
            <div className="stat__value tabular">
              {formatPercent(summary.top_10_share_of_revenue_at_risk, 0)}
            </div>
            <div className="stat__hint">of sales-at-risk sits in the top 10 SKUs</div>
          </div>
        </div>
        <p style={{ marginTop: 'var(--space-4)' }}>
          <Link to="/risk/stockout" className="button button--ghost">
            Open the Stockout Risk Dashboard →
          </Link>
        </p>
      </ExecSection>

      <section className="section">
        <p className="callout" style={{ maxWidth: 'none' }}>
          Continue to{' '}
          <Link to="/executive-recommendation" className="button button--ghost">
            Executive Recommendation →
          </Link>{' '}
          for what this does not do and where to go next.
        </p>
      </section>
    </>
  );
}

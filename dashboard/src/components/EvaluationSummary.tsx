/**
 * The headline read of an uploaded-actuals evaluation: what got scored, and
 * the five numbers that answer "does the forecast still hold". Shared between
 * the inline upload result (Demand Forecast page) and the full Test Results
 * page, so the two views can never disagree about what the upload produced.
 */

import type { EvaluationResponse } from '../lib/api';
import { formatPercent, formatSignedPercent, formatUnits, formatWape } from '../lib/format';

interface EvaluationSummaryProps {
  result: EvaluationResponse;
  intervalTarget: number;
}

export function EvaluationSummary({ result, intervalTarget }: EvaluationSummaryProps) {
  const improvement = result.improvement_vs_baseline;
  const coverageGap = result.interval_coverage - intervalTarget;

  return (
    <>
      <div className="callout" style={{ marginBottom: 'var(--space-5)' }}>
        <span className="callout__mark" aria-hidden="true">
          ◆
        </span>
        <span>
          Scored <strong>{formatUnits(result.rows_matched)}</strong> of{' '}
          {formatUnits(result.rows_submitted)} submitted rows, covering{' '}
          <strong>{result.weeks_covered.length}</strong> weeks from {result.weeks_covered[0]} to{' '}
          {result.weeks_covered.at(-1)}.
          {result.rows_unmatched > 0 ? (
            <>
              {' '}
              <strong>{formatUnits(result.rows_unmatched)}</strong> rows had no matching forecast
              {result.unmatched_examples.length > 0 ? (
                <>
                  {' '}
                  (e.g. <span className="mono">{result.unmatched_examples[0]}</span>)
                </>
              ) : null}{' '}
              — usually a product code we do not hold, or a week outside the forecast range.
            </>
          ) : null}
        </span>
      </div>

      <div className="stat-grid">
        <div className="stat">
          <div className="stat__label">Forecast error on your data</div>
          <div className="stat__value tabular">{formatWape(result.accuracy.wape)}</div>
          <div className="stat__hint">
            {formatUnits(result.accuracy.total_actual)} units of actual demand
          </div>
        </div>

        <div className="stat">
          <div className="stat__label">Seasonal-naive on the same rows</div>
          <div className="stat__value tabular" style={{ color: 'var(--ink-3)' }}>
            {result.baseline ? formatWape(result.baseline.wape) : '—'}
          </div>
          <div className="stat__hint">
            {result.baseline
              ? `computable on ${formatPercent(result.baseline_row_coverage, 0)} of rows`
              : 'not enough history to compute'}
          </div>
        </div>

        <div className="stat">
          <div className="stat__label">Better than the baseline by</div>
          <div
            className="stat__value tabular"
            style={{
              color:
                improvement === null
                  ? 'var(--ink-3)'
                  : improvement > 0
                    ? 'var(--signal-ok)'
                    : 'var(--signal-critical)',
            }}
          >
            {improvement === null ? '—' : formatPercent(improvement, 0)}
          </div>
          <div className="stat__hint">
            {improvement === null
              ? 'no baseline available'
              : improvement > 0
                ? 'the model is still ahead'
                : 'the baseline won — investigate'}
          </div>
        </div>

        <div className="stat">
          <div className="stat__label">Bias</div>
          <div className="stat__value tabular">
            {formatSignedPercent(result.accuracy.bias_relative)}
          </div>
          <div className="stat__hint">
            {Math.abs(result.accuracy.bias_relative) < 0.05
              ? 'balanced'
              : result.accuracy.bias_relative < 0
                ? 'forecasting light — risks under-ordering'
                : 'forecasting heavy — risks over-ordering'}
          </div>
        </div>

        <div className="stat">
          <div className="stat__label">Interval coverage</div>
          <div
            className="stat__value tabular"
            style={{
              color: Math.abs(coverageGap) <= 0.07 ? 'var(--signal-ok)' : 'var(--signal-warn-ink)',
            }}
          >
            {formatPercent(result.interval_coverage)}
          </div>
          <div className="stat__hint">
            target {formatPercent(intervalTarget, 0)} —{' '}
            {Math.abs(coverageGap) <= 0.07
              ? 'still calibrated'
              : coverageGap < 0
                ? 'intervals too narrow'
                : 'intervals too wide'}
          </div>
        </div>
      </div>
    </>
  );
}

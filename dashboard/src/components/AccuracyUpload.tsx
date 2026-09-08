/**
 * Check the forecast against your own actuals.
 *
 * The backtest says how the model performed on history. It cannot say how it is
 * performing *now*. Once the weeks a plan covered have happened, the client has
 * the one thing no backtest can provide — ground truth on the forecast they
 * actually acted on. This is where they bring it.
 *
 * Deliberately forgiving about the file: column headers are matched leniently
 * server-side, and any date inside a week snaps to that week. An ops team should
 * be able to export from their system and drop it in, not reshape a spreadsheet
 * to match our column names.
 */

import { useCallback, useRef, useState } from 'react';

import { ApiError, api, type EvaluationResponse } from '../lib/api';
import { formatPercent, formatSignedPercent, formatUnits, formatWape } from '../lib/format';

const TEMPLATE = `sku_id,week_starting,units
NBL-LGT-004,2026-08-31,42
NBL-LGT-004,2026-09-07,38
NBL-KTD-012,2026-08-31,17
`;

function downloadTemplate(): void {
  const blob = new Blob([TEMPLATE], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = 'foresight-actuals-template.csv';
  anchor.click();
  // Revoking immediately can cancel the download in some browsers; a tick is enough.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

interface AccuracyUploadProps {
  intervalTarget: number;
}

export function AccuracyUpload({ intervalTarget }: AccuracyUploadProps) {
  const [result, setResult] = useState<EvaluationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [filenames, setFilenames] = useState<string[]>([]);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const submit = useCallback(async (files: File[]) => {
    if (files.length === 0) return;
    setBusy(true);
    setError(null);
    setResult(null);
    setFilenames(files.map((file) => file.name));
    try {
      setResult(await api.evaluateUpload(files));
    } catch (cause) {
      setError(
        cause instanceof ApiError
          ? cause.message
          : 'The files could not be processed.',
      );
    } finally {
      setBusy(false);
    }
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      setDragging(false);
      void submit(Array.from(event.dataTransfer.files ?? []));
    },
    [submit],
  );

  const busyLabel =
    filenames.length === 1 ? `Scoring ${filenames[0]}…` : `Scoring ${filenames.length} files…`;

  const improvement = result?.improvement_vs_baseline ?? null;
  const coverageGap = result ? result.interval_coverage - intervalTarget : 0;

  return (
    <section className="section">
      <div className="section__head">
        <p className="eyebrow">Calibrate on your own data</p>
        <h3 className="section__title">Check the forecast against what actually happened</h3>
        <p className="section__note">
          Upload a file of actual weekly sales. We score the forecast we gave you against it,
          using the same measures and the same baseline as the backtest above — so you can see
          whether the accuracy still holds on your data rather than taking ours on trust.
        </p>
      </div>

      <div
        className="dropzone"
        data-dragging={dragging}
        data-busy={busy}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            inputRef.current?.click();
          }
        }}
        role="button"
        tabIndex={0}
        aria-label="Upload CSV files of actual sales, and optionally a forecast"
        aria-busy={busy}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".csv,text/csv"
          multiple
          hidden
          onChange={(event) => {
            void submit(Array.from(event.target.files ?? []));
            // Reset so re-selecting the same filenames fires change again.
            event.target.value = '';
          }}
        />

        <p className="dropzone__mark" aria-hidden="true">
          {busy ? '◴' : '↑'}
        </p>
        <p className="dropzone__title">
          {busy ? busyLabel : 'Drop your CSVs here, or click to choose them'}
        </p>
        <p className="dropzone__hint">
          Drop <strong>both files together</strong> — the actuals and, if you have one, the
          forecast to score them against. We work out which is which from the headers, so the
          order does not matter. Actuals need a product code, a week date and a quantity;
          names are matched loosely — <span className="mono">sku</span> /{' '}
          <span className="mono">product</span>, <span className="mono">week</span> /{' '}
          <span className="mono">date</span>, <span className="mono">units</span> /{' '}
          <span className="mono">qty</span> all work. A forecast is recognised by a{' '}
          <span className="mono">prediction</span> or <span className="mono">forecast</span>{' '}
          column. Send actuals alone and they are scored against our own forecast.
        </p>
        <button
          type="button"
          className="button button--ghost"
          onClick={(event) => {
            event.stopPropagation();
            downloadTemplate();
          }}
        >
          Download a template
        </button>
      </div>

      {error ? (
        <div className="callout callout--warn" style={{ marginTop: 'var(--space-4)' }}>
          <span className="callout__mark" aria-hidden="true">
            !
          </span>
          <span>{error}</span>
        </div>
      ) : null}

      {result ? (
        <div style={{ marginTop: 'var(--space-5)' }}>
          {/* --- Match quality: say plainly what was and was not scored ----- */}
          <div className="callout" style={{ marginBottom: 'var(--space-5)' }}>
            <span className="callout__mark" aria-hidden="true">
              ◆
            </span>
            <span>
              Scored <strong>{formatUnits(result.rows_matched)}</strong> of{' '}
              {formatUnits(result.rows_submitted)} submitted rows, covering{' '}
              <strong>{result.weeks_covered.length}</strong> weeks from{' '}
              {result.weeks_covered[0]} to {result.weeks_covered.at(-1)}.
              {result.rows_unmatched > 0 ? (
                <>
                  {' '}
                  <strong>{formatUnits(result.rows_unmatched)}</strong> rows had no matching
                  forecast
                  {result.unmatched_examples.length > 0 ? (
                    <>
                      {' '}
                      (e.g. <span className="mono">{result.unmatched_examples[0]}</span>)
                    </>
                  ) : null}
                  {' '}— usually a product code we do not hold, or a week outside the forecast
                  range.
                </>
              ) : null}
            </span>
          </div>

          {/* --- Headline --------------------------------------------------- */}
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
                  color:
                    Math.abs(coverageGap) <= 0.07
                      ? 'var(--signal-ok)'
                      : 'var(--signal-warn-ink)',
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

          {/* --- Breakdowns -------------------------------------------------- */}
          <div className="two-col" style={{ marginTop: 'var(--space-5)' }}>
            <div>
              <p className="eyebrow" style={{ marginBottom: 'var(--space-2)' }}>
                By weeks ahead
              </p>
              <table className="model-table">
                <thead>
                  <tr>
                    <th scope="col">Ahead</th>
                    <th scope="col" className="numeric">
                      Rows
                    </th>
                    <th scope="col" className="numeric">
                      Model
                    </th>
                    <th scope="col" className="numeric">
                      Baseline
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {result.by_horizon.map((row) => (
                    <tr key={row.key}>
                      <td>+{row.key}</td>
                      <td className="numeric">{formatUnits(row.n_observations)}</td>
                      <td className="numeric" style={{ fontWeight: 600 }}>
                        {formatWape(row.wape)}
                      </td>
                      <td className="numeric" style={{ color: 'var(--ink-3)' }}>
                        {row.baseline_wape === null ? '—' : formatWape(row.baseline_wape)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div>
              <p className="eyebrow" style={{ marginBottom: 'var(--space-2)' }}>
                Biggest sources of error
              </p>
              <p
                className="section__note"
                style={{ marginBottom: 'var(--space-2)', fontSize: 11.5 }}
              >
                Where the misses concentrate. These are the products to look at first.
              </p>
              <table className="model-table">
                <thead>
                  <tr>
                    <th scope="col">Product</th>
                    <th scope="col" className="numeric">
                      Actual
                    </th>
                    <th scope="col" className="numeric">
                      Forecast
                    </th>
                    <th scope="col" className="numeric">
                      Share of error
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {result.worst_contributors.slice(0, 8).map((row) => (
                    <tr key={row.sku_id}>
                      <td className="mono" style={{ fontSize: 12 }}>
                        {row.sku_id}
                      </td>
                      <td className="numeric">{formatUnits(row.total_actual)}</td>
                      <td className="numeric">{formatUnits(row.total_forecast)}</td>
                      <td className="numeric" style={{ fontWeight: 600 }}>
                        {formatPercent(row.share_of_total_error, 1)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}

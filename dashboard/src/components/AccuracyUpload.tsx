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
import { useNavigate } from 'react-router-dom';

import { ApiError, api, type EvaluationResponse } from '../lib/api';
import { EvaluationSummary } from './EvaluationSummary';

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
  const navigate = useNavigate();
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
          <EvaluationSummary result={result} intervalTarget={intervalTarget} />

          <button
            type="button"
            className="button"
            style={{ marginTop: 'var(--space-5)' }}
            onClick={() => navigate('/test-results', { state: { result, intervalTarget } })}
          >
            Open full test results ({result.worst_contributors.length} products) →
          </button>
        </div>
      ) : null}
    </section>
  );
}

/**
 * Model Benchmark: every forecasting method this project knows about, on
 * identical folds — 9 standard/off-the-shelf models plus the 7 custom
 * models, scored (or, for the measured-effect custom models, described)
 * side by side. Reads `/api/models/benchmark`, backed by
 * `artifacts/all_models_comparison.json` (written by
 * `scripts/10_run_all_models.py`).
 */

import { useEffect, useState } from 'react';

import { ApiError, api, type ModelBenchmarkRow } from '../lib/api';
import { useCoreData } from '../components/Layout';
import { Empty, ErrorState, Loading } from '../components/States';
import { NoDataYet } from '../components/NoDataYet';
import { formatSignedPercent } from '../lib/format';

const BASELINE_MODELS = new Set(['seasonal_naive', 'naive_last_value']);

function rowClass(row: ModelBenchmarkRow): string {
  if (BASELINE_MODELS.has(row.model)) return 'model-row--baseline';
  if (row.model === 'adaptive_ensemble' || row.model === 'lightgbm') return 'model-row--headline';
  return '';
}

export function ModelBenchmark() {
  const { ready } = useCoreData();
  const [rows, setRows] = useState<ModelBenchmarkRow[]>([]);
  const [folds, setFolds] = useState(0);
  const [available, setAvailable] = useState(true);
  const [reason, setReason] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [reloadToken, setReloadToken] = useState(0);

  useEffect(() => {
    if (!ready?.ready) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .modelBenchmark()
      .then((result) => {
        if (cancelled) return;
        setAvailable(result.available);
        setReason(result.reason);
        setRows(result.rows);
        setFolds(result.folds);
      })
      .catch((cause: unknown) => {
        if (!cancelled) {
          setError(cause instanceof ApiError ? cause.message : 'Could not load the model benchmark.');
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [reloadToken, ready?.ready]);

  if (!ready?.ready) {
    return (
      <section className="section">
        <NoDataYet
          title="No model benchmark trained yet"
          body="This instance has no backtest to compare models on. Once real data has been run through the pipeline, every model's accuracy will show up here."
          missing={ready?.artifacts_missing}
        />
      </section>
    );
  }

  return (
    <section className="section">
      <div className="section__head">
        <p className="eyebrow">Model benchmark</p>
        <h2 className="section__title">Every model this project has, side by side</h2>
        <p className="section__note">
          {folds ? `Scored on ${folds} identical rolling-origin folds. ` : ''}
          Standard models are scored on WAPE; the measured-effect custom models describe their
          own effect instead, by design (see <span className="mono">reports/model_suite.md</span>{' '}
          #7).
        </p>
      </div>

      {error ? (
        <ErrorState message={error} onRetry={() => setReloadToken((token) => token + 1)} />
      ) : loading ? (
        <Loading rows={10} label="Loading the model benchmark" />
      ) : !available ? (
        <Empty
          title="Not run yet"
          body={reason ?? 'Run scripts/10_run_all_models.py to populate this page.'}
        />
      ) : rows.length === 0 ? (
        <Empty title="No rows" body="The benchmark artifact is empty." />
      ) : (
        <div className="table-scroll">
          <table className="table table--static">
            <caption className="visually-hidden">
              Every forecasting method, WAPE and bias where comparable, otherwise a measured
              effect.
            </caption>
            <thead>
              <tr>
                <th scope="col">Model</th>
                <th scope="col" className="numeric">
                  WAPE
                </th>
                <th scope="col" className="numeric">
                  Bias
                </th>
                <th scope="col">Note</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.model} className={rowClass(row)}>
                  <td>
                    <span className="sku-cell__id">{row.model}</span>
                  </td>
                  <td className="numeric">
                    {row.wape !== null ? `${(row.wape * 100).toFixed(1)}%` : '—'}
                  </td>
                  <td className="numeric">
                    {row.bias_relative !== null ? formatSignedPercent(row.bias_relative) : '—'}
                  </td>
                  <td style={{ maxWidth: 480 }}>{row.note ?? row.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

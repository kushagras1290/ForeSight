/**
 * Per-page empty state for "nothing has been trained on this instance yet" -
 * a stable, expected state for a freshly stood-up instance, not an error.
 * Distinct from `ErrorState` (something broke) and the loading skeletons
 * (data is on its way): here, nothing is wrong and nothing is coming until
 * someone runs the pipeline on real data.
 */

interface NoDataYetProps {
  title: string;
  body: string;
  missing?: string[];
}

export function NoDataYet({ title, body, missing }: NoDataYetProps) {
  return (
    <div className="state">
      <div className="state__mark" aria-hidden="true">
        ◌
      </div>
      <p className="state__title">{title}</p>
      <p className="state__body">{body}</p>
      <pre className="state__hint">
        {`python scripts/01_run_pipeline.py
python scripts/03_train_backtest.py
python scripts/04_score_risk.py`}
      </pre>
      {missing?.length ? (
        <p className="state__body" style={{ fontSize: 12 }}>
          Missing: <span className="mono">{missing.join(', ')}</span>
        </p>
      ) : null}
    </div>
  );
}

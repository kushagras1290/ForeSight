/**
 * Demand Forecast page: how the forecast performs on the backtest and the
 * independent holdout, plus a direct lookup of the forward forecast for any
 * product — the two questions "can I trust it" and "what does it say".
 */

import { useState } from 'react';
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import { ApiError, api, type SkuForecastResponse } from '../lib/api';
import { useCoreData } from '../components/Layout';
import { AccuracyPanel } from '../components/AccuracyPanel';
import { HoldoutPanel } from '../components/HoldoutPanel';
import { ErrorState, LoadingChart } from '../components/States';
import { NoDataYet } from '../components/NoDataYet';
import { formatDate, formatShortDate, formatUnits } from '../lib/format';

interface ForecastTooltipPayloadEntry {
  dataKey?: string | number;
  value?: number | number[];
}

function ForecastTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: ForecastTooltipPayloadEntry[];
  label?: string;
}) {
  if (!active || !payload?.length || !label) return null;
  const forecast = payload.find((entry) => entry.dataKey === 'forecast_units')?.value;
  const band = payload.find((entry) => entry.dataKey === 'band')?.value;

  return (
    <div className="tooltip">
      <div className="tooltip__title">Week of {formatDate(label)}</div>
      {typeof forecast === 'number' ? (
        <div className="tooltip__row">
          <span>Forecast</span>
          <strong>{formatUnits(forecast)} units</strong>
        </div>
      ) : null}
      {Array.isArray(band) ? (
        <div className="tooltip__row">
          <span>80% range</span>
          <strong>
            {formatUnits(band[0] ?? 0)}–{formatUnits(band[1] ?? 0)}
          </strong>
        </div>
      ) : null}
    </div>
  );
}

export function DemandForecast() {
  const { core, ready } = useCoreData();
  const [skuInput, setSkuInput] = useState('');
  const [lookup, setLookup] = useState<SkuForecastResponse | null>(null);
  const [lookupLoading, setLookupLoading] = useState(false);
  const [lookupError, setLookupError] = useState<string | null>(null);

  if (!ready?.ready || !core) {
    return (
      <section className="section">
        <NoDataYet
          title="No forecast model trained yet"
          body="This instance has no model to look up forecasts against, backtest, or calibrate. Once real data has been run through the pipeline, this page comes alive."
          missing={ready?.artifacts_missing}
        />
      </section>
    );
  }

  const runLookup = (skuId: string) => {
    const trimmed = skuId.trim();
    if (!trimmed) return;
    setLookupLoading(true);
    setLookupError(null);
    api
      .forecast(trimmed)
      .then(setLookup)
      .catch((cause: unknown) => {
        setLookup(null);
        setLookupError(
          cause instanceof ApiError ? cause.message : 'Could not load a forecast for that SKU.',
        );
      })
      .finally(() => setLookupLoading(false));
  };

  return (
    <>
      <section className="section">
        <div className="section__head">
          <p className="eyebrow">Look up a product</p>
          <h2 className="section__title">The forward forecast, for any SKU</h2>
          <p className="section__note">
            Enter a product code to see the {core.summary.horizon_weeks}-week forecast, straight
            from <span className="mono">/api/forecast/&#123;sku_id&#125;</span>.
          </p>
        </div>

        <form
          className="toolbar"
          onSubmit={(event) => {
            event.preventDefault();
            runLookup(skuInput);
          }}
        >
          <input
            className="input"
            type="search"
            aria-label="Product code"
            placeholder="e.g. NBL-LGT-004"
            value={skuInput}
            onChange={(event) => setSkuInput(event.target.value)}
          />
          <button type="submit" className="button">
            Forecast
          </button>
        </form>

        {lookupError ? (
          <ErrorState message={lookupError} onRetry={() => runLookup(skuInput)} />
        ) : lookupLoading ? (
          <LoadingChart height={260} />
        ) : lookup ? (
          <div className="chart-card">
            <p className="eyebrow">
              {lookup.sku_id} — {lookup.category} · {lookup.subcategory} — model {lookup.model}
            </p>
            <ResponsiveContainer width="100%" height={260}>
              <ComposedChart
                data={lookup.forecast.map((point) => ({
                  ...point,
                  band: [point.lower_units, point.upper_units],
                }))}
                margin={{ top: 16, right: 8, bottom: 4, left: -14 }}
              >
                <CartesianGrid stroke="var(--rule)" strokeWidth={0.6} vertical={false} />
                <XAxis
                  dataKey="week_starting"
                  tickFormatter={formatShortDate}
                  tick={{ fontSize: 10, fill: 'var(--ink-3)' }}
                  stroke="var(--rule)"
                />
                <YAxis
                  tick={{ fontSize: 10, fill: 'var(--ink-3)' }}
                  stroke="var(--rule)"
                  width={52}
                />
                <Tooltip content={<ForecastTooltip />} cursor={{ stroke: 'var(--rule-strong)' }} />
                <Area
                  dataKey="band"
                  stroke="none"
                  fill="var(--series-forecast)"
                  fillOpacity={0.14}
                  isAnimationActive={false}
                />
                <Line
                  dataKey="forecast_units"
                  stroke="var(--series-forecast)"
                  strokeWidth={2}
                  dot={{ r: 2 }}
                  isAnimationActive={false}
                />
              </ComposedChart>
            </ResponsiveContainer>
            {lookup.risk ? (
              <p className="section__note" style={{ marginTop: 'var(--space-2)' }}>
                {lookup.risk.action_label}: {lookup.risk.action_rationale}
              </p>
            ) : null}
          </div>
        ) : null}
      </section>

      <section className="section">
        <div className="section__head">
          <p className="eyebrow">Trust</p>
          <h2 className="section__title">Backtested accuracy</h2>
        </div>
        <AccuracyPanel accuracy={core.accuracy} />
      </section>

      <section className="section">
        <div className="section__head">
          <p className="eyebrow">Independent check</p>
          <h2 className="section__title">Chronological holdout</h2>
        </div>
        <HoldoutPanel holdout={core.holdout} />
      </section>
    </>
  );
}

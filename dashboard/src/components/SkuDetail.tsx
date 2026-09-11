/**
 * Per-SKU detail drawer.
 *
 * Answers the two questions a planner asks once the worklist has flagged
 * something: *why* is this being recommended, and *should I believe the
 * forecast?* The chart shows the model's own backtested track record against
 * what actually happened, next to the forward forecast — so the recommendation
 * is shown alongside the evidence for it rather than asserted.
 */

import { useEffect, useMemo } from 'react';
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import { ACTION_STYLE, type SkuDetail as SkuDetailData } from '../lib/api';
import {
  formatDate,
  formatInr,
  formatInrExact,
  formatShortDate,
  formatUnits,
  formatWeeks,
} from '../lib/format';
import { ErrorState, LoadingChart } from './States';

/** Weeks of history to plot. Two years shows the seasonal cycle twice over. */
const HISTORY_WEEKS = 104;

interface ChartRow {
  week: string;
  actual?: number;
  baseline?: number;
  forecast?: number;
  band?: [number, number];
}

function buildChartRows(detail: SkuDetailData): { rows: ChartRow[]; originWeek: string | null } {
  const byWeek = new Map<string, ChartRow>();

  const upsert = (week: string): ChartRow => {
    const existing = byWeek.get(week);
    if (existing) return existing;
    const created: ChartRow = { week };
    byWeek.set(week, created);
    return created;
  };

  for (const point of detail.history.slice(-HISTORY_WEEKS)) {
    upsert(point.week_starting).actual = point.units;
  }

  // Backtested weeks: what the model said at the time, against what happened.
  for (const point of detail.backtest) {
    const row = upsert(point.week_starting);
    row.forecast = point.forecast;
    row.baseline = point.baseline;
    row.band = [point.lower, point.upper];
  }

  // Forward forecast: no actual exists yet, by definition.
  for (const point of detail.forecast) {
    const row = upsert(point.week_starting);
    row.forecast = point.forecast_units;
    row.band = [point.lower_units, point.upper_units];
  }

  const rows = Array.from(byWeek.values()).sort((left, right) =>
    left.week.localeCompare(right.week),
  );

  const firstForward = detail.forecast[0]?.week_starting ?? null;
  return { rows, originWeek: firstForward };
}

interface TooltipPayloadEntry {
  dataKey?: string | number;
  value?: number | number[];
}

function ChartTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: TooltipPayloadEntry[];
  label?: string;
}) {
  if (!active || !payload?.length || !label) return null;

  const find = (key: string) => payload.find((entry) => entry.dataKey === key)?.value;
  const actual = find('actual');
  const forecast = find('forecast');
  const baseline = find('baseline');
  const band = find('band');

  return (
    <div className="tooltip">
      <div className="tooltip__title">Week of {formatDate(label)}</div>
      {typeof actual === 'number' ? (
        <div className="tooltip__row">
          <span>Actual</span>
          <strong>{formatUnits(actual)} units</strong>
        </div>
      ) : null}
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
      {typeof baseline === 'number' ? (
        <div className="tooltip__row">
          <span>Seasonal-naive</span>
          <strong>{formatUnits(baseline)} units</strong>
        </div>
      ) : null}
    </div>
  );
}

interface SkuDetailContentProps {
  detail: SkuDetailData | null;
  loading: boolean;
  error: string | null;
  onRetry: () => void;
}

/**
 * The actual detail view: rationale, forecast chart, stock position, money.
 *
 * Split out from `SkuDetailDrawer` so the same content can be shown two ways —
 * as a slide-in quick look (the drawer) or as the body of the standalone
 * Product Details page — without the chart-building logic or markup existing
 * in two places that could drift apart.
 */
export function SkuDetailContent({ detail, loading, error, onRetry }: SkuDetailContentProps) {
  const chart = useMemo(() => (detail ? buildChartRows(detail) : null), [detail]);
  const risk = detail?.risk ?? null;
  const style = risk ? ACTION_STYLE[risk.action] : null;

  return (
    <>
      {error ? (
            <ErrorState message={error} onRetry={onRetry} />
          ) : loading || !detail || !chart ? (
            <>
              <LoadingChart height={280} />
              <LoadingChart height={120} />
            </>
          ) : (
            <>
              {risk && style ? (
                <div
                  className="rationale"
                  style={
                    {
                      '--action-colour': style.colour,
                      '--action-soft': style.soft,
                    } as React.CSSProperties
                  }
                >
                  <strong style={{ color: style.ink }}>
                    {style.icon} {risk.action_label}
                  </strong>{' '}
                  — {risk.action_rationale}
                </div>
              ) : null}

              {/* --- Forecast vs actual ------------------------------------ */}
              <div className="chart-card">
                <p className="eyebrow">Demand — actual, backtested forecast and forward view</p>
                <ResponsiveContainer width="100%" height={276}>
                  <ComposedChart
                    data={chart.rows}
                    margin={{ top: 16, right: 8, bottom: 4, left: -14 }}
                  >
                    <CartesianGrid stroke="var(--rule)" strokeWidth={0.6} vertical={false} />
                    <XAxis
                      dataKey="week"
                      tickFormatter={formatShortDate}
                      tick={{ fontSize: 10, fill: 'var(--ink-3)' }}
                      stroke="var(--rule)"
                      minTickGap={44}
                    />
                    <YAxis
                      tick={{ fontSize: 10, fill: 'var(--ink-3)' }}
                      stroke="var(--rule)"
                      width={52}
                      label={{
                        value: 'Units / week',
                        angle: -90,
                        position: 'insideLeft',
                        offset: 18,
                        style: { fontSize: 10, fill: 'var(--ink-3)' },
                      }}
                    />
                    <Tooltip content={<ChartTooltip />} cursor={{ stroke: 'var(--rule-strong)' }} />

                    {/* Prediction interval, drawn first so lines sit on top. */}
                    <Area
                      dataKey="band"
                      stroke="none"
                      fill="var(--series-forecast)"
                      fillOpacity={0.14}
                      isAnimationActive={false}
                      connectNulls={false}
                    />
                    <Line
                      dataKey="actual"
                      stroke="var(--series-actual)"
                      strokeWidth={1.8}
                      dot={false}
                      isAnimationActive={false}
                      connectNulls={false}
                    />
                    <Line
                      dataKey="baseline"
                      stroke="var(--series-baseline)"
                      strokeWidth={1.4}
                      strokeDasharray="4 3"
                      dot={false}
                      isAnimationActive={false}
                      connectNulls={false}
                    />
                    <Line
                      dataKey="forecast"
                      stroke="var(--series-forecast)"
                      strokeWidth={2}
                      dot={false}
                      isAnimationActive={false}
                      connectNulls={false}
                    />

                    {/* Everything right of this line has not happened yet. */}
                    {chart.originWeek ? (
                      <ReferenceLine
                        x={chart.originWeek}
                        stroke="var(--harbour-500)"
                        strokeWidth={1.2}
                        strokeDasharray="3 3"
                        label={{
                          value: 'forecast',
                          position: 'insideTopRight',
                          style: {
                            fontSize: 9.5,
                            fill: 'var(--harbour-500)',
                            fontWeight: 600,
                            letterSpacing: '0.08em',
                            textTransform: 'uppercase',
                          },
                        }}
                      />
                    ) : null}
                  </ComposedChart>
                </ResponsiveContainer>

                <div className="chart-legend">
                  <span className="chart-legend__item">
                    <span
                      className="chart-legend__line"
                      style={{ borderColor: 'var(--series-actual)' }}
                    />
                    Actual
                  </span>
                  <span className="chart-legend__item">
                    <span
                      className="chart-legend__line"
                      style={{ borderColor: 'var(--series-forecast)' }}
                    />
                    Model forecast
                  </span>
                  <span className="chart-legend__item">
                    <span
                      className="chart-legend__line"
                      style={{
                        borderColor: 'var(--series-baseline)',
                        borderTopStyle: 'dashed',
                      }}
                    />
                    Seasonal-naive baseline
                  </span>
                  <span className="chart-legend__item">
                    <span className="chart-legend__band" />
                    80% prediction interval
                  </span>
                </div>
              </div>

              {/* --- Stock position ---------------------------------------- */}
              {risk ? (
                <div>
                  <p className="eyebrow" style={{ marginBottom: 'var(--space-2)' }}>
                    Stock position &amp; recommendation
                  </p>
                  <div className="stat-grid">
                    <div className="stat">
                      <div className="stat__label">On hand</div>
                      <div className="stat__value tabular">
                        {formatUnits(risk.on_hand_units)}
                      </div>
                      <div className="stat__hint">
                        as of {formatDate(risk.inventory_as_of)}
                      </div>
                    </div>
                    <div className="stat">
                      <div className="stat__label">On order</div>
                      <div className="stat__value tabular">
                        {formatUnits(risk.on_order_units)}
                      </div>
                      <div className="stat__hint">
                        {formatUnits(risk.lead_time_days)}-day lead time
                      </div>
                    </div>
                    <div className="stat">
                      <div className="stat__label">Cover</div>
                      <div className="stat__value tabular">
                        {formatWeeks(risk.cover_weeks)}
                      </div>
                      <div className="stat__hint">at the forecast rate</div>
                    </div>
                    <div className="stat">
                      <div className="stat__label">Demand over lead time</div>
                      <div className="stat__value tabular">
                        {formatUnits(risk.forecast_lead_time_units)}
                      </div>
                      <div className="stat__hint">
                        + {formatUnits(risk.safety_stock_units)} safety stock
                      </div>
                    </div>
                    <div className="stat">
                      <div className="stat__label">Recommended order</div>
                      <div
                        className="stat__value tabular"
                        style={{
                          color:
                            risk.recommended_order_units > 0
                              ? 'var(--signal-critical)'
                              : 'var(--ink)',
                        }}
                      >
                        {risk.recommended_order_units > 0
                          ? formatUnits(risk.recommended_order_units)
                          : 'None'}
                      </div>
                      <div className="stat__hint">
                        to a {(risk.service_level * 100).toFixed(0)}% service level
                      </div>
                    </div>
                    <div className="stat">
                      <div className="stat__label">Excess held</div>
                      <div className="stat__value tabular">
                        {risk.excess_units > 0 ? formatUnits(risk.excess_units) : 'None'}
                      </div>
                      <div className="stat__hint">beyond a 12-week cover</div>
                    </div>
                  </div>
                </div>
              ) : null}

              {/* --- Money -------------------------------------------------- */}
              {risk ? (
                <div>
                  <p className="eyebrow" style={{ marginBottom: 'var(--space-2)' }}>
                    What it is worth
                  </p>
                  <div className="stat-grid">
                    <div className="stat">
                      <div className="stat__label">Sales at risk</div>
                      <div
                        className="stat__value tabular"
                        style={{ color: 'var(--signal-critical)' }}
                      >
                        {formatInr(risk.revenue_at_risk)}
                      </div>
                      <div className="stat__hint">
                        {formatUnits(risk.expected_lost_units)} units likely unmet
                      </div>
                    </div>
                    <div className="stat">
                      <div className="stat__label">Margin at risk</div>
                      <div className="stat__value tabular">
                        {formatInr(risk.margin_at_risk)}
                      </div>
                      <div className="stat__hint">
                        at {formatInrExact(detail.list_price - detail.unit_cost)} per unit
                      </div>
                    </div>
                    <div className="stat">
                      <div className="stat__label">Capital locked</div>
                      <div
                        className="stat__value tabular"
                        style={{ color: 'var(--signal-warn-ink)' }}
                      >
                        {formatInr(risk.locked_capital)}
                      </div>
                      <div className="stat__hint">
                        at {formatInrExact(detail.unit_cost)} cost per unit
                      </div>
                    </div>
                    <div className="stat">
                      <div className="stat__label">Forecast confidence</div>
                      <div className="stat__value" style={{ textTransform: 'capitalize' }}>
                        {risk.forecast_confidence}
                      </div>
                      <div className="stat__hint">
                        {risk.forecast_confidence === 'high'
                          ? 'ample history, stable demand'
                          : risk.forecast_confidence === 'medium'
                            ? 'demand is erratic'
                            : 'too little history to be sure'}
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="callout callout--warn">
                  <span className="callout__mark" aria-hidden="true">
                    !
                  </span>
                  <span>
                    No inventory snapshot exists for this product, so stock risk could not be
                    scored. The forecast above is still valid.
                  </span>
                </div>
              )}
            </>
          )}
    </>
  );
}

interface SkuDetailProps {
  skuId: string;
  detail: SkuDetailData | null;
  loading: boolean;
  error: string | null;
  onClose: () => void;
  onRetry: () => void;
}

/** Slide-in quick look, used from the Risk Dashboard's grid and worklist. */
export function SkuDetailDrawer({
  skuId,
  detail,
  loading,
  error,
  onClose,
  onRetry,
}: SkuDetailProps) {
  // Escape closes the drawer — expected of any overlay, and the only way out
  // for a keyboard user who never reaches the close button.
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  return (
    <>
      <div className="drawer-backdrop" onClick={onClose} aria-hidden="true" />
      <aside
        className="drawer"
        role="dialog"
        aria-modal="true"
        aria-label={`Detail for product ${skuId}`}
      >
        <div className="drawer__head">
          <div>
            <div className="drawer__title">{skuId}</div>
            <div className="drawer__subtitle">
              {detail ? `${detail.category} · ${detail.subcategory}` : 'Loading…'}
            </div>
          </div>
          <button
            type="button"
            className="drawer__close"
            onClick={onClose}
            aria-label="Close detail panel"
          >
            ×
          </button>
        </div>

        <div className="drawer__body">
          <SkuDetailContent detail={detail} loading={loading} error={error} onRetry={onRetry} />
        </div>
      </aside>
    </>
  );
}

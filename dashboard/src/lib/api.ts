/**
 * Typed client for the FORESIGHT scoring service.
 *
 * Every response from the API is wrapped in the same envelope, so unwrapping and
 * error translation happen exactly once, here. Components receive either data or
 * a thrown `ApiError` — never a half-parsed response they have to inspect.
 *
 * URLs are relative. In development Vite proxies `/api` to the local service; in
 * production that same service serves these assets. One code path, both
 * environments.
 */

/* ========================================================================== */
/* Wire types — mirror service/models.py                                       */
/* ========================================================================== */

export type RiskAction = 'reorder_now' | 'markdown_clear' | 'watch_volatile' | 'healthy';

export interface Envelope<T> {
  success: boolean;
  data: T | null;
  error: { code: string; message: string; details?: Record<string, unknown> } | null;
}

export interface SkuSummary {
  sku_id: string;
  category: string;
  subcategory: string;
}

export interface ForecastPoint {
  week_starting: string;
  horizon: number;
  forecast_units: number;
  lower_units: number;
  upper_units: number;
}

export interface HistoryPoint {
  week_starting: string;
  units: number;
  revenue: number;
  promo_days: number;
}

export interface BacktestPoint {
  week_starting: string;
  actual: number;
  forecast: number;
  baseline: number;
  lower: number;
  upper: number;
}

export interface RiskRecord {
  sku_id: string;
  category: string;
  subcategory: string;

  on_hand_units: number;
  on_order_units: number;
  available_units: number;
  lead_time_days: number;
  inventory_as_of: string;

  forecast_lead_time_units: number;
  forecast_horizon_units: number;
  cover_weeks: number;

  stockout_score: number;
  overstock_score: number;
  stockout_level: string;
  overstock_level: string;

  action: RiskAction;
  action_label: string;
  action_rationale: string;

  recommended_order_units: number;
  safety_stock_units: number;
  expected_lost_units: number;
  excess_units: number;
  service_level: number;

  revenue_at_risk: number;
  margin_at_risk: number;
  locked_capital: number;
  value_at_stake: number;
  priority_rank: number;

  forecast_confidence: 'high' | 'medium' | 'low';
}

export interface SkuDetail {
  sku_id: string;
  category: string;
  subcategory: string;
  unit_cost: number;
  list_price: number;
  history: HistoryPoint[];
  backtest: BacktestPoint[];
  forecast: ForecastPoint[];
  risk: RiskRecord | null;
}

export interface PortfolioSummary {
  origin_week: string;
  horizon_weeks: number;
  model: string;
  total_skus: number;
  reorder_now_skus: number;
  markdown_skus: number;
  watch_skus: number;
  healthy_skus: number;
  low_confidence_skus: number;
  revenue_at_risk_total: number;
  margin_at_risk_total: number;
  locked_capital_total: number;
  expected_lost_units_total: number;
  excess_units_total: number;
  reorder_now_order_units: number;
  top_10_share_of_revenue_at_risk: number;
}

export interface AccuracySummary {
  selected_model: string;
  selected_wape: number;
  baseline_wape: number;
  naive_wape: number;
  gbm_wape: number;
  ensemble_wape: number | null;
  improvement_vs_baseline: number;
  interval_coverage: number;
  interval_coverage_target: number;
  bias_relative: number;
  folds: number;
  test_observations: number;
  horizon_weeks: number;
  by_horizon: Array<Record<string, number | string>>;
  by_category: Array<Record<string, number | string>>;
  by_regime: Array<Record<string, number | string>>;
}

/** One holdout week: total demand against total forecast. */
export interface HoldoutWeek {
  week: string;
  actual: number;
  predicted: number;
  baseline: number;
  lower: number;
  upper: number;
  skus: number;
  covered: number;
  error_pct: number | null;
}

/** Per-SKU scorecard over the holdout period. */
export interface HoldoutSku {
  sku_id: string;
  category: string;
  weeks: number;
  actual: number;
  predicted: number;
  abs_error: number;
  coverage: number;
  wape: number | null;
  accuracy: number | null;
  bias: number | null;
}

/**
 * The chronological 70/30 test.
 *
 * Distinct from `AccuracySummary`, which reports the rolling-origin backtest.
 * This is one split, one training run and a full year of unseen weeks.
 */
export interface HoldoutSummary {
  available: boolean;
  reason: string | null;
  split_week: string | null;
  holdout_start: string | null;
  holdout_end: string | null;
  train_weeks: number;
  holdout_weeks: number;
  train_fraction: number;
  observations: number;
  model_name: string;
  model_wape: number;
  baseline_wape: number;
  improvement_vs_baseline: number;
  bias_relative: number;
  interval_coverage: number;
  interval_coverage_target: number;
  weekly: HoldoutWeek[];
  by_category: Array<Record<string, number | string>>;
  by_horizon: Array<Record<string, number | string>>;
  skus: HoldoutSku[];
}

export interface GridPoint {
  sku_id: string;
  category: string;
  subcategory: string;
  stockout_score: number;
  overstock_score: number;
  action: RiskAction;
  action_label: string;
  value_at_stake: number;
  revenue_at_risk: number;
  locked_capital: number;
  cover_weeks: number;
  available_units: number;
  forecast_confidence: 'high' | 'medium' | 'low';
}

export interface MetricBlock {
  n_observations: number;
  total_actual: number;
  total_predicted: number;
  wape: number;
  mape: number;
  mape_coverage: number;
  bias_relative: number;
  mae: number;
  rmse: number;
}

export interface EvaluationBreakdown {
  key: string;
  n_observations: number;
  total_actual: number;
  wape: number;
  baseline_wape: number | null;
  bias_relative: number;
}

export interface ErrorContributor {
  sku_id: string;
  n_observations: number;
  total_actual: number;
  total_forecast: number;
  absolute_error: number;
  share_of_total_error: number;
}

export interface EvaluationResponse {
  rows_submitted: number;
  rows_matched: number;
  rows_unmatched: number;
  unmatched_examples: string[];
  weeks_covered: string[];
  source_counts: Record<string, number>;
  model: string;
  accuracy: MetricBlock;
  baseline: MetricBlock | null;
  baseline_row_coverage: number;
  improvement_vs_baseline: number | null;
  interval_coverage: number;
  interval_coverage_target: number;
  by_horizon: EvaluationBreakdown[];
  by_category: EvaluationBreakdown[];
  worst_contributors: ErrorContributor[];
}

export interface ReadyState {
  ready: boolean;
  artifacts_loaded: string[];
  artifacts_missing: string[];
  model: string | null;
  origin_week: string | null;
  skus: number;
}

/* ========================================================================== */
/* Client                                                                      */
/* ========================================================================== */

/** An error the UI can render directly. `code` distinguishes recoverable cases. */
export class ApiError extends Error {
  readonly code: string;
  readonly status: number;

  constructor(message: string, code: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.status = status;
  }
}

/** Requests are aborted rather than left hanging if the service stalls. */
const REQUEST_TIMEOUT_MS = 20_000;

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  let response: Response;
  try {
    response = await fetch(path, {
      ...init,
      signal: controller.signal,
      headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    });
  } catch (cause) {
    clearTimeout(timeout);
    if (cause instanceof DOMException && cause.name === 'AbortError') {
      throw new ApiError(
        'The scoring service did not respond in time. It may still be starting up.',
        'TIMEOUT',
        0,
      );
    }
    throw new ApiError(
      'Could not reach the scoring service. Check that it is running on port 8000.',
      'NETWORK_ERROR',
      0,
    );
  }
  clearTimeout(timeout);

  let payload: Envelope<T>;
  try {
    payload = (await response.json()) as Envelope<T>;
  } catch {
    throw new ApiError(
      `The service returned a response that could not be read (HTTP ${response.status}).`,
      'BAD_RESPONSE',
      response.status,
    );
  }

  if (!response.ok || !payload.success || payload.data === null) {
    throw new ApiError(
      payload.error?.message ?? `Request failed with HTTP ${response.status}.`,
      payload.error?.code ?? 'UNKNOWN',
      response.status,
    );
  }

  return payload.data;
}

/**
 * Upload one or more CSVs for scoring.
 *
 * Kept out of `request()` because the browser must set its own multipart
 * `Content-Type` with the boundary; overriding it produces an unparseable body.
 *
 * Every file goes under the same `files` field. The service decides what each
 * one is from its own headers, so the caller does not have to label them and
 * cannot mislabel them.
 */
async function upload<T>(path: string, files: File[]): Promise<T> {
  const body = new FormData();
  for (const file of files) body.append('files', file);

  const controller = new AbortController();
  // Longer than a normal request: a large file has to be parsed and joined.
  const timeout = setTimeout(() => controller.abort(), 60_000);

  let response: Response;
  try {
    response = await fetch(path, { method: 'POST', body, signal: controller.signal });
  } catch (cause) {
    clearTimeout(timeout);
    if (cause instanceof DOMException && cause.name === 'AbortError') {
      throw new ApiError('The upload timed out. Try a smaller file.', 'TIMEOUT', 0);
    }
    throw new ApiError('Could not reach the scoring service.', 'NETWORK_ERROR', 0);
  }
  clearTimeout(timeout);

  let payload: Envelope<T>;
  try {
    payload = (await response.json()) as Envelope<T>;
  } catch {
    throw new ApiError(
      `The service returned an unreadable response (HTTP ${response.status}).`,
      'BAD_RESPONSE',
      response.status,
    );
  }

  if (!response.ok || !payload.success || payload.data === null) {
    throw new ApiError(
      payload.error?.message ?? `Upload failed with HTTP ${response.status}.`,
      payload.error?.code ?? 'UNKNOWN',
      response.status,
    );
  }
  return payload.data;
}

/** `/ready` is not enveloped — it must answer even when nothing else can. */
export async function fetchReady(): Promise<ReadyState> {
  const response = await fetch('/ready');
  if (!response.ok) {
    throw new ApiError('The scoring service is not reachable.', 'NETWORK_ERROR', response.status);
  }
  return (await response.json()) as ReadyState;
}

export const api = {
  summary: () => request<PortfolioSummary>('/api/summary'),
  accuracy: () => request<AccuracySummary>('/api/accuracy'),
  grid: () => request<GridPoint[]>('/api/grid'),
  categories: () => request<string[]>('/api/categories'),
  skus: () => request<SkuSummary[]>('/api/skus'),
  sku: (skuId: string) => request<SkuDetail>(`/api/sku/${encodeURIComponent(skuId)}`),
  holdout: () => request<HoldoutSummary>('/api/holdout'),
  evaluateUpload: (files: File[]) => upload<EvaluationResponse>('/api/evaluate/upload', files),
  risk: (params: {
    action?: RiskAction | '';
    category?: string;
    search?: string;
    limit?: number;
    offset?: number;
  }) => {
    const query = new URLSearchParams();
    if (params.action) query.set('action', params.action);
    if (params.category) query.set('category', params.category);
    if (params.search) query.set('search', params.search);
    query.set('limit', String(params.limit ?? 200));
    query.set('offset', String(params.offset ?? 0));
    return request<RiskRecord[]>(`/api/risk?${query.toString()}`);
  },
};

/* ========================================================================== */
/* Presentation constants shared across components                             */
/* ========================================================================== */

/**
 * Colour, icon and label per action.
 *
 * The icon exists so risk is never communicated by colour alone — required for
 * the rust/moss pairing to remain readable with red-green colour blindness.
 */
export const ACTION_STYLE: Record<
  RiskAction,
  {
    label: string;
    short: string;
    colour: string;
    soft: string;
    ink: string;
    border: string;
    icon: string;
  }
> = {
  reorder_now: {
    label: 'Reorder now',
    short: 'Reorder',
    colour: 'var(--signal-critical)',
    soft: 'var(--signal-critical-soft)',
    ink: 'var(--signal-critical)',
    border: 'var(--signal-critical-border)',
    icon: '▲',
  },
  markdown_clear: {
    label: 'Markdown / clear',
    short: 'Markdown',
    colour: 'var(--signal-warn)',
    soft: 'var(--signal-warn-soft)',
    ink: 'var(--signal-warn-ink)',
    border: 'var(--signal-warn-border)',
    icon: '▼',
  },
  watch_volatile: {
    label: 'Watch / volatile',
    short: 'Watch',
    colour: 'var(--signal-watch)',
    soft: 'var(--signal-watch-soft)',
    ink: 'var(--signal-watch)',
    border: 'var(--signal-watch-border)',
    icon: '◆',
  },
  healthy: {
    label: 'Healthy',
    short: 'Healthy',
    colour: 'var(--signal-ok)',
    soft: 'var(--signal-ok-soft)',
    ink: 'var(--signal-ok)',
    border: 'var(--signal-ok-border)',
    icon: '●',
  },
};

export const ACTION_ORDER: RiskAction[] = [
  'reorder_now',
  'markdown_clear',
  'watch_volatile',
  'healthy',
];

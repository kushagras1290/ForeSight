/**
 * Formatting helpers.
 *
 * The audience is NorthBay's operations and finance teams in India, so currency
 * is rendered in lakh and crore rather than thousands and millions. "Rs 1.24 Cr"
 * lands instantly with that reader; "12,400,000" has to be counted.
 */

const INR_LAKH = 100_000;
const INR_CRORE = 10_000_000;

/** Compact rupee figure for KPI tiles and table cells. */
export function formatInr(value: number, decimals = 2): string {
  if (!Number.isFinite(value)) return '—';
  const sign = value < 0 ? '-' : '';
  const magnitude = Math.abs(value);

  if (magnitude >= INR_CRORE) {
    return `${sign}₹${(magnitude / INR_CRORE).toFixed(decimals)} Cr`;
  }
  if (magnitude >= INR_LAKH) {
    return `${sign}₹${(magnitude / INR_LAKH).toFixed(decimals)} L`;
  }
  return `${sign}₹${Math.round(magnitude).toLocaleString('en-IN')}`;
}

/** Full rupee figure, for tooltips where precision matters. */
export function formatInrExact(value: number): string {
  if (!Number.isFinite(value)) return '—';
  return `₹${Math.round(value).toLocaleString('en-IN')}`;
}

export function formatUnits(value: number, decimals = 0): string {
  if (!Number.isFinite(value)) return '—';
  return value.toLocaleString('en-IN', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

export function formatPercent(value: number, decimals = 1): string {
  if (!Number.isFinite(value)) return '—';
  return `${(value * 100).toFixed(decimals)}%`;
}

/** Signed percentage, for improvement figures where direction is the point. */
export function formatSignedPercent(value: number, decimals = 1): string {
  if (!Number.isFinite(value)) return '—';
  const sign = value > 0 ? '+' : '';
  return `${sign}${(value * 100).toFixed(decimals)}%`;
}

export function formatWeeks(value: number): string {
  if (!Number.isFinite(value)) return '—';
  // The risk layer caps unbounded cover at 999 for a SKU with no forward
  // demand. Showing "999 wks" implies a precision that does not exist.
  if (value >= 520) return '> 10 yrs';
  if (value >= 104) return `${(value / 52).toFixed(1)} yrs`;
  return `${value.toFixed(1)} wks`;
}

const DATE_FORMAT = new Intl.DateTimeFormat('en-GB', {
  day: '2-digit',
  month: 'short',
  year: 'numeric',
});

const SHORT_DATE_FORMAT = new Intl.DateTimeFormat('en-GB', {
  day: '2-digit',
  month: 'short',
});

export function formatDate(iso: string): string {
  const parsed = new Date(iso);
  return Number.isNaN(parsed.getTime()) ? '—' : DATE_FORMAT.format(parsed);
}

export function formatShortDate(iso: string): string {
  const parsed = new Date(iso);
  return Number.isNaN(parsed.getTime()) ? '—' : SHORT_DATE_FORMAT.format(parsed);
}

/** WAPE is stored as a ratio; the client reads it as an error percentage. */
export function formatWape(value: number): string {
  return formatPercent(value, 1);
}

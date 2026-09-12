/**
 * Application header.
 *
 * Carries the three facts an ops user needs before trusting anything below it:
 * which week the plan was cut from, which model produced it, and how accurate
 * that model was on backtest. Accuracy is shown up front rather than buried in a
 * tab, because a planner acting on a number is entitled to know its track record
 * without going looking for it.
 */

import type { AccuracySummary, PortfolioSummary } from '../lib/api';
import { formatDate, formatSignedPercent, formatWape } from '../lib/format';

interface HeaderProps {
  summary: PortfolioSummary | null;
  accuracy: AccuracySummary | null;
  /** Opens the sidebar overlay on narrow viewports. Omitted on the loading/error shells. */
  onMenuToggle?: () => void;
}

const MODEL_LABELS: Record<string, string> = {
  adaptive_ensemble: 'Adaptive Ensemble',
  lightgbm: 'LightGBM',
  seasonal_naive: 'Seasonal-naive',
};

export function Header({ summary, accuracy, onMenuToggle }: HeaderProps) {
  const modelLabel = summary ? (MODEL_LABELS[summary.model] ?? summary.model) : '—';

  return (
    <header className="header">
      <div className="shell header__inner">
        <div className="header__brand">
          {onMenuToggle ? (
            <button
              type="button"
              className="header__menu-toggle"
              onClick={onMenuToggle}
              aria-label="Toggle navigation menu"
            >
              ☰
            </button>
          ) : null}
          <span className="header__mark">FORESIGHT</span>
          <span className="header__client">
            NorthBay Living · Demand &amp; Inventory Intelligence
          </span>
        </div>

        <div className="header__meta">
          {summary ? (
            <div className="header__stat">
              <span className="header__stat-label">Plan week</span>
              <span className="header__stat-value">{formatDate(summary.origin_week)}</span>
            </div>
          ) : null}

          {summary ? (
            <div className="header__stat">
              <span className="header__stat-label">Horizon</span>
              <span className="header__stat-value">{summary.horizon_weeks} weeks</span>
            </div>
          ) : null}

          {accuracy ? (
            <div className="header__stat">
              <span className="header__stat-label">Forecast error</span>
              <span className="header__stat-value">
                {formatWape(accuracy.selected_wape)}{' '}
                <span style={{ color: 'var(--harbour-300)', fontWeight: 500 }}>
                  ({formatSignedPercent(-accuracy.improvement_vs_baseline)} vs baseline)
                </span>
              </span>
            </div>
          ) : null}

          <span className="header__badge" title="The model selected by the rolling-origin backtest">
            {modelLabel}
          </span>
        </div>
      </div>
    </header>
  );
}

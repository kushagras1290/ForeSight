/**
 * Filter and legend in one slim row.
 *
 * Replaces four stacked cards that repeated numbers already shown above and ate
 * a third of the screen. As a single row of pills it does the same two jobs —
 * filtering the worklist and keying the colours on the grid — in one line.
 */

import { ACTION_ORDER, ACTION_STYLE, type PortfolioSummary, type RiskAction } from '../lib/api';

interface FilterBarProps {
  summary: PortfolioSummary;
  active: RiskAction | '';
  onChange: (action: RiskAction | '') => void;
}

export function FilterBar({ summary, active, onChange }: FilterBarProps) {
  const counts: Record<RiskAction, number> = {
    reorder_now: summary.reorder_now_skus,
    markdown_clear: summary.markdown_skus,
    watch_volatile: summary.watch_skus,
    healthy: summary.healthy_skus,
  };

  return (
    <div className="filter-bar" role="group" aria-label="Filter by recommended action">
      <button
        type="button"
        className="filter-pill"
        aria-pressed={active === ''}
        onClick={() => onChange('')}
      >
        All
        <span className="filter-pill__count">{summary.total_skus}</span>
      </button>

      {ACTION_ORDER.map((action) => {
        const style = ACTION_STYLE[action];
        const isActive = active === action;
        return (
          <button
            key={action}
            type="button"
            className="filter-pill"
            aria-pressed={isActive}
            // Clicking an active pill clears it, so "All" is never the only way back.
            onClick={() => onChange(isActive ? '' : action)}
            style={
              {
                '--pill-colour': style.colour,
                '--pill-soft': style.soft,
                '--pill-ink': style.ink,
                '--pill-border': style.border,
              } as React.CSSProperties
            }
          >
            <span className="filter-pill__icon" aria-hidden="true">
              {style.icon}
            </span>
            {style.short}
            <span className="filter-pill__count">{counts[action]}</span>
          </button>
        );
      })}
    </div>
  );
}

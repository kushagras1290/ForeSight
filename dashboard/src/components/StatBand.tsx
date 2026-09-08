/**
 * The headline figures, as a quiet band rather than a row of boxes.
 *
 * Three numbers, not five. The two that matter to Finance — revenue exposed and
 * cash tied up — plus one line summarising the week's workload. Boxing each one
 * added five borders, five shadows and five fills competing with the chart that
 * follows; hairline dividers do the same job of separating them and nothing more.
 *
 * Revenue at risk and locked capital are never summed: one is revenue that may
 * not happen, the other is cash already spent.
 */

import type { PortfolioSummary } from '../lib/api';
import { formatInr, formatUnits } from '../lib/format';

interface StatBandProps {
  summary: PortfolioSummary;
}

export function StatBand({ summary }: StatBandProps) {
  const stats = [
    {
      label: 'Sales at risk',
      value: formatInr(summary.revenue_at_risk_total),
      detail: `${summary.reorder_now_skus} products likely to run out`,
      accent: 'var(--signal-critical)',
    },
    {
      label: 'Capital locked',
      value: formatInr(summary.locked_capital_total),
      detail: `${formatUnits(summary.excess_units_total)} excess units held`,
      accent: 'var(--signal-warn-ink)',
    },
    {
      label: 'This week',
      value: `${summary.reorder_now_skus} + ${summary.markdown_skus}`,
      detail: `to reorder and clear · ${summary.watch_skus} to review`,
      accent: 'var(--harbour-700)',
    },
  ];

  return (
    <div className="stat-band">
      {stats.map((stat) => (
        <div key={stat.label} className="stat-band__item">
          <p className="eyebrow">{stat.label}</p>
          <p className="stat-band__value" style={{ color: stat.accent }}>
            {stat.value}
          </p>
          <p className="stat-band__detail">{stat.detail}</p>
        </div>
      ))}
    </div>
  );
}

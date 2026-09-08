/**
 * The prioritised reorder / markdown list (D5 acceptance criterion 2).
 *
 * Six columns, not nine. The earlier version showed stockout and overstock as
 * two separate score bars, but a product is only ever in trouble in one
 * direction at a time — so one of the two was always reading zero and taking up
 * a column to say nothing. They are collapsed into a single risk bar coloured by
 * whichever direction is actually driving the recommendation.
 *
 * Rank and confidence came out for the same reason: rank is just row order, and
 * confidence matters when you are deciding on one product, which is the detail
 * panel's job. A low-confidence product is marked with a dot next to its code
 * so the signal survives without a column.
 *
 * Every risk state is colour **plus** icon **plus** text, so nothing depends on
 * distinguishing rust from moss.
 */

import { ACTION_STYLE, type RiskRecord } from '../lib/api';
import { formatInr, formatUnits, formatWeeks } from '../lib/format';
import { Empty, Loading } from './States';

interface WorklistProps {
  rows: RiskRecord[];
  loading: boolean;
  selectedSku: string | null;
  onSelect: (skuId: string) => void;
  onClearFilters: () => void;
  hasFilters: boolean;
}

/** The risk actually driving this row's recommendation. */
function dominantRisk(row: RiskRecord): { value: number; colour: string; label: string } {
  return row.overstock_score > row.stockout_score
    ? { value: row.overstock_score, colour: 'var(--signal-warn)', label: 'overstock' }
    : { value: row.stockout_score, colour: 'var(--signal-critical)', label: 'stockout' };
}

/** What this row is asking the planner to do, in units. */
function quantity(row: RiskRecord): string {
  if (row.recommended_order_units > 0) return `+${formatUnits(row.recommended_order_units)}`;
  if (row.excess_units > 0) return `−${formatUnits(row.excess_units)}`;
  return '—';
}

export function Worklist({
  rows,
  loading,
  selectedSku,
  onSelect,
  onClearFilters,
  hasFilters,
}: WorklistProps) {
  if (loading) {
    return <Loading rows={8} label="Loading the worklist" />;
  }

  if (rows.length === 0) {
    // An empty list is usually good news, so it is only framed as a problem
    // when a filter is what emptied it.
    return hasFilters ? (
      <Empty
        title="Nothing matches these filters"
        body="No product meets every condition you have set."
        action={
          <button type="button" className="button button--ghost" onClick={onClearFilters}>
            Clear filters
          </button>
        }
      />
    ) : (
      <Empty
        mark="✓"
        title="Nothing needs action this week"
        body="Every product is holding enough stock to cover the forecast without carrying excess."
      />
    );
  }

  return (
    <div className="table-scroll">
      <table className="table">
        <caption className="visually-hidden">
          Products requiring action, ordered by the rupee value at stake.
        </caption>
        <thead>
          <tr>
            <th scope="col">Product</th>
            <th scope="col">Action</th>
            <th scope="col" className="numeric">
              Cover
            </th>
            <th scope="col" className="numeric">
              Risk
            </th>
            <th scope="col" className="numeric">
              Units
            </th>
            <th scope="col" className="numeric">
              At stake
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const style = ACTION_STYLE[row.action];
            const risk = dominantRisk(row);
            return (
              <tr
                key={row.sku_id}
                data-selected={selectedSku === row.sku_id}
                onClick={() => onSelect(row.sku_id)}
                tabIndex={0}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault();
                    onSelect(row.sku_id);
                  }
                }}
              >
                <td>
                  <span className="sku-cell">
                    <span className="sku-cell__id">
                      {row.sku_id}
                      {row.forecast_confidence === 'low' ? (
                        <span
                          className="sku-cell__flag"
                          title="Low forecast confidence — thin history"
                          aria-label="Low forecast confidence"
                        >
                          ·
                        </span>
                      ) : null}
                    </span>
                    <span className="sku-cell__cat">{row.subcategory}</span>
                  </span>
                </td>
                <td>
                  <span
                    className="pill"
                    style={
                      {
                        '--pill-colour': style.colour,
                        '--pill-soft': style.soft,
                        '--pill-ink': style.ink,
                      } as React.CSSProperties
                    }
                  >
                    <span className="pill__icon" aria-hidden="true">
                      {style.icon}
                    </span>
                    {style.short}
                  </span>
                </td>
                <td className="numeric">{formatWeeks(row.cover_weeks)}</td>
                <td className="numeric">
                  <span className="score">
                    <span className="score__value">{(risk.value * 100).toFixed(0)}%</span>
                    <span className="score__bar">
                      <span
                        className="score__fill"
                        style={
                          {
                            width: `${Math.min(100, Math.max(0, risk.value * 100))}%`,
                            '--score-colour': risk.colour,
                          } as React.CSSProperties
                        }
                      />
                    </span>
                  </span>
                  <span className="visually-hidden"> {risk.label} risk</span>
                </td>
                <td className="numeric mono">{quantity(row)}</td>
                <td className="numeric at-stake">
                  {row.value_at_stake > 0 ? formatInr(row.value_at_stake) : '—'}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

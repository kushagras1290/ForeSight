/**
 * The decisioning view from section 8.2 of the brief.
 *
 * Every SKU is a point on a stockout-versus-overstock grid, sized by the rupee
 * value at stake, so the ops team can triage 200 products at a glance instead of
 * reading a table row by row.
 *
 * Built as hand-written SVG rather than with a charting library for two reasons:
 * the quadrant framing is the entire point and needs exact control, and text
 * inside a scaled `viewBox` would scale with it, leaving labels at unpredictable
 * sizes. The container is measured instead and the chart drawn at real pixels,
 * so typography stays fixed at every width.
 *
 * ACCESSIBILITY
 * Position carries the meaning here — x is overstock, y is stockout — so colour
 * is redundant rather than load-bearing, and the red/green pairing is safe. The
 * table below the grid conveys the same information without any reliance on
 * either channel.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { ACTION_STYLE, type GridPoint, type RiskAction } from '../lib/api';
import { formatInr, formatUnits, formatWeeks } from '../lib/format';

const MARGIN = { top: 24, right: 22, bottom: 46, left: 54 };
const MIN_RADIUS = 3;
const MAX_RADIUS = 16;

/**
 * Height from width, capped.
 *
 * A fixed aspect ratio works at 900px and produces a 1100px-tall chart on a wide
 * monitor, which pushes the worklist entirely below the fold. The cap keeps the
 * grid and the first rows of the list visible together, which is how the screen
 * is actually used.
 */
const MIN_HEIGHT = 320;
const MAX_HEIGHT = 430;

function plotHeightFor(width: number): number {
  // Deliberately letterboxed. Both axes are probabilities that pile up at 0 and
  // 1, so a square plot spends most of its area on empty middle ground.
  return Math.round(Math.min(MAX_HEIGHT, Math.max(MIN_HEIGHT, width * 0.34)));
}

/** Deterministic 0..1 hash, so jitter is stable across renders and reloads. */
function hashUnit(value: string): number {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return ((hash >>> 0) % 10000) / 10000;
}

function useMeasuredWidth<T extends HTMLElement>() {
  const ref = useRef<T | null>(null);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) setWidth(entry.contentRect.width);
    });
    observer.observe(element);
    setWidth(element.getBoundingClientRect().width);
    return () => observer.disconnect();
  }, []);

  return [ref, width] as const;
}

interface DecisionGridProps {
  points: GridPoint[];
  threshold: number;
  activeAction: RiskAction | '';
  selectedSku: string | null;
  onSelect: (skuId: string) => void;
}

interface HoverState {
  point: GridPoint;
  x: number;
  y: number;
}

export function DecisionGrid({
  points,
  threshold,
  activeAction,
  selectedSku,
  onSelect,
}: DecisionGridProps) {
  const [wrapRef, width] = useMeasuredWidth<HTMLDivElement>();
  const [hover, setHover] = useState<HoverState | null>(null);

  const height = plotHeightFor(width);
  const plotWidth = Math.max(10, width - MARGIN.left - MARGIN.right);
  const plotHeight = Math.max(10, height - MARGIN.top - MARGIN.bottom);

  const scaleX = useCallback(
    (value: number) => MARGIN.left + value * plotWidth,
    [plotWidth],
  );
  // SVG y grows downward; a stockout score of 1 must sit at the top.
  const scaleY = useCallback(
    (value: number) => MARGIN.top + (1 - value) * plotHeight,
    [plotHeight],
  );

  const radiusScale = useMemo(() => {
    const maxValue = points.reduce((peak, point) => Math.max(peak, point.value_at_stake), 0);
    // Square root so area, not radius, encodes value — radius alone exaggerates
    // large SKUs by the square of their true share.
    return (value: number) => {
      if (maxValue <= 0) return MIN_RADIUS;
      const ratio = Math.sqrt(Math.max(value, 0) / maxValue);
      return MIN_RADIUS + ratio * (MAX_RADIUS - MIN_RADIUS);
    };
  }, [points]);

  /**
   * Both scores are probabilities that pile up hard at exactly 0 and 1, so a
   * large share of the catalogue would land on the very same pixel. A small
   * deterministic jitter — under 3% of an axis, and applied only at the extremes
   * — separates them enough to be countable without moving any point into a
   * different quadrant.
   */
  const positioned = useMemo(
    () =>
      points.map((point) => {
        const seedX = hashUnit(`${point.sku_id}:x`);
        const seedY = hashUnit(`${point.sku_id}:y`);
        const atEdgeX = point.overstock_score <= 0.001 || point.overstock_score >= 0.999;
        const atEdgeY = point.stockout_score <= 0.001 || point.stockout_score >= 0.999;
        const nudge = (score: number, seed: number, atEdge: boolean) => {
          if (!atEdge) return score;
          const offset = seed * 0.028 + 0.004;
          return score <= 0.5 ? score + offset : score - offset;
        };
        return {
          point,
          cx: scaleX(nudge(point.overstock_score, seedX, atEdgeX)),
          cy: scaleY(nudge(point.stockout_score, seedY, atEdgeY)),
          r: radiusScale(point.value_at_stake),
        };
      }),
    [points, radiusScale, scaleX, scaleY],
  );

  const ticks = [0, 0.25, 0.5, 0.75, 1];

  const quadrants = [
    {
      key: 'markdown',
      label: 'Markdown / clear',
      x: scaleX(threshold),
      y: scaleY(threshold),
      w: plotWidth * (1 - threshold),
      h: plotHeight * (1 - threshold),
      colour: ACTION_STYLE.markdown_clear.colour,
      anchor: 'end' as const,
      // Labels sit just under the threshold line rather than in the corners:
      // the corners are exactly where the points pile up, so a corner label
      // ends up printed over the data it is describing.
      labelX: scaleX(0.99),
      labelY: scaleY(threshold) + 15,
    },
    {
      key: 'reorder',
      label: 'Reorder now',
      x: MARGIN.left,
      y: MARGIN.top,
      w: plotWidth * threshold,
      h: plotHeight * (1 - threshold),
      colour: ACTION_STYLE.reorder_now.colour,
      // Right-aligned against the vertical divider. The left edge is where the
      // low-overstock points live, so a left-anchored label lands on top of them.
      anchor: 'end' as const,
      labelX: scaleX(threshold) - 10,
      labelY: scaleY(threshold) - 9,
    },
    {
      key: 'healthy',
      label: 'Healthy',
      x: MARGIN.left,
      y: scaleY(threshold),
      w: plotWidth * threshold,
      h: plotHeight * (1 - threshold),
      colour: ACTION_STYLE.healthy.colour,
      anchor: 'end' as const,
      labelX: scaleX(threshold) - 10,
      labelY: scaleY(threshold) + 17,
    },
  ];

  if (width === 0) {
    return <div ref={wrapRef} className="grid-wrap" style={{ minHeight: 340 }} />;
  }

  return (
    <div ref={wrapRef} className="grid-wrap">
      <svg
        className="grid-svg"
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`Decisioning grid plotting ${points.length} products by stockout risk against overstock risk.`}
      >
        {/* Quadrant tints — kept very low so they frame the space without
            competing with the points that carry the data. */}
        {quadrants.map((quadrant) => (
          <rect
            key={quadrant.key}
            x={quadrant.x}
            y={quadrant.y}
            width={quadrant.w}
            height={quadrant.h}
            fill={quadrant.colour}
            opacity={0.045}
          />
        ))}

        {/* Grid lines */}
        {ticks.map((tick) => (
          <g key={`grid-${tick}`}>
            <line
              x1={scaleX(tick)}
              y1={MARGIN.top}
              x2={scaleX(tick)}
              y2={MARGIN.top + plotHeight}
              stroke="var(--rule)"
              strokeWidth={tick === 0 ? 1 : 0.6}
            />
            <line
              x1={MARGIN.left}
              y1={scaleY(tick)}
              x2={MARGIN.left + plotWidth}
              y2={scaleY(tick)}
              stroke="var(--rule)"
              strokeWidth={tick === 0 ? 1 : 0.6}
            />
          </g>
        ))}

        {/* Decision thresholds */}
        <line
          x1={scaleX(threshold)}
          y1={MARGIN.top}
          x2={scaleX(threshold)}
          y2={MARGIN.top + plotHeight}
          stroke="var(--ink-3)"
          strokeWidth={1.1}
          strokeDasharray="5 4"
        />
        <line
          x1={MARGIN.left}
          y1={scaleY(threshold)}
          x2={MARGIN.left + plotWidth}
          y2={scaleY(threshold)}
          stroke="var(--ink-3)"
          strokeWidth={1.1}
          strokeDasharray="5 4"
        />

        {/* Quadrant labels */}
        {quadrants.map((quadrant) => (
          <text
            key={`label-${quadrant.key}`}
            className="grid-quadrant-label"
            x={quadrant.labelX}
            y={quadrant.labelY}
            textAnchor={quadrant.anchor}
            fill={quadrant.colour}
            opacity={0.72}
          >
            {quadrant.label}
          </text>
        ))}

        {/* Ticks */}
        {ticks.map((tick) => (
          <g key={`tick-${tick}`}>
            <text
              className="grid-tick"
              x={scaleX(tick)}
              y={MARGIN.top + plotHeight + 16}
              textAnchor="middle"
            >
              {tick.toFixed(2)}
            </text>
            <text
              className="grid-tick"
              x={MARGIN.left - 9}
              y={scaleY(tick) + 3.5}
              textAnchor="end"
            >
              {tick.toFixed(2)}
            </text>
          </g>
        ))}

        {/* Axis titles */}
        <text
          className="grid-axis-label"
          x={MARGIN.left + plotWidth / 2}
          y={height - 10}
          textAnchor="middle"
        >
          Overstock risk — chance 12 weeks of sales won&rsquo;t clear current stock
        </text>
        <text
          className="grid-axis-label"
          transform={`translate(14, ${MARGIN.top + plotHeight / 2}) rotate(-90)`}
          textAnchor="middle"
        >
          Stockout risk — chance of running out before delivery
        </text>

        {/* Points. Drawn smallest-last so a large SKU never hides a small one. */}
        {positioned
          .slice()
          .sort((left, right) => right.r - left.r)
          .map(({ point, cx, cy, r }) => {
            const style = ACTION_STYLE[point.action];
            const dimmed = activeAction !== '' && point.action !== activeAction;
            const selected = selectedSku === point.sku_id;

            // "Watch" is assigned by forecast volatility, not by position, so a
            // watch product can sit anywhere on the grid — including inside the
            // Healthy quadrant. Drawing it as a ring makes it findable wherever
            // it lands, instead of silently contradicting the quadrant it is in.
            const isWatch = point.action === 'watch_volatile';

            return (
              <g
                key={point.sku_id}
                className="grid-point"
                data-dimmed={dimmed}
                data-selected={selected}
                onMouseEnter={() => setHover({ point, x: cx, y: cy })}
                onMouseLeave={() => setHover(null)}
                onClick={() => onSelect(point.sku_id)}
                role="button"
                tabIndex={-1}
                aria-label={`${point.sku_id}, ${point.action_label}`}
              >
                <circle
                  cx={cx}
                  cy={cy}
                  r={r}
                  fill={style.colour}
                  fillOpacity={isWatch ? 0.12 : selected ? 0.9 : 0.62}
                  stroke={style.colour}
                  strokeWidth={isWatch ? 2 : selected ? 1.8 : 0.9}
                  strokeDasharray={isWatch ? '2.5 2' : undefined}
                />
                {isWatch ? (
                  <circle cx={cx} cy={cy} r={1.6} fill={style.colour} />
                ) : null}
              </g>
            );
          })}
      </svg>

      {hover ? (
        <div className="grid-tooltip" style={{ left: hover.x, top: hover.y }}>
          <div className="grid-tooltip__sku">{hover.point.sku_id}</div>
          <div className="grid-tooltip__row">
            <span>{hover.point.category}</span>
          </div>
          <div className="grid-tooltip__row">
            <span>Action</span>
            <strong style={{ color: ACTION_STYLE[hover.point.action].colour }}>
              {hover.point.action_label}
            </strong>
          </div>
          <div className="grid-tooltip__row">
            <span>Cover</span>
            <strong>{formatWeeks(hover.point.cover_weeks)}</strong>
          </div>
          <div className="grid-tooltip__row">
            <span>On hand + on order</span>
            <strong>{formatUnits(hover.point.available_units)}</strong>
          </div>
          <div className="grid-tooltip__row">
            <span>At stake</span>
            <strong>{formatInr(hover.point.value_at_stake)}</strong>
          </div>
        </div>
      ) : null}

      {/* No colour legend here: the filter pills above already key the colours,
          and repeating them would be the third time the same four labels appear
          on one screen. */}
      <p className="grid-caption">
        Marker area is proportional to the rupees at stake. Products with no forward demand
        sit at the far right — dead stock.{' '}
        <span style={{ color: 'var(--signal-watch)' }}>
          Dashed rings are products flagged for review because their demand is too erratic to
          forecast confidently — they can appear anywhere on the grid.
        </span>{' '}
        Click any marker for its forecast.
      </p>
    </div>
  );
}

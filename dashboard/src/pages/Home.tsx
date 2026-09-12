/**
 * Landing page. Answers "where do things stand" in one glance, then routes
 * the visitor to whichever page actually answers their next question, rather
 * than repeating content that page already owns.
 */

import { Link } from 'react-router-dom';

import { useCoreData } from '../components/Layout';
import { StatBand } from '../components/StatBand';
import { formatDate, formatWape } from '../lib/format';

interface NavCard {
  to: string;
  title: string;
  body: string;
}

const CARDS: NavCard[] = [
  {
    to: '/executive-dashboard',
    title: 'Executive Dashboard',
    body: 'The engagement end to end — problem, data, model, results.',
  },
  {
    to: '/sales-analytics',
    title: 'Sales Analytics',
    body: 'Weekly units and revenue across the portfolio, by category.',
  },
  {
    to: '/product-performance',
    title: 'Product Performance',
    body: 'Every SKU ranked by revenue, units, trend, and risk action.',
  },
  {
    to: '/category-performance',
    title: 'Category Performance',
    body: 'Revenue and risk mix, one category at a time.',
  },
  {
    to: '/demand-forecast',
    title: 'Forecast Dashboard',
    body: 'The forward forecast for any product, and how accurate it has been.',
  },
  {
    to: '/model-benchmark',
    title: 'Model Benchmark',
    body: 'Every forecasting method this project has, scored side by side.',
  },
  {
    to: '/inventory',
    title: 'Inventory Dashboard',
    body: 'Stock position, cover, and a live what-if against a different count.',
  },
  {
    to: '/risk/stockout',
    title: 'Stockout Risk',
    body: 'Products likely to run out over their lead time.',
  },
  {
    to: '/risk/overstock',
    title: 'Overstock',
    body: 'Products likely to be left holding excess stock.',
  },
  {
    to: '/risk/watchlist',
    title: 'Watchlist',
    body: 'Products where the forecast itself is too volatile to call confidently.',
  },
  {
    to: '/promotions',
    title: 'Promotion Dashboard',
    body: 'The fitted promotional uplift and post-promotion dip, per category.',
  },
  {
    to: '/seasonality',
    title: 'Seasonality Dashboard',
    body: "The model's actual seasonal curve, global and per category.",
  },
  {
    to: '/business-insights',
    title: 'Business Insights',
    body: 'Revenue concentration, dead stock, and top movers.',
  },
  {
    to: '/products',
    title: 'Product Details',
    body: 'Look up one SKU: history, forecast, rationale, and what it is worth.',
  },
  {
    to: '/executive-recommendation',
    title: 'Executive Recommendation',
    body: 'What this does not do, and where to take it next.',
  },
];

export function Home() {
  const { core } = useCoreData();

  return (
    <>
      <section className="section">
        <div className="section__head">
          <p className="eyebrow">Project FORESIGHT</p>
          <h2 className="section__title">Demand &amp; inventory intelligence for NorthBay Living</h2>
          {core ? (
            <p className="section__note">
              Plan cut for the week of {formatDate(core.summary.origin_week)},{' '}
              {core.summary.horizon_weeks} weeks forward, current error{' '}
              {formatWape(core.accuracy.selected_wape)}.
            </p>
          ) : (
            <p className="section__note">
              No model has been trained on this instance yet. Run the pipeline against real data
              to see live numbers here — everything below still works to look around.
            </p>
          )}
        </div>
        {core ? <StatBand summary={core.summary} /> : null}
      </section>

      <section className="section">
        <div className="section__head">
          <p className="eyebrow">Go to</p>
          <h2 className="section__title">Everything on this plan</h2>
        </div>
        <div className="card-grid">
          {CARDS.map((card) => (
            <Link key={card.to} to={card.to} className="nav-card">
              <span className="nav-card__title">{card.title}</span>
              <span className="nav-card__body">{card.body}</span>
              <span className="nav-card__arrow" aria-hidden="true">
                →
              </span>
            </Link>
          ))}
        </div>
      </section>
    </>
  );
}

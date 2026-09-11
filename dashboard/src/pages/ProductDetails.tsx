/**
 * Product Details: the full-page version of the SKU detail drawer, reachable
 * on its own URL (`/products/:skuId`) so a product's page can be shared or
 * bookmarked, rather than existing only as transient overlay state.
 */

import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';

import { ApiError, api, type SkuDetail, type SkuSummary } from '../lib/api';
import { SkuDetailContent } from '../components/SkuDetail';
import { Empty, Loading } from '../components/States';

export function ProductDetails() {
  const { skuId } = useParams<{ skuId?: string }>();
  const navigate = useNavigate();

  const [detail, setDetail] = useState<SkuDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = (id: string) => {
    setLoading(true);
    setError(null);
    setDetail(null);
    api
      .sku(id)
      .then(setDetail)
      .catch((cause: unknown) => {
        setError(cause instanceof ApiError ? cause.message : 'Could not load this product.');
      })
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    if (skuId) load(skuId);
  }, [skuId]);

  // --- Picker, shown when no SKU is selected -------------------------------- //
  const [query, setQuery] = useState('');
  const [options, setOptions] = useState<SkuSummary[]>([]);
  const [optionsLoading, setOptionsLoading] = useState(false);

  useEffect(() => {
    if (skuId) return;
    let cancelled = false;
    setOptionsLoading(true);
    const timer = setTimeout(() => {
      api
        .skus()
        .then((result) => {
          if (cancelled) return;
          const needle = query.trim().toLowerCase();
          setOptions(
            needle
              ? result.filter(
                  (sku) =>
                    sku.sku_id.toLowerCase().includes(needle) ||
                    sku.subcategory.toLowerCase().includes(needle),
                )
              : result,
          );
        })
        .catch(() => {
          if (!cancelled) setOptions([]);
        })
        .finally(() => {
          if (!cancelled) setOptionsLoading(false);
        });
    }, 200);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [query, skuId]);

  if (!skuId) {
    return (
      <section className="section">
        <div className="section__head">
          <p className="eyebrow">Product details</p>
          <h2 className="section__title">Look up one product</h2>
          <p className="section__note">
            Search by product code or subcategory, then pick one to see its full history,
            forecast and risk rationale.
          </p>
        </div>

        <div className="toolbar">
          <input
            className="input"
            type="search"
            aria-label="Search products"
            placeholder="Search product code or subcategory…"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            autoFocus
          />
        </div>

        {optionsLoading ? (
          <Loading rows={6} />
        ) : options.length === 0 ? (
          <Empty
            title="No products to show"
            body={
              query
                ? 'Nothing matches that search.'
                : 'This instance has no product catalogue loaded yet. Once real data has been run through the pipeline, products will show up here.'
            }
          />
        ) : (
          <div className="table-scroll">
            <table className="table">
              <thead>
                <tr>
                  <th>Product</th>
                  <th>Category</th>
                  <th>Subcategory</th>
                </tr>
              </thead>
              <tbody>
                {options.slice(0, 100).map((sku) => (
                  <tr key={sku.sku_id} onClick={() => navigate(`/products/${sku.sku_id}`)}>
                    <td className="mono">{sku.sku_id}</td>
                    <td>{sku.category}</td>
                    <td>{sku.subcategory}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    );
  }

  return (
    <section className="section">
      <div className="section__head">
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 'var(--space-3)' }}>
          <h2 className="section__title mono">{skuId}</h2>
          {detail ? (
            <span className="section__note">
              {detail.category} · {detail.subcategory}
            </span>
          ) : null}
        </div>
        <button
          type="button"
          className="button button--ghost"
          style={{ marginTop: 'var(--space-3)' }}
          onClick={() => navigate('/products')}
        >
          ← Choose a different product
        </button>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-5)' }}>
        <SkuDetailContent
          detail={detail}
          loading={loading}
          error={error}
          onRetry={() => load(skuId)}
        />
      </div>
    </section>
  );
}

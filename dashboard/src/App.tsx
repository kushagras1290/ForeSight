/**
 * FORESIGHT planning dashboard (deliverable D5).
 *
 * Public routes (sign in, register, forgot password) sit outside the auth
 * gate; everything else is wrapped in `RequireAuth`, which redirects to
 * /login unless a session is present, then `Layout`, which loads core data
 * once via `useForesightData` and gates every page behind the same
 * readiness/error states.
 *
 * 14 sidebar destinations, each answering one question a stakeholder asks —
 * see `components/Layout.tsx`'s `NAV_GROUPS` for the grouping. Product
 * Details (`/products/:skuId?`) is a drill-down destination linked from
 * every table row rather than its own sidebar entry, superseded by Product
 * Performance as the portfolio-wide entry point.
 */

import { Route, Routes } from 'react-router-dom';

import { Layout } from './components/Layout';
import { RequireAuth } from './components/RequireAuth';
import { Home } from './pages/Home';
import { SalesAnalytics } from './pages/SalesAnalytics';
import { ProductPerformance } from './pages/ProductPerformance';
import { CategoryPerformance } from './pages/CategoryPerformance';
import { DemandForecast } from './pages/DemandForecast';
import { ModelBenchmark } from './pages/ModelBenchmark';
import { InventoryDashboard } from './pages/InventoryDashboard';
import { StockoutRisk } from './pages/StockoutRisk';
import { Overstock } from './pages/Overstock';
import { Watchlist } from './pages/Watchlist';
import { PromotionDashboard } from './pages/PromotionDashboard';
import { SeasonalityDashboard } from './pages/SeasonalityDashboard';
import { BusinessInsights } from './pages/BusinessInsights';
import { ProductDetails } from './pages/ProductDetails';
import { ExecutiveDashboard } from './pages/ExecutiveDashboard';
import { ExecutiveRecommendation } from './pages/ExecutiveRecommendation';
import { TestResults } from './pages/TestResults';
import { Account } from './pages/Account';
import { Login } from './pages/Login';
import { Register } from './pages/Register';
import { ForgotPassword } from './pages/ForgotPassword';

import './styles/theme.css';
import './styles/app.css';
import './styles/auth.css';

export default function App() {
  return (
    <Routes>
      <Route path="login" element={<Login />} />
      <Route path="register" element={<Register />} />
      <Route path="forgot-password" element={<ForgotPassword />} />

      <Route element={<RequireAuth />}>
        <Route element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="executive-dashboard" element={<ExecutiveDashboard />} />
          <Route path="executive-recommendation" element={<ExecutiveRecommendation />} />
          <Route path="sales-analytics" element={<SalesAnalytics />} />
          <Route path="product-performance" element={<ProductPerformance />} />
          <Route path="category-performance" element={<CategoryPerformance />} />
          <Route path="demand-forecast" element={<DemandForecast />} />
          <Route path="model-benchmark" element={<ModelBenchmark />} />
          <Route path="inventory" element={<InventoryDashboard />} />
          <Route path="risk/stockout" element={<StockoutRisk />} />
          <Route path="risk/overstock" element={<Overstock />} />
          <Route path="risk/watchlist" element={<Watchlist />} />
          <Route path="promotions" element={<PromotionDashboard />} />
          <Route path="seasonality" element={<SeasonalityDashboard />} />
          <Route path="business-insights" element={<BusinessInsights />} />
          <Route path="products" element={<ProductDetails />} />
          <Route path="products/:skuId" element={<ProductDetails />} />
          <Route path="test-results" element={<TestResults />} />
          <Route path="account" element={<Account />} />
        </Route>
      </Route>
    </Routes>
  );
}

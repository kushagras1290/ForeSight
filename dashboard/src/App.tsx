/**
 * FORESIGHT planning dashboard (deliverable D5).
 *
 * Public routes (sign in, register, forgot password) sit outside the auth
 * gate; everything else is wrapped in `RequireAuth`, which redirects to
 * /login unless a session is present, then `Layout`, which loads core data
 * once via `useForesightData` and gates every page behind the same
 * readiness/error states. Each page answers one question a stakeholder asks:
 * Home ("where do things stand"), Sales Analytics ("what's selling"), Demand
 * Forecast ("what will we sell, and can I trust it"), Inventory ("what do I
 * have, right now"), Risk Dashboard ("what do I reorder and what do I
 * clear"), Product Details ("tell me about this one SKU"), Executive Summary
 * ("the whole engagement, presented"), and Account ("who am I signed in as").
 */

import { Route, Routes } from 'react-router-dom';

import { Layout } from './components/Layout';
import { RequireAuth } from './components/RequireAuth';
import { Home } from './pages/Home';
import { SalesAnalytics } from './pages/SalesAnalytics';
import { DemandForecast } from './pages/DemandForecast';
import { InventoryDashboard } from './pages/InventoryDashboard';
import { RiskDashboard } from './pages/RiskDashboard';
import { ProductDetails } from './pages/ProductDetails';
import { ExecutiveSummary } from './pages/ExecutiveSummary';
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
          <Route path="sales-analytics" element={<SalesAnalytics />} />
          <Route path="demand-forecast" element={<DemandForecast />} />
          <Route path="inventory" element={<InventoryDashboard />} />
          <Route path="risk" element={<RiskDashboard />} />
          <Route path="products" element={<ProductDetails />} />
          <Route path="products/:skuId" element={<ProductDetails />} />
          <Route path="executive-summary" element={<ExecutiveSummary />} />
          <Route path="test-results" element={<TestResults />} />
          <Route path="account" element={<Account />} />
        </Route>
      </Route>
    </Routes>
  );
}

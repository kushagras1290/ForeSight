import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

import App from './App';

const container = document.getElementById('root');

// Failing loudly here beats a blank page: if the mount point is missing the
// index.html and this entry point have drifted apart, and silence would hide it.
if (!container) {
  throw new Error('Mount point #root was not found in index.html.');
}

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
);

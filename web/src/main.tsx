import React from 'react';
import ReactDOM from 'react-dom/client';
import App from '@/App';
// Self-hosted UI face (cambia-845). Variable weight axis only: one file per
// subset covering 100-900, no width axis, no CDN request.
import '@fontsource-variable/archivo/wght.css';
import './index.css'; // Import Tailwind CSS / global styles
import 'uplot/dist/uPlot.min.css';
import { BrowserRouter } from 'react-router-dom';
import { getAppRoot } from '@/lib/appRoot';

// Through getAppRoot, the same lookup ds/core/Modal uses: a rename in
// index.html reports which two files disagree instead of throwing React's
// "Target container is not a DOM element" (cambia-935, F7).
const mount = getAppRoot();
if (mount) {
  ReactDOM.createRoot(mount).render(
    <React.StrictMode>
      {/* Opts into the v7 behaviors early per React Router's own warning (cambia-958 D7):
          v7_startTransition wraps navigation state updates in React.startTransition;
          v7_relativeSplatPath changes relative link resolution under a nested splat route,
          which this app has none of (its one splat, path="*" in App.tsx, is a top-level
          catch-all with no children), so neither flag changes route behavior here. */}
      <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <App />
      </BrowserRouter>
    </React.StrictMode>
  );
}

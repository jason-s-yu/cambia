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
import { consumeBootIdentity } from '@/lib/tabSession';
import { mintForIntent } from '@/services/devSessionService';
import DevSessionSwitcher from '@/components/dev/DevSessionSwitcher';

// Through getAppRoot, the same lookup ds/core/Modal uses: a rename in
// index.html reports which two files disagree instead of throwing React's
// "Target container is not a DOM element" (cambia-935, F7).
const mount = getAppRoot();
if (mount) {
  // A `?as=` or `#tab=` URL asks this tab to be somebody in particular, and it
  // has to be honoured before anything probes /user/me: the probe carries the
  // tab's token, so a render that starts first answers for the shared cookie
  // identity and the tab shows the wrong player (cambia-1149). With no such
  // parameter this settles on the next microtask and costs nothing. It never
  // rejects; `finally` is there so a future throw still paints the app.
  void consumeBootIdentity(mintForIntent).finally(() => {
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
        {/* Dev builds only: `vite build` folds the flag to false and drops both the
            element and the component with it, so the switcher's markup and copy are
            absent from a production bundle. Mounted outside the router because it
            belongs to the browser tab, not to a route. */}
        {import.meta.env.DEV && <DevSessionSwitcher />}
      </React.StrictMode>
    );
  });
}

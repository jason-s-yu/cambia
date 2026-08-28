import React from 'react';
import ReactDOM from 'react-dom/client';
import App from '@/App';
// Self-hosted UI face (cambia-845). Variable weight axis only: one file per
// subset covering 100-900, no width axis, no CDN request.
import '@fontsource-variable/archivo/wght.css';
import './index.css'; // Import Tailwind CSS / global styles
import 'uplot/dist/uPlot.min.css';
import { BrowserRouter } from 'react-router-dom';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>
);
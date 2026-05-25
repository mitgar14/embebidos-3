// web/src/index.tsx
import { render }      from 'solid-js/web';
import { lazy }        from 'solid-js';
import { HashRouter, Route } from '@solidjs/router';

// Fuentes self-hosted (Geist variable: un import cubre todo el eje de pesos)
import '@fontsource-variable/geist';
import '@fontsource-variable/geist-mono';
import './styles/tokens.css'; // tokens dual-theme

// Rutas lazy — chunks separados de Vite
const Hub       = lazy(() => import('./routes/hub'));
const Dashboard = lazy(() => import('./routes/dashboard'));
const Engine    = lazy(() => import('./routes/engine'));
const Labelling = lazy(() => import('./routes/labelling'));
const Guia      = lazy(() => import('./routes/guia'));
const Control   = lazy(() => import('./routes/control'));

function App() {
  return (
    <HashRouter>
      <Route path="/"           component={Hub} />
      <Route path="/dashboard"  component={Dashboard} />
      <Route path="/engine"     component={Engine} />
      <Route path="/labelling"  component={Labelling} />
      <Route path="/guia"       component={Guia} />
      <Route path="/control"    component={Control} />
    </HashRouter>
  );
}

render(App, document.getElementById('root')!);

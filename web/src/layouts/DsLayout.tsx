import React from 'react';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import TopBar, { type TopBarTab, type TopBarUser } from '@/components/ds/chrome/TopBar';
import { useAuthStore } from '@/stores/authStore';
import { useDsPreviewTheme } from '@/hooks/useDsPreviewTheme';

/** Kit fallback identity when no authenticated user is present in the preview. */
const FALLBACK_USER: TopBarUser = { name: 'Juniper', rating: '1520 ± 140' };

/**
 * Chrome shell for the platform DS preview screens (cambia-438).
 *
 * Additive: renders the DS TopBar plus an <Outlet> for /ds/home, /ds/lobby,
 * /ds/game and /ds/leaderboard, on a design-system surface independent of the
 * app's Tailwind AppLayout. The design tokens key light mode off a
 * `data-theme="light"` attribute (design-system-import/tokens/colors.css), so
 * this layout mirrors the real uiStore theme onto that attribute while a DS
 * route is mounted and clears it on unmount.
 */
const DsLayout: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const authUser = useAuthStore((state) => state.user);
  const { light, toggle } = useDsPreviewTheme();

  const active: TopBarTab = location.pathname.startsWith('/ds/leaderboard') ? 'leaderboard' : 'home';

  const onNav = (tab: TopBarTab) => {
    navigate(tab === 'leaderboard' ? '/ds/leaderboard' : '/ds/home');
  };

  const user: TopBarUser = authUser
    ? {
      name: authUser.username || FALLBACK_USER.name,
      rating: authUser.elo !== undefined ? `${Math.round(authUser.elo)} ± ${Math.round(authUser.rd ?? 0)}` : FALLBACK_USER.rating
    }
    : FALLBACK_USER;

  return (
    <div
      style={{
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        background: 'var(--surface-page)',
        color: 'var(--text-primary)',
        fontFamily: 'var(--font-ui)'
      }}
    >
      <TopBar active={active} onNav={onNav} light={light} onToggleTheme={toggle} user={user} />
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
        <Outlet />
      </div>
    </div>
  );
};

export default DsLayout;

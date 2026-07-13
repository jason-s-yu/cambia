// Shared chrome for the Cambia platform kit.
const { Button, IconButton, Badge, Switch } = window.Cambia_cc8727;

function Icon({ name, size = 17, style }) {
  return <i data-lucide={name} style={{ width: size, height: size, display: 'inline-block', flex: 'none', ...style }}></i>;
}

function Wordmark({ size = 26 }) {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'baseline', gap: 7 }}>
      <span style={{ fontFamily: 'var(--font-display)', fontSize: size, lineHeight: 1 }}>Cambia</span>
      <span style={{ color: 'var(--berry-400)', fontSize: size * 0.6 }}>♥</span>
    </span>
  );
}

function TopBar({ active, onNav, light, onToggleTheme }) {
  const tabs = [['home', 'Play'], ['leaderboard', 'Leaderboard']];
  return (
    <div style={{
      height: 'var(--topbar-h)', display: 'flex', alignItems: 'center', gap: 22,
      padding: '0 22px', background: 'var(--surface-card)',
      borderBottom: '1.5px solid var(--border-default)', flex: 'none',
    }}>
      <a onClick={() => onNav('home')} style={{ cursor: 'pointer', color: 'var(--text-primary)', textDecoration: 'none' }}><Wordmark /></a>
      <nav style={{ display: 'flex', gap: 4 }}>
        {tabs.map(([id, label]) => (
          <button key={id} onClick={() => onNav(id)} style={{
            padding: '7px 14px', fontFamily: 'var(--font-ui)', fontSize: 'var(--text-md)',
            fontWeight: 700, cursor: 'pointer', borderRadius: 'var(--radius-md)',
            background: active === id ? 'var(--surface-raised)' : 'transparent',
            color: active === id ? 'var(--text-primary)' : 'var(--text-secondary)',
            border: active === id ? '1.5px solid var(--border-strong)' : '1.5px solid transparent',
          }}>{label}</button>
        ))}
      </nav>
      <div style={{ flex: 1 }}></div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--text-secondary)', fontSize: 'var(--text-xs)', fontWeight: 700 }}>
        <span>☾</span>
        <Switch checked={light} onChange={onToggleTheme} />
        <span>☀</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 9, padding: '5px 12px 5px 6px', background: 'var(--surface-raised)', border: '1.5px solid var(--border-default)', borderRadius: 'var(--radius-pill)' }}>
        <span style={{ width: 26, height: 26, borderRadius: '50%', background: 'var(--dusk-500)', border: '1.5px solid var(--outline-ink)', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', fontWeight: 800, fontSize: 12, color: 'var(--text-on-ember)' }}>J</span>
        <span style={{ lineHeight: 1.15 }}>
          <span style={{ display: 'block', fontWeight: 700, fontSize: 'var(--text-sm)' }}>Juniper</span>
          <span style={{ display: 'block', fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-tertiary)' }}>1520 ± 140</span>
        </span>
      </div>
    </div>
  );
}

function Panel({ title, action, children, style }) {
  return (
    <section style={{ background: 'var(--surface-card)', border: '1.5px solid var(--border-default)', borderRadius: 'var(--radius-lg)', boxShadow: 'var(--shadow-card)', padding: '16px 18px', ...style }}>
      {(title || action) && (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
          <h3 style={{ margin: 0, fontSize: 'var(--text-2xs)', fontWeight: 800, letterSpacing: 'var(--tracking-caps)', textTransform: 'uppercase', color: 'var(--text-tertiary)' }}>{title}</h3>
          {action}
        </div>
      )}
      {children}
    </section>
  );
}

Object.assign(window, { Icon, Wordmark, TopBar, Panel });

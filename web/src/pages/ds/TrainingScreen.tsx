import React, { useState } from 'react';
import Button from '@/components/ds/core/Button';
import Badge from '@/components/ds/core/Badge';
import StatusBadge, { type Status } from '@/components/ds/data/StatusBadge';
import StatRow from '@/components/ds/data/StatRow';
import Panel from '@/components/ds/chrome/Panel';
import { useDsPreviewTheme } from '@/hooks/useDsPreviewTheme';

interface TrainingRun {
  name: string;
  algorithm: string;
  status: Status;
  best: number | null;
  iter: number | null;
  host: string | null;
}

interface CheckpointRow {
  iter: number;
  size: string;
  when: string;
  best: boolean;
}

interface EvalRow {
  id: string;
  target: string;
  games: number;
  status: Status;
  device: string;
}

const RUNS: TrainingRun[] = [
  { name: 'prtcfr-prod-v4', algorithm: 'prtcfr', status: 'running', best: 0.687, iter: 3200, host: null },
  { name: 'escher-e34-sweep', algorithm: 'escher', status: 'running', best: 0.641, iter: 2100, host: 'runner-01' },
  { name: 'deep-train-baseline', algorithm: 'deep_cfr', status: 'stopped', best: 0.612, iter: 4800, host: null },
  { name: 'ppo-selfplay-e2', algorithm: 'ppo', status: 'crashed', best: 0.554, iter: 900, host: null },
  { name: 'sog-v4-rebuild', algorithm: 'sog', status: 'created', best: null, iter: null, host: 'runner-01' }
];

const CHECKPOINTS: CheckpointRow[] = [
  { iter: 3200, size: '412 MB', when: '14 min ago', best: true },
  { iter: 3000, size: '412 MB', when: '1.2 h ago', best: false },
  { iter: 2800, size: '411 MB', when: '2.3 h ago', best: false }
];

const EVALS: EvalRow[] = [
  { id: 'ev-2291', target: 'greedy', games: 2000, status: 'running', device: 'cuda' },
  { id: 'ev-2290', target: 'random', games: 2000, status: 'succeeded', device: 'cuda' },
  { id: 'ev-2287', target: 'ckpt-2800', games: 1000, status: 'failed', device: 'cpu' }
];

const LOG_LINES: string[] = [
  '[03:12:44] iter 3199 | adv_loss 0.0181 | strat_loss 0.0324 | buf 12.4M',
  '[03:12:59] iter 3200 | adv_loss 0.0179 | strat_loss 0.0321 | buf 12.4M',
  '[03:13:02] checkpoint saved: runs/prtcfr-prod-v4/ckpt_3200.pt',
  '[03:13:04] eval_watcher: queued eval vs greedy (2000 games, cuda)',
  '[03:13:18] iter 3201 | adv_loss 0.0180 | strat_loss 0.0319 | buf 12.5M'
];

// Win-rate curve data: [iteration, win rate vs baseline].
const CURVE: Array<[number, number]> = [[0, 0.31], [400, 0.44], [800, 0.50], [1200, 0.55], [1600, 0.58], [2000, 0.61], [2400, 0.635], [2800, 0.66], [3200, 0.687]];
const CURVE2: Array<[number, number]> = [[0, 0.30], [400, 0.41], [800, 0.47], [1200, 0.50], [1600, 0.54], [2000, 0.55], [2400, 0.57], [2800, 0.585], [3200, 0.60]];

const MetricsChart: React.FC = () => {
  const W = 560;
  const H = 180;
  const PL = 40;
  const PB = 24;
  const x = (it: number) => PL + (it / 3200) * (W - PL - 10);
  const y = (v: number) => (H - PB) - ((v - 0.25) / 0.5) * (H - PB - 10);
  const path = (pts: Array<[number, number]>) => pts.map((p, i) => (i ? 'L' : 'M') + x(p[0]).toFixed(1) + ',' + y(p[1]).toFixed(1)).join(' ');
  return (
    <svg width='100%' viewBox={'0 0 ' + W + ' ' + H} style={{ display: 'block' }}>
      {[0.3, 0.4, 0.5, 0.6, 0.7].map((v) => (
        <g key={v}>
          <line x1={PL} x2={W - 10} y1={y(v)} y2={y(v)} stroke='var(--border-subtle)' strokeWidth='1' />
          <text x={PL - 6} y={y(v) + 3} textAnchor='end' fontSize='9' fill='var(--text-tertiary)' fontFamily='var(--ds-font-mono)'>{Math.round(v * 100)}%</text>
        </g>
      ))}
      <line x1={PL} x2={W - 10} y1={y(0.5)} y2={y(0.5)} stroke='var(--border-strong)' strokeWidth='1' strokeDasharray='3 3' />
      <path d={path(CURVE2)} fill='none' stroke='var(--dusk-500)' strokeWidth='2' />
      <path d={path(CURVE)} fill='none' stroke='var(--ember-500)' strokeWidth='2.5' />
      <circle cx={x(3200)} cy={y(0.687)} r='4' fill='var(--ember-500)' stroke='var(--outline-ink)' strokeWidth='1.5' />
      {[0, 800, 1600, 2400, 3200].map((it) => (
        <text key={it} x={x(it)} y={H - 8} textAnchor='middle' fontSize='9' fill='var(--text-tertiary)' fontFamily='var(--ds-font-mono)'>{it}</text>
      ))}
    </svg>
  );
};

interface GpuBarProps {
  label: string;
  pct: number;
  detail: string;
}

const GpuBar: React.FC<GpuBarProps> = ({ label, pct, detail }) => {
  return (
    <div style={{ flex: 1, minWidth: 150 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 'var(--text-2xs)', marginBottom: 4 }}>
        <span style={{ fontWeight: 'var(--weight-black)', letterSpacing: 'var(--tracking-caps)', textTransform: 'uppercase', color: 'var(--text-tertiary)' }}>{label}</span>
        <span style={{ fontFamily: 'var(--ds-font-mono)', color: pct > 85 ? 'var(--berry-400)' : 'var(--text-secondary)' }}>{detail}</span>
      </div>
      <div style={{ height: 8, background: 'var(--surface-inset)', border: '1px solid var(--border-default)', borderRadius: 'var(--radius-pill)', overflow: 'hidden' }}>
        <div style={{ width: pct + '%', height: '100%', background: pct > 85 ? 'var(--berry-500)' : pct > 60 ? 'var(--honey-500)' : 'var(--moss-500)' }}></div>
      </div>
    </div>
  );
};

interface RunRowProps {
  r: TrainingRun;
  active: boolean;
  onClick: () => void;
}

const RunRow: React.FC<RunRowProps> = ({ r, active, onClick }) => {
  return (
    <button
      onClick={onClick}
      style={{
        display: 'block',
        width: '100%',
        textAlign: 'left',
        cursor: 'pointer',
        padding: '10px 12px',
        borderRadius: 'var(--ds-radius-md)',
        background: active ? 'var(--surface-raised)' : 'transparent',
        border: active ? '1.5px solid var(--border-strong)' : '1.5px solid transparent',
        fontFamily: 'var(--font-ui)',
        color: 'var(--text-primary)'
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ fontFamily: 'var(--ds-font-mono)', fontWeight: 'var(--weight-bold)', fontSize: 'var(--ds-text-sm)', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.name}</span>
        <StatusBadge status={r.status} />
      </div>
      <div style={{ display: 'flex', gap: 10, marginTop: 4, fontSize: 'var(--text-2xs)', color: 'var(--text-tertiary)', fontFamily: 'var(--ds-font-mono)' }}>
        <span>{r.algorithm}</span>
        {r.best !== null && <span>best {(r.best * 100).toFixed(1)}% @ {r.iter}</span>}
        {r.host && <span style={{ color: 'var(--dusk-400)' }}>&#8645; {r.host}</span>}
      </div>
    </button>
  );
};

/**
 * DS preview: training dashboard (cambia-438). Mirrors the runs / checkpoints
 * / evals / GPU-bars domain of the real training dashboard, which is a
 * separate app served by its own backend (trainingStore + REST /training/*).
 * Data here is typed mock matching the kit; wiring the real store would make
 * the preview depend on that backend, out of scope for a standalone preview.
 */
const TrainingScreen: React.FC = () => {
  useDsPreviewTheme();
  const [sel, setSel] = useState('prtcfr-prod-v4');
  const run = RUNS.find((r) => r.name === sel) || RUNS[0];
  const remote = !!run.host;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', background: 'var(--surface-page)', color: 'var(--text-primary)', fontFamily: 'var(--font-ui)' }}>
      <div style={{ height: 'var(--topbar-h)', display: 'flex', alignItems: 'center', gap: 14, padding: '0 22px', background: 'var(--surface-card)', borderBottom: '1.5px solid var(--border-default)', flex: 'none' }}>
        <span style={{ fontFamily: 'var(--font-display)', fontSize: 24 }}>Cambia</span>
        <Badge tone='ember'>TRAINING</Badge>
        <span style={{ fontSize: 'var(--ds-text-xs)', color: 'var(--text-tertiary)' }}>admin · local hardware</span>
        <div style={{ flex: 1 }}></div>
        <Badge tone='success' dot>runnerd connected</Badge>
        <Button size='sm'>New run</Button>
      </div>

      <div style={{ display: 'flex', gap: 20, padding: '12px 22px', background: 'var(--surface-inset)', borderBottom: '1.5px solid var(--border-subtle)', flexWrap: 'wrap' }}>
        <GpuBar label='GPU 0 · RTX 4090' pct={92} detail='92% · 21.3/24 GB · 71°C' />
        <GpuBar label='GPU 1 · RTX 4090' pct={38} detail='38% · 9.8/24 GB · 54°C' />
        <GpuBar label='CPU' pct={64} detail='64% · load 18.2' />
        <GpuBar label='Disk' pct={47} detail='1.9/4.0 TB' />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr', gap: 18, padding: 18, flex: 1, alignItems: 'start', maxWidth: 1360, width: '100%', margin: '0 auto' }}>
        <Panel title={'Runs · ' + RUNS.length} style={{ padding: '12px 10px' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            {RUNS.map((r) => <RunRow key={r.name} r={r} active={r.name === sel} onClick={() => setSel(r.name)} />)}
          </div>
        </Panel>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 16, minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
            <h1 style={{ margin: 0, fontFamily: 'var(--ds-font-mono)', fontWeight: 'var(--weight-bold)', fontSize: 'var(--ds-text-xl)' }}>{run.name}</h1>
            <StatusBadge status={run.status} />
            {remote && <Badge tone='info'>remote · {run.host} · synced 40s ago</Badge>}
            <div style={{ flex: 1 }}></div>
            {run.status === 'running'
              ? <Button size='sm' variant='cambia' disabled={remote}>Stop</Button>
              : <Button size='sm' disabled={remote}>{run.status === 'stopped' || run.status === 'crashed' ? 'Resume' : 'Start'}</Button>}
            <Button size='sm' variant='secondary'>Eval now</Button>
            {remote && <span style={{ fontSize: 'var(--text-2xs)', color: 'var(--text-tertiary)' }}>remote runs are read-only</span>}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: 16, alignItems: 'start' }}>
            <Panel title='Win rate vs baselines' action={<span style={{ fontSize: 'var(--text-2xs)', fontFamily: 'var(--ds-font-mono)' }}><span style={{ color: 'var(--ember-400)' }}>■ vs greedy</span>&nbsp;&nbsp;<span style={{ color: 'var(--dusk-400)' }}>■ vs random-plus</span></span>}>
              <MetricsChart />
            </Panel>
            <Panel title='Summary'>
              <StatRow label='Algorithm' value={run.algorithm} />
              <StatRow label='Best win rate' value={run.best !== null ? (run.best * 100).toFixed(1) : '—'} unit='%' delta={run.best ? '+2.1' : undefined} />
              <StatRow label='Best iteration' value={run.iter ?? '—'} />
              <StatRow label='Buffer' value='12.5M' unit='samples' />
              <StatRow label='Wall clock' value='31.4' unit='h' />
              <StatRow label='Config' value='prtcfr_production.yaml' />
            </Panel>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, alignItems: 'start' }}>
            <Panel title='Checkpoints'>
              {CHECKPOINTS.map((c, i) => (
                <div key={c.iter} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '8px 2px', borderTop: i ? '1px solid var(--border-subtle)' : 'none' }}>
                  <span style={{ fontFamily: 'var(--ds-font-mono)', fontWeight: 'var(--weight-bold)' }}>ckpt_{c.iter}</span>
                  {c.best && <Badge tone='gold'>BEST</Badge>}
                  <span style={{ flex: 1 }}></span>
                  <span style={{ fontFamily: 'var(--ds-font-mono)', fontSize: 'var(--ds-text-xs)', color: 'var(--text-secondary)' }}>{c.size}</span>
                  <span style={{ fontSize: 'var(--ds-text-xs)', color: 'var(--text-tertiary)', width: 70, textAlign: 'right' }}>{c.when}</span>
                </div>
              ))}
            </Panel>
            <Panel title='Eval jobs'>
              {EVALS.map((e, i) => (
                <div key={e.id} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '8px 2px', borderTop: i ? '1px solid var(--border-subtle)' : 'none' }}>
                  <span style={{ fontFamily: 'var(--ds-font-mono)', fontSize: 'var(--ds-text-sm)', fontWeight: 'var(--weight-bold)' }}>{e.id}</span>
                  <span style={{ fontSize: 'var(--ds-text-xs)', color: 'var(--text-secondary)', flex: 1 }}>vs {e.target} · {e.games} games · {e.device}</span>
                  <StatusBadge status={e.status} />
                </div>
              ))}
            </Panel>
          </div>

          <Panel title='Live log · runs/prtcfr-prod-v4/train.log'>
            <div style={{ background: 'var(--surface-inset)', border: '1.5px solid var(--border-subtle)', borderRadius: 'var(--ds-radius-md)', padding: '10px 14px', fontFamily: 'var(--ds-font-mono)', fontSize: 'var(--ds-text-xs)', lineHeight: 1.7, color: 'var(--text-secondary)', overflowX: 'auto', whiteSpace: 'pre' }}>
              {LOG_LINES.join('\n')}
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
};

export default TrainingScreen;

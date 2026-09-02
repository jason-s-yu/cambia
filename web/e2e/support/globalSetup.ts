// web/e2e/support/globalSetup.ts
// Refuses to start the pass when the game service is not up, or is up without dev accounts.
//
// Both checks fail the whole run at once with the command that fixes them. Without this the suite
// starts, every seat lands on a login page it cannot pass, and each test reports a different
// timeout for one missing process.
import { API_URL } from '../playwright.config';

const LAUNCH = 'tools/e2e/launch-dev-service.sh (from the repository root)';

async function getJson(path: string): Promise<{ status: number; body: unknown }> {
  const res = await fetch(`${API_URL}${path}`);
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    body = null;
  }
  return { status: res.status, body };
}

export default async function globalSetup(): Promise<void> {
  let health: { status: number; body: unknown };
  try {
    health = await getJson('/healthz');
  } catch (err) {
    throw new Error(
      `No game service on ${API_URL}: ${(err as Error).message}\nStart one with ${LAUNCH}`
    );
  }
  const probe = health.body as { status?: string; db?: boolean } | null;
  if (health.status !== 200 || probe?.status !== 'ok') {
    throw new Error(`${API_URL}/healthz answered ${health.status} ${JSON.stringify(probe)}\nStart one with ${LAUNCH}`);
  }
  if (probe?.db !== true) {
    throw new Error(
      `${API_URL} is up but has no database. Seats cannot be minted and no game can be recorded.\nCheck the Postgres container and relaunch with ${LAUNCH}`
    );
  }

  const dev = await getJson('/dev/session');
  const devBody = dev.body as { enabled?: boolean } | null;
  if (dev.status !== 200 || devBody?.enabled !== true) {
    throw new Error(
      `${API_URL}/dev/session answered ${dev.status}: the service is running without CAMBIA_DEV_ACCOUNTS=1, so ?as=<name> cannot pin a seat.\nRelaunch with ${LAUNCH}`
    );
  }
}

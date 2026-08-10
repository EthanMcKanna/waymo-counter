import { Container, getContainer } from '@cloudflare/containers'

export class WaymoCounter extends Container { sleepAfter = '5m' }

function startJob(env) {
  return getContainer(env.WAYMO_COUNTER, 'scheduled').start({
    entrypoint: ['python', '-u', '-m', 'src.main'], enableInternet: true,
    envVars: {
      ROBOTAXI_INGEST_URL: env.ROBOTAXI_INGEST_URL, ROBOTAXI_INGEST_SECRET: env.ROBOTAXI_INGEST_SECRET,
      MODEL_URL: env.MODEL_URL, MODEL_IMAGE_SIZE: env.MODEL_IMAGE_SIZE, CONFIDENCE_THRESHOLD: env.CONFIDENCE_THRESHOLD,
      VERIFIER_ENABLED: env.VERIFIER_ENABLED, VERIFIER_MODEL_URL: env.VERIFIER_MODEL_URL,
      VERIFIER_IMAGE_SIZE: env.VERIFIER_IMAGE_SIZE, VERIFIER_CROP_PADDING: env.VERIFIER_CROP_PADDING,
      VERIFIER_CALIBRATION_ENABLED: env.VERIFIER_CALIBRATION_ENABLED, VERIFIER_THRESHOLD: env.VERIFIER_THRESHOLD,
      VERIFIER_NON_AUSTIN_THRESHOLD: env.VERIFIER_NON_AUSTIN_THRESHOLD, FETCH_WORKERS: env.FETCH_WORKERS,
      SCAN_LOCK_MINUTES: env.SCAN_LOCK_MINUTES, SCAN_SCOPE: env.SCAN_SCOPE, ENABLED_MARKETS: env.ENABLED_MARKETS,
    },
  })
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url)
    if (request.method === 'GET' && url.pathname === '/health') return Response.json({ ok: true, producer: 'waymo-counter', runtime: 'cloudflare-container', configured: Boolean(env.ROBOTAXI_INGEST_SECRET && env.RUN_TOKEN) })
    if (request.method === 'POST' && url.pathname === '/run') {
      const token = request.headers.get('authorization')?.replace(/^Bearer\s+/i, '')
      if (!env.RUN_TOKEN || token !== env.RUN_TOKEN) return new Response('Unauthorized', { status: 401 })
      await startJob(env); return Response.json({ accepted: true }, { status: 202 })
    }
    return new Response('Not found', { status: 404 })
  },
  async scheduled(_event, env, ctx) { ctx.waitUntil(startJob(env)) },
}

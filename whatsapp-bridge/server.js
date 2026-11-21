const express = require('express');
const axios = require('axios');
const qrcode = require('qrcode');
const pino = require('pino');
const path = require('path');
const fs = require('fs');
const {
  default: makeWASocket,
  useMultiFileAuthState,
  DisconnectReason,
  fetchLatestBaileysVersion,
} = require('@whiskeysockets/baileys');

const PORT = process.env.PORT || 3001;
const API_BACKEND_URL = process.env.API_BACKEND_URL || 'http://api:8000';
const AUTH_DIR = path.resolve(__dirname, 'auth_info_baileys');

const app = express();
app.use(express.json());

// CORS for local web UI
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  if (req.method === 'OPTIONS') return res.sendStatus(200);
  next();
});

let sock = null;
let isConnected = false;
let currentQRDataUrl = null;
let currentRawQR = null;
let userJid = null;
let reconnectAttempts = 0;

async function startWhatsApp() {
  if (!fs.existsSync(AUTH_DIR)) {
    fs.mkdirSync(AUTH_DIR, { recursive: true });
  }

  const { state, saveCreds } = await useMultiFileAuthState(AUTH_DIR);
  const { version, isLatest } = await fetchLatestBaileysVersion();
  console.log(`[WhatsApp Bridge] Using Baileys v${version.join('.')}, isLatest: ${isLatest}`);

  sock = makeWASocket({
    version,
    auth: state,
    printQRInTerminal: true,
    logger: pino({ level: 'silent' }),
    browser: ['DeltaBot 2.0', 'Desktop', '2.0.0'],
    connectTimeoutMs: 60000,
    keepAliveIntervalMs: 25000,
    emitOwnEvents: false,
    generateHighQualityLinkPreview: false,
  });

  sock.ev.on('creds.update', saveCreds);

  sock.ev.on('connection.update', async (update) => {
    const { connection, lastDisconnect, qr } = update;

    if (qr) {
      currentRawQR = qr;
      try {
        currentQRDataUrl = await qrcode.toDataURL(qr, { scale: 8, margin: 2 });
        console.log('[WhatsApp Bridge] New QR code generated. Scan in browser at http://localhost:3001/qr or terminal.');
      } catch (err) {
        console.error('[WhatsApp Bridge] Error creating QR image:', err);
      }
    }

    if (connection === 'close') {
      const statusCode = lastDisconnect?.error?.output?.statusCode;
      const shouldReconnect = statusCode !== DisconnectReason.loggedOut;
      isConnected = false;
      currentQRDataUrl = null;
      currentRawQR = null;
      userJid = null;

      console.log(`[WhatsApp Bridge] Connection closed (status: ${statusCode}). Reconnecting: ${shouldReconnect}`);

      if (statusCode === DisconnectReason.loggedOut) {
        console.log('[WhatsApp Bridge] Device logged out. Cleaning auth state...');
        try {
          fs.rmSync(AUTH_DIR, { recursive: true, force: true });
        } catch (e) {
          console.error('[WhatsApp Bridge] Error clearing auth dir:', e);
        }
      }

      if (shouldReconnect) {
        const delay = Math.min(3000 * Math.pow(1.5, reconnectAttempts), 30000);
        reconnectAttempts++;
        console.log(`[WhatsApp Bridge] Reconnecting in ${Math.round(delay / 1000)}s... (attempt #${reconnectAttempts})`);
        setTimeout(startWhatsApp, delay);
      } else {
        reconnectAttempts = 0;
        setTimeout(startWhatsApp, 3000);
      }
    } else if (connection === 'open') {
      isConnected = true;
      reconnectAttempts = 0;
      currentQRDataUrl = null;
      currentRawQR = null;
      userJid = sock.user?.id || null;
      const phoneNumber = userJid ? userJid.split(':')[0] : 'unknown';
      console.log(`[WhatsApp Bridge] Connected successfully! Linked WhatsApp user: +${phoneNumber}`);

      // Notify FastAPI backend
      try {
        await axios.post(`${API_BACKEND_URL}/api/v1/whatsapp/status`, {
          connected: true,
          phone: phoneNumber,
          jid: userJid,
        }, { timeout: 5000 });
      } catch (err) {
        console.log('[WhatsApp Bridge] Backend notified status, note:', err.message);
      }
    }
  });

  // Handle incoming messages from user
  sock.ev.on('messages.upsert', async ({ messages, type }) => {
    if (type !== 'notify') return;

    for (const msg of messages) {
      if (!msg.message || msg.key.fromMe) continue;

      const remoteJid = msg.key.remoteJid;
      if (!remoteJid || remoteJid.includes('@g.us') || remoteJid === 'status@broadcast') {
        continue; // Only process direct 1-on-1 messages
      }

      const text =
        msg.message.conversation ||
        msg.message.extendedTextMessage?.text ||
        msg.message.buttonsResponseMessage?.selectedButtonId ||
        msg.message.templateButtonReplyMessage?.selectedId ||
        '';

      const cleanText = text.trim();
      if (!cleanText) continue;

      const senderPhone = remoteJid.split('@')[0];
      console.log(`[WhatsApp Bridge] Inbound message from +${senderPhone}: "${cleanText}"`);

      // Forward to FastAPI backend
      try {
        const res = await axios.post(
          `${API_BACKEND_URL}/api/v1/whatsapp/incoming`,
          {
            from_jid: remoteJid,
            phone: senderPhone,
            text: cleanText,
            timestamp: msg.messageTimestamp,
          },
          { timeout: 15000 }
        );

        if (res.data && res.data.reply) {
          await sock.sendMessage(remoteJid, { text: res.data.reply });
          console.log(`[WhatsApp Bridge] Sent reply to +${senderPhone}: "${res.data.reply.slice(0, 60)}..."`);
        }
      } catch (err) {
        console.error('[WhatsApp Bridge] Failed forwarding to backend:', err.message);
        if (err.response && err.response.data && err.response.data.detail) {
          await sock.sendMessage(remoteJid, { text: `⚠️ Error: ${err.response.data.detail}` });
        }
      }
    }
  });
}

// REST Endpoints
app.get('/status', (req, res) => {
  const phone = userJid ? userJid.split(':')[0] : null;
  res.json({
    connected: isConnected,
    phone: phone,
    has_qr: !!currentQRDataUrl,
    reconnect_attempts: reconnectAttempts,
  });
});

app.get('/qr', (req, res) => {
  if (isConnected) {
    const phone = userJid ? userJid.split(':')[0] : 'Linked Number';
    return res.send(`
      <!DOCTYPE html>
      <html>
      <head>
        <title>DeltaBot 2.0 WhatsApp Status</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
          body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; display: flex; align-items: center; justify-content: center; min-height: 100vh; margin: 0; }
          .card { background: #1e293b; border: 1px solid #334155; padding: 2rem; border-radius: 1rem; text-align: center; max-width: 420px; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5); }
          .badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 9999px; background: #065f46; color: #34d399; font-weight: 600; font-size: 0.875rem; margin-bottom: 1rem; }
          h1 { margin: 0 0 0.5rem 0; font-size: 1.5rem; }
          p { color: #94a3b8; font-size: 0.875rem; line-height: 1.5; }
          .phone { font-family: monospace; font-size: 1.25rem; color: #38bdf8; font-weight: bold; margin: 1rem 0; }
        </style>
      </head>
      <body>
        <div class="card">
          <div class="badge">● CONNECTED</div>
          <h1>WhatsApp Linked</h1>
          <p>DeltaBot 2.0 is actively linked to:</p>
          <div class="phone">+${phone}</div>
          <p>Trade alerts will be sent here. You can reply with <b>1</b> to place/override trades, or <b>2</b> to reject them.</p>
        </div>
      </body>
      </html>
    `);
  }

  if (currentQRDataUrl) {
    return res.send(`
      <!DOCTYPE html>
      <html>
      <head>
        <title>Scan WhatsApp QR — DeltaBot 2.0</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
          body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; display: flex; align-items: center; justify-content: center; min-height: 100vh; margin: 0; }
          .card { background: #1e293b; border: 1px solid #334155; padding: 2rem; border-radius: 1rem; text-align: center; max-width: 440px; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5); }
          h1 { margin: 0 0 0.5rem 0; font-size: 1.5rem; }
          p { color: #94a3b8; font-size: 0.875rem; line-height: 1.5; margin-bottom: 1.5rem; }
          .qr-box { background: white; padding: 1rem; border-radius: 0.75rem; display: inline-block; margin-bottom: 1rem; }
          .qr-box img { display: block; width: 280px; height: 280px; }
          ol { text-align: left; color: #cbd5e1; font-size: 0.8125rem; padding-left: 1.5rem; margin: 1rem 0; line-height: 1.6; }
          .timer { color: #f59e0b; font-size: 0.75rem; font-family: monospace; }
        </style>
        <script>
          // Auto reload to refresh QR or transition to Connected
          setInterval(async () => {
            try {
              const res = await fetch('/status');
              const data = await res.json();
              if (data.connected) {
                location.reload();
              }
            } catch(e) {}
          }, 3000);
        </script>
      </head>
      <body>
        <div class="card">
          <h1>Link WhatsApp Bot</h1>
          <p>Scan this QR code with WhatsApp on your phone to authorize DeltaBot 2.0 trade alerts.</p>
          <div class="qr-box">
            <img src="${currentQRDataUrl}" alt="WhatsApp QR Code" />
          </div>
          <ol>
            <li>Open <b>WhatsApp</b> on your phone.</li>
            <li>Tap <b>Menu (⋮)</b> or <b>Settings</b> &gt; <b>Linked Devices</b>.</li>
            <li>Tap <b>Link a Device</b> and point camera at this screen.</li>
          </ol>
          <div class="timer">QR refreshes automatically...</div>
        </div>
      </body>
      </html>
    `);
  }

  res.send(`
    <!DOCTYPE html>
    <html>
    <head>
      <title>DeltaBot 2.0 WhatsApp Initializing</title>
      <meta http-equiv="refresh" content="3">
      <style>
        body { font-family: sans-serif; background: #0f172a; color: white; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }
        .box { text-align: center; }
      </style>
    </head>
    <body>
      <div class="box">
        <h2>Initializing WhatsApp Web connection...</h2>
        <p>Please wait a few seconds. This page refreshes automatically.</p>
      </div>
    </body>
    </html>
  `);
});

app.post('/send', async (req, res) => {
  const { to, text } = req.body;
  if (!text) {
    return res.status(400).json({ error: 'Text is required' });
  }

  if (!isConnected || !sock) {
    return res.status(503).json({ error: 'WhatsApp is not connected. Scan QR first at /qr.' });
  }

  // Format recipient JID
  let recipient = to;
  if (!recipient) {
    // If no specific 'to' provided, default to the linked user itself (send message to self/notes)
    recipient = userJid;
  }

  if (!recipient) {
    return res.status(400).json({ error: 'Recipient phone number is required' });
  }

  if (!recipient.includes('@s.whatsapp.net')) {
    const cleanNumber = recipient.replace(/[^0-9]/g, '');
    recipient = `${cleanNumber}@s.whatsapp.net`;
  }

  try {
    const result = await sock.sendMessage(recipient, { text });
    console.log(`[WhatsApp Bridge] Outbound message sent to ${recipient}: ${text.slice(0, 50)}...`);
    res.json({ success: true, message_id: result.key?.id });
  } catch (err) {
    console.error('[WhatsApp Bridge] Error sending message:', err);
    res.status(500).json({ error: err.message });
  }
});

// Start Express server and WhatsApp client
app.listen(PORT, '0.0.0.0', () => {
  console.log(`[WhatsApp Bridge] HTTP Server listening on port ${PORT}`);
  startWhatsApp().catch((err) => {
    console.error('[WhatsApp Bridge] Failed starting Baileys client:', err);
  });
});

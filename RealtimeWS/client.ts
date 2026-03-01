/**
 * RealtimeWS -- TypeScript WebSocket client with auto-reconnect.
 * Pairs with server.py relay hub.
 */

export interface RelayMessage { from: string; room: string; payload: any; ts: number; }

export interface WSClientOptions {
  url: string; room: string; clientId: string;
  reconnectMs?: number; maxRetries?: number;
  onMessage?: (msg: RelayMessage) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
}

export class RealtimeWSClient {
  private ws: WebSocket | null = null;
  private retries = 0;
  private closed = false;
  private opts: Required<WSClientOptions>;

  constructor(opts: WSClientOptions) {
    this.opts = { reconnectMs: 2000, maxRetries: 10, onMessage: () => {}, onConnect: () => {}, onDisconnect: () => {}, ...opts };
  }

  connect(): void {
    this.closed = false;
    const wsUrl = `${this.opts.url}/ws/${this.opts.room}/${this.opts.clientId}`;
    this.ws = new WebSocket(wsUrl);
    this.ws.onopen = () => { this.retries = 0; this.opts.onConnect(); };
    this.ws.onmessage = (e) => {
      try { this.opts.onMessage(JSON.parse(e.data)); }
      catch { this.opts.onMessage({ from: "", room: "", payload: e.data, ts: Date.now() / 1000 }); }
    };
    this.ws.onclose = () => {
      this.opts.onDisconnect();
      if (!this.closed && this.retries < this.opts.maxRetries) {
        this.retries++;
        setTimeout(() => this.connect(), this.opts.reconnectMs * this.retries);
      }
    };
  }

  send(payload: any): void {
    if (this.ws?.readyState === WebSocket.OPEN)
      this.ws.send(typeof payload === "string" ? payload : JSON.stringify(payload));
  }
  disconnect(): void { this.closed = true; this.ws?.close(); }
  get connected(): boolean { return this.ws?.readyState === WebSocket.OPEN; }
}
